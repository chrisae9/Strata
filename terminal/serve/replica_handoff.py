"""Process-private, bounded session checkpoints for opt-in text handoff between unbatched replicas."""
from __future__ import annotations

import collections
import contextlib
import tempfile
import threading
import time
import uuid
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from serve.server_parts.engines import SessionRefused


@dataclass(eq=False)
class Checkpoint:
    path: Path
    prefix: array
    size: int
    local: set[int] = field(default_factory=set)
    readers: int = 0


class Handoff:
    def __init__(self, directory, budget_mib, replicas, min_tokens=8192):
        if budget_mib < 1 or min_tokens < 1:
            raise ValueError("replica handoff budget and minimum tokens must be positive")
        self.directory = tempfile.TemporaryDirectory(prefix="strata-handoff-", dir=directory)
        self.budget = budget_mib << 20
        self.minimum = min_tokens
        self.entries = collections.OrderedDict()
        self.size = 0
        self.lock = threading.Lock()
        # None is a fresh engine. False marks state that must survive a restore (e.g. vision).
        self.live = [None] * replicas

    @contextlib.contextmanager
    def match(self, ids):
        tokens = memoryview(array("i", ids))
        with self.lock:
            entry = max((e for e in reversed(self.entries.values())
                         if len(e.prefix) <= len(ids) and tokens[:len(e.prefix)] == memoryview(e.prefix)),
                        key=lambda e: len(e.prefix), default=None)
            if entry:
                entry.readers += 1
                self.entries.move_to_end(entry.path)
        try:
            yield entry
        finally:
            if entry:
                with self.lock:
                    entry.readers -= 1

    def save(self, k, engine, ids):
        """Publish only a completed native SAVE. The engine checks actual token identity again on GEN."""
        path = Path(self.directory.name) / uuid.uuid4().hex
        start = time.monotonic()
        try:
            info = engine.session_file("save", str(path))
            # A stop token may be returned but never committed. The tail can also be
            # re-rendered by the client. Index the prompt itself, allowing one edited tail block.
            anchor = min(len(ids), info["tokens"]) - 64
            if anchor < self.minimum or info["bytes"] > self.budget:
                return
            entry = Checkpoint(path, array("i", ids[:anchor]), path.stat().st_size, {k})
            with self.lock:
                # Readers own a lease through RESTORE. Admission cannot evict their files.
                for old in list(self.entries.values()):
                    if self.size + entry.size <= self.budget:
                        break
                    if not old.readers:
                        old.path.unlink()
                        self.size -= old.size
                        del self.entries[old.path]
                if self.size + entry.size > self.budget:
                    return
                self.entries[path] = entry
                self.size += entry.size
                self.live[k] = entry
            path = None
            print(f"[strata] handoff saved replica {k}: {info['tokens']} tokens, {entry.size} bytes, "
                  f"{time.monotonic()-start:.3f}s", flush=True)
        except (SessionRefused, OSError) as exc:
            print(f"[strata] replica {k} handoff save skipped: {exc}", flush=True)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def ready(self, entry, k):
        with self.lock:
            prev = self.live[k]
            return (k in entry.local or prev is None or
                    isinstance(prev, Checkpoint) and prev.path in self.entries)

    def restore(self, entry, k, engine):
        with self.lock:
            if k in entry.local:
                return
            prev = self.live[k]
            if prev is False or isinstance(prev, Checkpoint) and prev.path not in self.entries:
                return
            if prev:
                prev.readers += 1
        start = time.monotonic()
        try:
            info = engine.session_file("restore", str(entry.path))
            with self.lock:
                if prev:
                    prev.local.discard(k)
                self.live[k] = entry
                entry.local.add(k)
            print(f"[strata] handoff restored replica {k}: {info['tokens']} tokens in "
                  f"{time.monotonic()-start:.3f}s", flush=True)
        except (SessionRefused, OSError) as exc:
            print(f"[strata] replica {k} handoff restore skipped: {exc}", flush=True)
        finally:
            if prev:
                with self.lock:
                    prev.readers -= 1

    def forget_live(self, k, protected):
        with self.lock:
            self.live[k] = False if protected else None

    def invalidate(self, k):
        """A restarted replica can still import a saved file, even one it originally wrote."""
        with self.lock:
            self.live[k] = None
            for entry in self.entries.values():
                entry.local.discard(k)

    def close(self):
        self.directory.cleanup()
