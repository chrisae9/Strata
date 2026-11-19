"""Process-private, bounded session checkpoints for opt-in text handoff between unbatched replicas."""
from __future__ import annotations

import collections
import contextlib
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from array import array
from dataclasses import dataclass
from pathlib import Path

from serve.server_parts.engines import SessionRefused


@dataclass(eq=False)
class Checkpoint:
    path: Path
    prefix: array
    size: int
    readers: int = 0


def _lock_owner(file):
    if os.name == "nt":
        import msvcrt
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _sweep_abandoned(parent):
    # A process publishes ready only after locking owner, before it writes any snapshots.
    for path in parent.glob("strata-handoff-v2-*"):
        try:
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or (os.name == "posix" and
                    (info.st_uid != os.getuid() or info.st_mode & 0o077)):
                continue
            if not (path / "strata-handoff-owner-v2.ready").is_file():
                continue
            with (path / "owner").open("r+b") as owner:
                _lock_owner(owner)
            # The owner file is created once, never re-opened to take ownership. Multiple
            # sweepers can race here; neither can remove a live process's directory.
            shutil.rmtree(path)
        except OSError:
            continue


class Handoff:
    def __init__(self, directory, budget_mib, replicas, min_tokens=8192):
        if budget_mib < 1 or min_tokens < 1:
            raise ValueError("replica handoff budget and minimum tokens must be positive")
        parent = Path(directory).resolve()
        _sweep_abandoned(parent)
        self.directory = tempfile.TemporaryDirectory(prefix="strata-handoff-v2-", dir=parent)
        path = Path(self.directory.name)
        self.owner = (path / "owner").open("x+b")
        try:
            _lock_owner(self.owner)
            (path / "strata-handoff-owner-v2.ready").touch(exist_ok=False)
        except BaseException:
            self.owner.close()
            self.directory.cleanup()
            raise
        self.budget = budget_mib << 20
        self.minimum = min_tokens
        self.entries = collections.OrderedDict()
        self.size = 0
        self.epoch = 0
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
                    if not entry.readers and entry.path not in self.entries:
                        entry.path.unlink(missing_ok=True)

    def save(self, k, engine, ids, epoch):
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
            entry = Checkpoint(path, array("i", ids[:anchor]), path.stat().st_size)
            with self.lock:
                if epoch != self.epoch:
                    return
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
            return (prev is entry or prev is None or
                    isinstance(prev, Checkpoint) and prev.path in self.entries)

    def restore(self, entry, k, engine):
        with self.lock:
            prev = self.live[k]
            if prev is entry or entry.path not in self.entries:
                return
            if prev is False or isinstance(prev, Checkpoint) and prev.path not in self.entries:
                return
            if prev:
                prev.readers += 1
        start = time.monotonic()
        try:
            info = engine.session_file("restore", str(entry.path))
            with self.lock:
                if entry.path in self.entries:
                    self.live[k] = entry
            print(f"[strata] handoff restored replica {k}: {info['tokens']} tokens in "
                  f"{time.monotonic()-start:.3f}s", flush=True)
        except (SessionRefused, OSError) as exc:
            print(f"[strata] replica {k} handoff restore skipped: {exc}", flush=True)
        finally:
            if prev:
                with self.lock:
                    prev.readers -= 1
                    if not prev.readers and prev.path not in self.entries:
                        prev.path.unlink(missing_ok=True)

    def forget_live(self, k, protected):
        with self.lock:
            self.live[k] = False if protected else None

    def invalidate(self, k):
        """A restarted replica can still import a saved file, even one it originally wrote."""
        with self.lock:
            self.live[k] = None

    def discard_suspect(self):
        # A numerical failure has unknown ancestry. Withdraw all checkpoint candidates.
        # Active readers unlink their files on release; pre-flush GENs cannot republish.
        with self.lock:
            self.epoch += 1
            for entry in self.entries.values():
                if not entry.readers:
                    entry.path.unlink(missing_ok=True)
            self.entries.clear()
            self.size = 0

    def close(self):
        self.owner.close()
        self.directory.cleanup()
