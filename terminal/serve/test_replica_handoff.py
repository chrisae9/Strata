import pathlib
import tempfile
import subprocess
import sys
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from serve.replica_handoff import Handoff
from serve.replicas import ReplicaEngine
from serve.server_parts.engines import EngineDied, EngineNonFinite, QueueWaitExpired, SessionRefused


class Engine:
    batch = 0
    gen = 1
    def __init__(self):
        self.last = {}
        self.state = 0
        self.restores = []
        self.refuse = False

    def alive(self):
        return True

    def generate(self, ids, count, sampling, cancel, embeddings=None):
        self.state = ids[0]
        try:
            yield 17
        finally:
            self.last = {"generated": 1}

    def session_file(self, action, path):
        file = pathlib.Path(path)
        if action == "save":
            file.write_bytes(bytes([self.state]) * 128)
        else:
            if self.refuse:
                raise SessionRefused("invalid", "incompatible file")
            self.restores.append(file.read_bytes()[0])
        return {"tokens": 300, "bytes": 128}


class ReplicaHandoffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Handoff(self.tmp.name, 1, 2, min_tokens=64)
        self.addCleanup(self.store.close)
        self.a, self.b = Engine(), Engine()
        self.r = ReplicaEngine([self.a, self.b], handoff=self.store)
        self.stop = threading.Event()

    def turn(self, n, img=None):
        return list(self.r.generate([n]*256, 2, {}, self.stop, embeddings=img))

    def test_eos_close_publishes_and_receiver_can_return(self):
        g = self.r.generate([3]*256, 2, {}, self.stop)
        next(g)
        g.close()   # frontend consumed its stop token
        self.turn(4)
        # Both busy: arriving continuation must wake when B frees, even though A saw it.
        with self.r.available:
            self.r.inflight[:] = [1, 1]
        done = threading.Event()
        def follow():
            self.turn(3)
            done.set()
        thread = threading.Thread(target=follow)
        thread.start()
        with self.r.available:
            self.r.inflight[1] = 0
            self.r.available.notify_all()
        self.assertTrue(done.wait(3))
        thread.join()
        self.assertEqual(self.b.restores, [3])
        self.turn(4)  # B's displaced live conversation was also checkpointed
        self.assertEqual(self.b.restores, [3, 4])

    def test_cancellation_and_vision_never_publish(self):
        self.turn(3, img=[1])
        self.assertFalse(self.store.entries)
        g = self.r.generate([4]*256, 2, {}, self.stop)
        next(g)
        self.stop.set()
        g.close()
        self.assertFalse(self.store.entries)

    def test_screenshot_imports_only_a_previous_text_prefix(self):
        self.turn(3)
        self.r.inflight[0] = 1
        self.stop._strata_text_before_image = 256
        request = [3] * 256 + [9] * 32
        with patch.object(self.b, "generate", wraps=self.b.generate) as generate:
            self.assertEqual(list(self.r.generate(request, 2, {}, self.stop, embeddings="new.sve")), [17])
        self.assertEqual(self.b.restores, [3])
        self.assertEqual(generate.call_args.kwargs["embeddings"], "new.sve")
        self.assertEqual(len(self.store.entries), 1)  # image state was not published

    def test_matching_pad_ids_after_first_image_do_not_authorize_a_restore(self):
        self.turn(3)
        self.r.inflight[0] = 1
        self.stop._strata_text_before_image = 32
        self.turn(3, img="unrelated.sve")
        self.assertFalse(self.b.restores)

    def test_cold_screenshot_chooses_first_release_and_can_cancel(self):
        for cancel in (False, True):
            self.r.inflight[:] = [1, 1]
            self.stop.clear()
            waiting = threading.Event()
            real_wait = self.r.available.wait

            def wait(timeout):
                waiting.set()
                return real_wait(timeout)

            with patch.object(self.r.available, "wait", side_effect=wait), ThreadPoolExecutor(1) as pool:
                future = pool.submit(self.turn, 6 if cancel else 5, "new.sve")
                try:
                    self.assertTrue(waiting.wait(2))
                finally:
                    with self.r.available:
                        if cancel:
                            self.stop.set()
                        self.r.inflight[1] = 0
                        self.r.available.notify_all()
                self.assertEqual(future.result(timeout=2), [] if cancel else [17])
            self.assertEqual(self.r.inflight, [1, 0])
        self.assertEqual(self.b.state, 5)
        self.assertEqual(self.r.routed, [0, 1])

    def test_refused_restore_falls_back(self):
        self.turn(3)
        self.r.inflight[0] = 1
        self.b.refuse = True
        self.assertEqual(self.turn(3), [17])
        self.assertFalse(self.b.restores)

    def test_lost_local_cache_can_restore_its_own_file(self):
        self.turn(3)
        self.r._forget_replica(0)
        self.r.inflight[1] = 1
        self.turn(3)
        self.assertEqual(self.a.restores, [3])

    def test_failed_optional_save_preserves_completed_response(self):
        with patch.object(self.a, "session_file", side_effect=EngineDied("lost pipe")):
            self.assertEqual(self.turn(3), [17])
        self.assertTrue(self.a.ended)
        self.assertEqual(self.r.inflight, [0, 0])
        self.assertFalse(self.store.entries)

    def test_new_image_of_same_size_does_not_queue_on_false_native_hit(self):
        # A long suffix after different images must not count toward affinity.
        self.stop._strata_text_before_image = 32
        ids = [2] * 32 + [9] * 40 + [6] * 8000
        def picture(key):
            return list(self.r.generate(ids, 2, {"strata_img_keys": [key]}, self.stop, embeddings="img"))
        picture("aaaa")
        self.r.inflight[0] = 1
        self.assertEqual(picture("bbbb"), [17])
        self.assertEqual(self.r.routed, [1, 1])

    def test_continuation_can_evict_its_own_old_snapshot(self):
        self.store.budget = 128
        for n in (160, 200, 240):
            list(self.r.generate([3] * n, 2, {}, self.stop))
            entries = list(self.store.entries.values())
            self.assertEqual([len(e.prefix) for e in entries], [n - 64])

    def test_alternating_chats_restore_when_native_residency_is_unknown(self):
        self.r.inflight[1] = 1
        for chat in (3, 4, 3):
            self.turn(chat)
        self.assertEqual(self.a.restores, [3])

    def test_numerical_failure_bypasses_restore_on_retry(self):
        self.r.inflight[1] = 1
        self.turn(3)
        with self.store.match([3]*256) as leased:
            with patch.object(self.a, "generate", side_effect=EngineNonFinite("bad logits")):
                with self.assertRaises(EngineNonFinite):
                    self.turn(3)
            self.assertFalse(self.store.entries)
            self.assertTrue(leased.path.exists())
            self.store.restore(leased, 0, self.a)
        self.assertFalse(leased.path.exists())
        self.turn(3)
        self.assertEqual(self.a.restores, [])

    def test_availability_and_reserved_lock_respect_deadline(self):
        self.r.inflight[:] = [1, 1]
        self.stop._strata_queue_deadline = time.monotonic() + .03
        with self.assertRaises(QueueWaitExpired):
            self.turn(3, "image")
        self.assertEqual(self.r.inflight, [1, 1])
        with self.r.run_locks[0], self.r.run_locks[1]:
            self.stop._strata_queue_deadline = time.monotonic() + .03
            with self.assertRaises(QueueWaitExpired):
                self.turn(3)
        self.assertEqual(self.r.inflight, [1, 1])

    def test_waiting_cold_screenshot_triggers_heal(self):
        self.r.inflight[0] = 1
        alive = threading.Event()
        with patch.object(self.b, "alive", side_effect=alive.is_set), \
             patch.object(self.b, "restart", side_effect=alive.set, create=True):
            self.stop._strata_queue_deadline = time.monotonic() + 2
            self.assertEqual(self.turn(4, "image"), [17])
        self.assertTrue(alive.is_set())
        self.assertEqual(self.r.routed, [0, 1])

    def test_restart_removes_only_abandoned_checkpoint_directory(self):
        script = ("from serve.replica_handoff import Handoff; import os, sys;"
                  "h=Handoff(sys.argv[1], 1, 2); print(h.directory.name, flush=True); os._exit(0)")
        result = subprocess.run([sys.executable, "-c", script, self.tmp.name],
                                cwd=pathlib.Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        abandoned = pathlib.Path(result.stdout.strip())
        self.assertTrue(abandoned.is_dir())
        new = Handoff(self.tmp.name, 1, 2)
        self.addCleanup(new.close)
        self.assertFalse(abandoned.exists())
        self.assertTrue(pathlib.Path(self.store.directory.name).is_dir())

    def test_in_use_file_survives_eviction(self):
        self.store.budget = 128
        self.turn(3)
        with self.store.match([3]*257) as entry:
            self.turn(4)
            self.assertTrue(entry.path.is_file())
            self.assertEqual(self.store.size, 128)
        self.turn(5)
        self.assertFalse(entry.path.exists())
        self.assertEqual(len(self.store.entries), 1)
