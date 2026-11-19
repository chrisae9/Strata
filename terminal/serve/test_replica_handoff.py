import pathlib
import tempfile
import threading
import unittest
from unittest.mock import patch

from serve.replica_handoff import Handoff
from serve.replicas import ReplicaEngine
from serve.server_parts.engines import EngineDied, SessionRefused


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
