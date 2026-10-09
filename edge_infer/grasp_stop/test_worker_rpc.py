"""Exercise the ROS worker RPC path without ROS or robot hardware."""

from __future__ import annotations

import ast
import queue
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_rpc_client():
    source = (Path(__file__).resolve().parents[1] / "infer_a3_edge.py").read_text()
    tree = ast.parse(source)
    robot = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "SubprocA3Robot")
    wanted = {"_reader_loop", "_call", "cancel_chunk"}
    methods = [node for node in robot.body
               if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"time": time}
    exec(compile(ast.Module(body=methods, type_ignores=[]),
                 "infer_a3_edge.py", "exec"), namespace)
    return type("RpcClient", (), {name: namespace[name] for name in wanted})


class WorkerRpcTests(unittest.TestCase):
    def setUp(self):
        self.commands = queue.Queue()
        self.responses = queue.Queue()
        commands, responses = self.commands, self.responses

        class Wire:
            @staticmethod
            def send_msg(_stream, message):
                commands.put(message)

            @staticmethod
            def recv_msg(_stream):
                return responses.get(timeout=2.0)

        self.client = load_rpc_client()()
        self.client._W = Wire
        self.client.proc = SimpleNamespace(stdin=object(), stdout=object())
        self.client._send_lock = threading.Lock()
        self.client._replies_cv = threading.Condition()
        self.client._next_request_id = 1
        self.client._pending_requests = set()
        self.client._replies_by_id = {}
        self.client._worker_error = None
        self.reader = threading.Thread(target=self.client._reader_loop, daemon=True)
        self.reader.start()

    def tearDown(self):
        self.responses.put(None)
        self.reader.join(timeout=2.0)
        self.assertFalse(self.reader.is_alive())

    def test_cancel_ack_is_not_taken_from_concurrent_chunk_send(self):
        results = {}

        def chunk_call():
            results["chunk"] = self.client._call("wb_chunk", timeout=1.0)

        def cancel_call():
            results["cancel"] = self.client.cancel_chunk()

        chunk_thread = threading.Thread(target=chunk_call)
        cancel_thread = threading.Thread(target=cancel_call)
        chunk_thread.start()
        cancel_thread.start()
        requests = [self.commands.get(timeout=1.0) for _ in range(2)]
        ids = {request[2]: request[1] for request in requests}
        self.assertEqual({"wb_chunk", "cancel"}, set(ids))
        self.assertTrue(all(request[0] == "rpc" for request in requests))

        # Deliver the cancel reply first, then the chunk reply. Both callers
        # must receive the response bearing their own request ID.
        self.responses.put(("rpc", ids["cancel"], "ok"))
        self.responses.put(("rpc", ids["wb_chunk"], "ok", {"actual_delay": 3}))
        chunk_thread.join(timeout=2.0)
        cancel_thread.join(timeout=2.0)
        self.assertFalse(chunk_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())
        self.assertEqual(results["chunk"], ("ok", {"actual_delay": 3}))
        self.assertIs(results["cancel"], True)

    def test_late_reply_cannot_satisfy_next_call(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.client._call("ping", timeout=0.01)
        old_id = self.commands.get(timeout=1.0)[1]
        self.responses.put(("rpc", old_id, "ok"))

        result = {}
        caller = threading.Thread(
            target=lambda: result.setdefault("reply", self.client._call("snapshot", timeout=1.0))
        )
        caller.start()
        new_id = self.commands.get(timeout=1.0)[1]
        self.assertNotEqual(old_id, new_id)
        self.responses.put(("rpc", new_id, "ok", 123))
        caller.join(timeout=2.0)
        self.assertFalse(caller.is_alive())
        self.assertEqual(result["reply"], ("ok", 123))

    def test_cancel_error_is_not_acknowledged_as_success(self):
        errors = []

        def cancel():
            try:
                self.client.cancel_chunk()
            except RuntimeError as exc:
                errors.append(str(exc))

        caller = threading.Thread(target=cancel, daemon=True)
        caller.start()
        request_id = self.commands.get(timeout=1.0)[1]
        self.responses.put(("rpc", request_id, "error", "cancel failed"))
        caller.join(timeout=1.0)
        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, ["ROS worker: cancel failed"])

    def test_worker_exit_wakes_all_pending_calls(self):
        errors = []

        def call(command):
            try:
                self.client._call(command, timeout=10.0)
            except RuntimeError as exc:
                errors.append(str(exc))

        callers = [threading.Thread(target=call, args=(cmd,), daemon=True)
                   for cmd in ("snapshot", "cancel")]
        for caller in callers:
            caller.start()
        for _ in callers:
            self.commands.get(timeout=1.0)
        self.responses.put(None)
        for caller in callers:
            caller.join(timeout=1.0)
            self.assertFalse(caller.is_alive())
        self.assertEqual(errors, ["ROS worker exited"] * 2)


if __name__ == "__main__":
    unittest.main()
