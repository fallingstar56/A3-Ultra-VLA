"""Offline checks for pressure layout, detection and the inference stop gate."""

from __future__ import annotations

import ast
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from detector import GraspDetector, PressureFrame, split_tactile


def frame(t: float, *, thumb=1.0, index=1.0, middle=1.0,
          hand="right", stamp=0) -> PressureFrame:
    right = []
    for name, count in (("thumb", 16), ("index", 16), ("middle", 16),
                        ("ring", 16), ("pinky", 16), ("palm", 25),
                        ("back_of_hand", 25)):
        value = {"thumb": thumb, "index": index, "middle": middle}.get(name, 1.0)
        right.extend([value] * count)
    left = [7.0] * 130
    tail = left + right if hand == "right" else right + left
    return PressureFrame(t, "O10Hand", tuple([99.0] * 20 + tail), stamp)


class DetectorTests(unittest.TestCase):
    def baseline(self, threshold=2.0):
        detector = GraspDetector(threshold=threshold)
        for i in range(21):
            detector.feed(frame(i * 0.05, stamp=i + 1))
        self.assertEqual(detector.state, "READY")
        detector.arm(1.0)
        return detector

    def test_right_tail_layout_with_prefix(self):
        self.assertEqual(split_tactile(frame(0), "right")["thumb"], (1.0,) * 16)
        self.assertEqual(split_tactile(frame(0), "left")["thumb"], (7.0,) * 16)

    def test_sustained_thumb_two_fingers(self):
        detector = self.baseline()
        self.assertEqual(detector.feed(frame(1.05, thumb=5, index=5, middle=5,
                                             stamp=22)), "CONTACT_CANDIDATE")
        self.assertEqual(detector.feed(frame(1.10, thumb=5, index=5, middle=5,
                                             stamp=23)), "CONTACT_CANDIDATE")
        self.assertEqual(detector.feed(frame(1.21, thumb=5, index=5, middle=5,
                                             stamp=24)), "GRASP_CONFIRMED")
        self.assertEqual(detector.feed(frame(1.26, stamp=25)), "GRASP_CONFIRMED")

    def test_one_opposing_finger_does_not_confirm(self):
        detector = self.baseline()
        for i in range(1, 6):
            detector.feed(frame(1 + i * 0.04, thumb=5, index=5, middle=1,
                                stamp=21 + i))
        self.assertEqual(detector.state, "ARMED")

    def test_stale_and_all_zero_fail_closed(self):
        detector = self.baseline()
        self.assertEqual(detector.tick(1.21), "SENSOR_UNAVAILABLE")
        zeros = GraspDetector(threshold=2)
        self.assertEqual(zeros.feed(PressureFrame(0, "O10Hand", (0.0,) * 260)),
                         "SENSOR_UNAVAILABLE")

    def test_blank_threshold_cannot_arm(self):
        detector = GraspDetector()
        for i in range(21):
            detector.feed(frame(i * 0.05, stamp=i + 1))
        with self.assertRaises(ValueError):
            detector.arm(1.0)


def load_control_symbols():
    """Compile only the control class/HTTP function; model imports need a robot."""
    source = (Path(__file__).resolve().parents[1] / "infer_a3_edge.py").read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ("KeyStateMachine", "start_human_in_loop_server")]
    namespace = {"threading": threading, "time": time, "sys": sys,
                 "select": __import__("select"), "termios": __import__("termios"),
                 "tty": __import__("tty")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "infer_a3_edge.py", "exec"),
         namespace)
    return namespace["KeyStateMachine"], namespace["start_human_in_loop_server"]


class GateTests(unittest.TestCase):
    def test_latch_blocks_restart_and_cancel_ack(self):
        machine, _ = load_control_symbols()
        gate = machine(grasp_stop_enabled=True)
        try:
            self.assertFalse(gate.set_running("unarmed"))
            self.assertTrue(gate.arm_grasp())
            self.assertTrue(gate.heartbeat_grasp())
            self.assertTrue(gate.set_running("test"))
            self.assertTrue(gate.latch_grasp("GRASP_CONFIRMED"))
            self.assertFalse(gate.set_running("restart"))
            self.assertFalse(gate.arm_grasp())
            gate.ack_cancel(True)
            status = gate.status_snapshot()
            self.assertEqual(status["cancel_ack_epoch"], status["stop_epoch"])
        finally:
            gate.close()

    def test_missing_heartbeat_forces_idle(self):
        machine, _ = load_control_symbols()
        gate = machine(grasp_stop_enabled=True)
        try:
            gate.arm_grasp()
            gate.set_running("test")
            with gate._state_lock:
                gate.grasp_heartbeat_at -= 1.0
            self.assertFalse(gate.is_running())
            self.assertEqual(gate.status_snapshot()["grasp_state"], "SENSOR_UNAVAILABLE")
        finally:
            gate.close()

    def test_human_stop_latches_abort(self):
        machine, _ = load_control_symbols()
        gate = machine(grasp_stop_enabled=True)
        try:
            gate.arm_grasp()
            gate.set_running("test")
            gate.force_idle("keyboard p")
            self.assertEqual(gate.status_snapshot()["grasp_state"], "HUMAN_ABORT")
            self.assertFalse(gate.set_running("restart"))
        finally:
            gate.close()

    def test_http_complete_status(self):
        machine, server_factory = load_control_symbols()
        gate = machine(grasp_stop_enabled=True)
        server = server_factory(gate, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{server.server_port}"
        local_http = build_opener(ProxyHandler({}))
        def post(path):
            with local_http.open(Request(url + path, data=b"{}", method="POST"), timeout=1) as response:
                return json.load(response)
        try:
            with self.assertRaises(HTTPError):
                post("/start")
            post("/grasp/arm")
            post("/grasp/heartbeat")
            post("/start")
            result = post("/grasp/complete")
            self.assertEqual(result["grasp_state"], "GRASP_CONFIRMED")
            with self.assertRaises(HTTPError):
                post("/start")
        finally:
            server.shutdown()
            server.server_close()
            gate.close()


if __name__ == "__main__":
    unittest.main()
