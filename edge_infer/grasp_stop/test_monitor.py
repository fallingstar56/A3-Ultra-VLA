"""Monitor integration with simulated ROS messages; no motion or ROS runtime."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import monitor
from test_grasp_stop import frame


class MonitorTests(unittest.TestCase):
    def test_completion_and_cancel_ack_precede_final_log_write(self):
        events = []
        clock = [0.0]
        samples = [frame(i * 0.05, stamp=i + 1) for i in range(21)]
        samples += [frame(t, thumb=5, index=5, middle=5, stamp=i + 22)
                    for i, t in enumerate((1.05, 1.10, 1.21))]

        class Node:
            def __init__(self, name):
                pass

            def create_subscription(self, message_type, topic, callback, qos):
                self.callback = callback

            def destroy_node(self):
                pass

        def spin_once(node, **kwargs):
            sample = samples.pop(0)
            clock[0] = sample.received_at
            node.callback(SimpleNamespace(
                header=SimpleNamespace(frame_id=sample.frame_id,
                                       stamp=SimpleNamespace(sec=0, nanosec=sample.source_stamp_ns)),
                effort=sample.effort,
            ))

        class Log:
            def write(self, text):
                events.append("final_log" if '"GRASP_CONFIRMED"' in text else "sample_log")

            def flush(self):
                pass

            def close(self):
                pass

        modules = {
            "rclpy": SimpleNamespace(init=lambda: None, shutdown=lambda: None,
                                     ok=lambda: bool(samples), spin_once=spin_once),
            "rclpy.node": SimpleNamespace(Node=Node),
            "rclpy.qos": SimpleNamespace(
                QoSProfile=lambda **kwargs: kwargs,
                QoSHistoryPolicy=SimpleNamespace(KEEP_LAST=1),
                QoSReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1)),
            "sensor_msgs.msg": SimpleNamespace(JointState=object),
        }

        def post(url, path):
            events.append(path)
            return {"ok": True, "stop_epoch": 1}

        argv = ["monitor.py", "--threshold", "2", "--control",
                "--confirm", "I_UNDERSTAND", "--log", "/unused/trial.jsonl"]
        with (patch.dict(sys.modules, modules), patch.object(sys, "argv", argv),
              patch.object(monitor.time, "monotonic", side_effect=lambda: clock[0]),
              patch.object(monitor.Path, "mkdir"),
              patch.object(monitor.Path, "open", return_value=Log()),
              patch.object(monitor, "post", side_effect=post),
              patch.object(monitor, "await_cancel", side_effect=lambda *a: events.append("ack"))):
            self.assertEqual(monitor.main(), 0)
        self.assertLess(events.index("/grasp/complete"), events.index("ack"))
        self.assertLess(events.index("ack"), events.index("final_log"))
        self.assertEqual(events.count("/start"), 1)
        self.assertEqual(events.count("/grasp/complete"), 1)


if __name__ == "__main__":
    unittest.main()
