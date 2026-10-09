#!/usr/bin/env python3
"""ADU ROS 2 pressure subscriber for the optional grasp-stop gate.

Use /usr/bin/python3 with the robot's ROS environment. The default mode is
read-only. Control mode requires an explicitly calibrated pressure threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from detector import GraspDetector, PressureFrame, split_tactile

LOCAL_HTTP = build_opener(ProxyHandler({}))


def post(base_url: str, path: str) -> dict:
    request = Request(base_url.rstrip("/") + path, data=b"{}", method="POST",
                      headers={"Content-Type": "application/json"})
    with LOCAL_HTTP.open(request, timeout=0.5) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(f"{path} refused: {result}")
    return result


def await_cancel(base_url: str, stop_epoch: int, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with LOCAL_HTTP.open(base_url.rstrip("/") + "/status", timeout=0.5) as response:
            state = json.load(response)
        if state.get("cancel_failed"):
            raise RuntimeError("model chunk cancellation failed")
        if int(state.get("cancel_ack_epoch", 0)) >= stop_epoch:
            return
        time.sleep(0.02)
    raise RuntimeError("model chunk cancellation was not acknowledged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Calibrated raw pressure threshold; intentionally blank by default")
    parser.add_argument("--control", action="store_true",
                        help="Arm/start the model and stop it on grasp/fault")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--control-url", default="http://127.0.0.1:5100")
    parser.add_argument("--log", type=Path, default=None,
                        help="Optional JSONL record; should be on ADU persistent data storage")
    args = parser.parse_args()
    if args.control and (args.threshold is None or args.confirm != "I_UNDERSTAND"):
        parser.error("--control requires calibrated --threshold and --confirm I_UNDERSTAND")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState

    detector = GraspDetector(hand=args.hand, threshold=args.threshold)
    log_file = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log.open("a", encoding="utf-8")

    class PressureNode(Node):
        def __init__(self):
            super().__init__("a3_grasp_stop_pressure")
            qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self.latest: PressureFrame | None = None
            self.seq = 0
            self.create_subscription(JointState, "/motion/control/hand_joint_state",
                                     self.on_sample, qos)

        def on_sample(self, msg: JointState) -> None:
            self.seq += 1
            stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            self.latest = PressureFrame(time.monotonic(), msg.header.frame_id,
                                        tuple(float(x) for x in msg.effort), stamp)

    rclpy.init()
    node = PressureNode()
    seen_seq = 0
    armed = False
    stopped = False
    next_heartbeat = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.seq != seen_seq and node.latest is not None:
                seen_seq = node.seq
                frame = node.latest
                state = detector.feed(frame)
                if log_file is not None:
                    item = {"seq": seen_seq, "received_at": frame.received_at,
                            "source_stamp_ns": frame.source_stamp_ns,
                            "frame_id": frame.frame_id, "effort_len": len(frame.effort),
                            "effort_tail": frame.effort[-260:],
                            "state": state, "score": detector.last_score}
                    try:
                        item["sections"] = split_tactile(frame, args.hand)
                    except ValueError as exc:
                        item["parse_error"] = str(exc)
                    log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    log_file.flush()
            state = detector.tick(time.monotonic())
            if state == "READY" and args.control and not armed:
                detector.arm(time.monotonic())
                post(args.control_url, "/grasp/arm")
                armed = True
                post(args.control_url, "/grasp/heartbeat")
                next_heartbeat = time.monotonic() + 0.05
                post(args.control_url, "/start")
                print("[grasp-stop] ARMED and model started", flush=True)
            elif state == "GRASP_CONFIRMED" and armed:
                result = post(args.control_url, "/grasp/complete")
                stopped = True
                await_cancel(args.control_url, int(result["stop_epoch"]))
                print("[grasp-stop] contact confirmed; stop latched", flush=True)
                return 0
            elif state in ("SENSOR_UNAVAILABLE", "TIMEOUT"):
                print(f"[grasp-stop] {state}: {detector.fault_reason}", file=sys.stderr)
                if armed:
                    result = post(args.control_url, "/grasp/fault")
                    stopped = True
                    await_cancel(args.control_url, int(result["stop_epoch"]))
                return 2
            if armed and not stopped and time.monotonic() >= next_heartbeat:
                post(args.control_url, "/grasp/heartbeat")
                next_heartbeat = time.monotonic() + 0.05
    except (KeyboardInterrupt, URLError, OSError, RuntimeError) as exc:
        print(f"[grasp-stop] stopped: {exc}", file=sys.stderr)
        return 3
    finally:
        if armed and not stopped:
            try:
                post(args.control_url, "/grasp/fault")
            except Exception as exc:
                print(f"[grasp-stop] STOP DELIVERY FAILED: {exc}", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
