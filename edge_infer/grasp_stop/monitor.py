#!/usr/bin/env python3
"""ADU ROS 2 pressure subscriber for the optional grasp-stop gate.

Use /usr/bin/python3 with the robot's ROS environment. The default mode is
read-only. The right-bottle profile supplies calibrated parameters; control
still requires explicit --control and --confirm I_UNDERSTAND.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from calibration import PROFILE_NAMES, resolve_profile
from detector import CONTACT_MODES, GraspDetector, PressureFrame, split_tactile

LOCAL_HTTP = build_opener(ProxyHandler({}))


def get_status(base_url: str) -> dict:
    with LOCAL_HTTP.open(base_url.rstrip("/") + "/status", timeout=0.5) as response:
        return json.load(response)


def post(base_url: str, path: str) -> dict:
    request = Request(base_url.rstrip("/") + path, data=b"{}", method="POST",
                      headers={"Content-Type": "application/json"})
    try:
        with LOCAL_HTTP.open(request, timeout=0.5) as response:
            result = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} HTTP {exc.code}: {body}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"{path} refused: {result}")
    return result


def await_cancel(base_url: str, stop_epoch: int, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        state = get_status(base_url)
        if state.get("cancel_failed"):
            raise RuntimeError("model chunk cancellation failed")
        if int(state.get("cancel_ack_epoch", 0)) >= stop_epoch:
            return
        time.sleep(0.02)
    raise RuntimeError("model chunk cancellation was not acknowledged")


class HeartbeatSender:
    """Send heartbeats independently, but only while the monitor loop is live."""

    def __init__(self, base_url: str, interval: float = 0.05,
                 progress_timeout: float = 0.20):
        self.base_url = base_url
        self.interval = interval
        self.progress_timeout = progress_timeout
        self.last_progress_at = time.monotonic()
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="grasp-stop-heartbeat", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def mark_progress(self) -> None:
        self.last_progress_at = time.monotonic()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            if time.monotonic() - self.last_progress_at > self.progress_timeout:
                self.error = RuntimeError("monitor loop stopped processing pressure samples")
                return
            try:
                post(self.base_url, "/grasp/heartbeat")
            except (URLError, OSError, RuntimeError) as exc:
                if not self.stop_event.is_set():
                    self.error = exc
                return

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"heartbeat failed: {self.error}") from self.error

    def stop(self) -> None:
        self.stop_event.set()

    def join(self) -> None:
        self.thread.join(timeout=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--profile", choices=PROFILE_NAMES,
                        default="right_bottle_v1",
                        help="Calibrated right bottle settings; 'none' restores explicit-only threshold")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the profile pressure threshold")
    parser.add_argument("--contact-mode", choices=CONTACT_MODES,
                        default=None)
    parser.add_argument("--dwell-seconds", type=float, default=None)
    parser.add_argument("--task-timeout-seconds", type=float, default=None)
    parser.add_argument("--control", action="store_true",
                        help="Arm/start the model and stop it on grasp/fault")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--control-url", default="http://127.0.0.1:5100")
    parser.add_argument("--log", type=Path, default=None,
                        help="Optional JSONL record; should be on ADU persistent data storage")
    args = parser.parse_args()
    try:
        settings = resolve_profile(
            args.profile, args.hand, threshold=args.threshold,
            contact_mode=args.contact_mode, dwell_seconds=args.dwell_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.threshold = settings.threshold
    args.contact_mode = settings.contact_mode
    args.dwell_seconds = settings.dwell_seconds
    args.task_timeout_seconds = settings.task_timeout_seconds
    if args.control and (args.threshold is None or args.confirm != "I_UNDERSTAND"):
        parser.error("--control requires calibrated --threshold and --confirm I_UNDERSTAND")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState

    detector = GraspDetector(hand=args.hand, threshold=args.threshold,
                            contact_mode=args.contact_mode,
                            dwell_seconds=args.dwell_seconds,
                            task_timeout=args.task_timeout_seconds)
    log_file = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log.open("a", encoding="utf-8")

    class PressureNode(Node):
        def __init__(self):
            super().__init__("a3_grasp_stop_pressure")
            qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self.latest: PressureFrame | None = None
            self.latest_wall_time_ns = 0
            self.seq = 0
            self.create_subscription(JointState, "/motion/control/hand_joint_state",
                                     self.on_sample, qos)

        def on_sample(self, msg: JointState) -> None:
            self.seq += 1
            stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            received_at = time.monotonic()
            self.latest_wall_time_ns = time.time_ns()
            self.latest = PressureFrame(received_at, msg.header.frame_id,
                                        tuple(float(x) for x in msg.effort), stamp)

    rclpy.init()
    node = PressureNode()
    seen_seq = 0
    armed = False
    stopped = False
    heartbeat: HeartbeatSender | None = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            item = None
            if node.seq != seen_seq and node.latest is not None:
                seen_seq = node.seq
                frame = node.latest
                state = detector.feed(frame)
                if log_file is not None:
                    item = {"seq": seen_seq, "received_at": frame.received_at,
                            "received_wall_time_ns": node.latest_wall_time_ns,
                            "source_stamp_ns": frame.source_stamp_ns,
                            "frame_id": frame.frame_id, "effort_len": len(frame.effort),
                            "effort_tail": frame.effort[-260:],
                            "hand": args.hand, "threshold": args.threshold,
                            "profile": args.profile,
                            "contact_mode": args.contact_mode,
                            "dwell_seconds": args.dwell_seconds,
                            "task_timeout_seconds": args.task_timeout_seconds,
                            "baseline": detector.baseline,
                            "section_scores": detector.last_section_scores,
                            "state": state, "score": detector.last_score}
                    try:
                        item["sections"] = split_tactile(frame, args.hand)
                    except ValueError as exc:
                        item["parse_error"] = str(exc)
            try:
                if heartbeat is not None:
                    heartbeat.check()
                state = detector.tick(time.monotonic())
                if heartbeat is not None:
                    heartbeat.mark_progress()
                if state == "READY" and args.control and not armed:
                    detector.arm(time.monotonic())
                    post(args.control_url, "/grasp/arm")
                    armed = True
                    post(args.control_url, "/grasp/heartbeat")
                    heartbeat = HeartbeatSender(args.control_url)
                    heartbeat.start()
                    post(args.control_url, "/start")
                    if item is not None:
                        item["control_event"] = {
                            "name": "start", "at": time.monotonic(),
                            "wall_time_ns": time.time_ns(),
                        }
                    print("[grasp-stop] ARMED and model started", flush=True)
                elif state == "GRASP_CONFIRMED" and armed:
                    heartbeat.stop()
                    request_at = time.monotonic()
                    request_wall_time_ns = time.time_ns()
                    result = post(args.control_url, "/grasp/complete")
                    stopped = True
                    await_cancel(args.control_url, int(result["stop_epoch"]))
                    if item is not None:
                        item["control_event"] = {
                            "name": "complete", "request_at": request_at,
                            "request_wall_time_ns": request_wall_time_ns,
                            "cancel_ack_at": time.monotonic(),
                            "cancel_ack_wall_time_ns": time.time_ns(),
                            "stop_epoch": int(result["stop_epoch"]),
                        }
                    print("[grasp-stop] contact confirmed; stop latched", flush=True)
                    return 0
                elif state in ("SENSOR_UNAVAILABLE", "TIMEOUT"):
                    print(f"[grasp-stop] {state}: {detector.fault_reason}", file=sys.stderr)
                    if armed:
                        heartbeat.stop()
                        request_at = time.monotonic()
                        request_wall_time_ns = time.time_ns()
                        result = post(args.control_url, "/grasp/fault")
                        stopped = True
                        await_cancel(args.control_url, int(result["stop_epoch"]))
                        if item is not None:
                            item["control_event"] = {
                                "name": "fault", "request_at": request_at,
                                "request_wall_time_ns": request_wall_time_ns,
                                "cancel_ack_at": time.monotonic(),
                                "cancel_ack_wall_time_ns": time.time_ns(),
                                "stop_epoch": int(result["stop_epoch"]),
                            }
                    return 2
            finally:
                # Preserve the final sample when a terminal branch returns.
                if item is not None:
                    log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    log_file.flush()
    except (KeyboardInterrupt, URLError, OSError, RuntimeError) as exc:
        print(f"[grasp-stop] stopped: {exc}", file=sys.stderr)
        return 3
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if armed and not stopped:
            try:
                if get_status(args.control_url).get("grasp_state") == "ARMED":
                    post(args.control_url, "/grasp/fault")
            except Exception as exc:
                print(f"[grasp-stop] STOP DELIVERY FAILED: {exc}", file=sys.stderr)
        if heartbeat is not None:
            heartbeat.join()
        node.destroy_node()
        rclpy.shutdown()
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
