#!/usr/bin/env python3
"""Offline replay of monitor JSONL; no ROS or robot connection required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration import PROFILE_NAMES, resolve_profile
from detector import CONTACT_MODES, GraspDetector, PressureFrame


def replay(path: Path, *, hand: str, threshold: float | None,
           contact_mode: str = "thumb_two_fingers",
           dwell_seconds: float = 0.150,
           task_timeout_seconds: float = 20.0,
           arm_wall_time_ns: int | None = None) -> dict:
    detector = GraspDetector(hand=hand, threshold=threshold,
                            contact_mode=contact_mode,
                            dwell_seconds=dwell_seconds,
                            task_timeout=task_timeout_seconds)
    frames = 0
    armed = False
    confirmed_wall_time_ns = None
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if "effort_tail" not in item:
                raise ValueError("log has no effort_tail; capture with the current monitor")
            frame = PressureFrame(
                received_at=float(item["received_at"]),
                frame_id=str(item["frame_id"]),
                effort=tuple(float(x) for x in item["effort_tail"]),
                source_stamp_ns=int(item.get("source_stamp_ns", 0)),
            )
            frames += 1
            state = detector.feed(frame)
            if (state == "READY" and threshold is not None and not armed
                    and (arm_wall_time_ns is None or
                         int(item.get("received_wall_time_ns", 0)) >= arm_wall_time_ns)):
                detector.arm(frame.received_at)
                armed = True
            if state in ("GRASP_CONFIRMED", "SENSOR_UNAVAILABLE", "TIMEOUT"):
                if state == "GRASP_CONFIRMED":
                    confirmed_wall_time_ns = item.get("received_wall_time_ns")
                break
    return {"frames": frames, "state": detector.state,
            "score": detector.last_score, "reason": detector.fault_reason,
            "threshold": threshold, "contact_mode": contact_mode,
            "dwell_seconds": dwell_seconds,
            "task_timeout_seconds": task_timeout_seconds,
            "arm_wall_time_ns": arm_wall_time_ns,
            "confirmed_wall_time_ns": confirmed_wall_time_ns}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="none")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Offline candidate; never writes production configuration")
    parser.add_argument("--contact-mode", choices=CONTACT_MODES, default=None)
    parser.add_argument("--dwell-seconds", type=float, default=None)
    parser.add_argument("--task-timeout-seconds", type=float, default=None)
    parser.add_argument("--arm-wall-time-ns", type=int, default=None,
                        help="Replay arm point for a labeled trial within a longer recording")
    args = parser.parse_args()
    try:
        settings = resolve_profile(
            args.profile, args.hand, threshold=args.threshold,
            contact_mode=args.contact_mode, dwell_seconds=args.dwell_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = replay(args.log, hand=args.hand, threshold=settings.threshold,
                    contact_mode=settings.contact_mode,
                    dwell_seconds=settings.dwell_seconds,
                    task_timeout_seconds=settings.task_timeout_seconds,
                    arm_wall_time_ns=args.arm_wall_time_ns)
    result["profile"] = args.profile
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
