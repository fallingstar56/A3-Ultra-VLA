#!/usr/bin/env python3
"""Offline replay of monitor JSONL; no ROS or robot connection required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector import GraspDetector, PressureFrame


def replay(path: Path, *, hand: str, threshold: float | None) -> dict:
    detector = GraspDetector(hand=hand, threshold=threshold)
    frames = 0
    armed = False
    for line in path.read_text(encoding="utf-8").splitlines():
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
        if state == "READY" and threshold is not None and not armed:
            detector.arm(frame.received_at)
            armed = True
        if state in ("GRASP_CONFIRMED", "SENSOR_UNAVAILABLE", "TIMEOUT"):
            break
    return {"frames": frames, "state": detector.state,
            "score": detector.last_score, "reason": detector.fault_reason,
            "threshold": threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Offline candidate; never writes production configuration")
    args = parser.parse_args()
    result = replay(args.log, hand=args.hand, threshold=args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
