"""Pure, ROS-independent O10Hand pressure detector.

The pressure layout is a candidate from AVATAR. It must be validated on the
specific robot before enabling motion. No hand command is issued here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

LAYOUT = (("thumb", 16), ("index", 16), ("middle", 16),
          ("ring", 16), ("pinky", 16), ("palm", 25), ("back_of_hand", 25))
POINTS_PER_HAND = 130
POINTS_BOTH = 260
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class PressureFrame:
    received_at: float  # time.monotonic() in the ROS subscriber
    frame_id: str
    effort: tuple[float, ...]
    source_stamp_ns: int = 0


def split_tactile(frame: PressureFrame, hand: str) -> dict[str, tuple[float, ...]]:
    if frame.frame_id != "O10Hand":
        raise ValueError(f"expected O10Hand, got {frame.frame_id!r}")
    if hand not in ("left", "right"):
        raise ValueError("hand must be left or right")
    if len(frame.effort) < POINTS_BOTH:
        raise ValueError(f"effort has {len(frame.effort)} values; need >=260")
    tail = frame.effort[-POINTS_BOTH:]
    if any(not math.isfinite(x) for x in tail):
        raise ValueError("non-finite tactile value")
    hand_values = tail[:POINTS_PER_HAND] if hand == "left" else tail[POINTS_PER_HAND:]
    result: dict[str, tuple[float, ...]] = {}
    offset = 0
    for name, count in LAYOUT:
        result[name] = tuple(hand_values[offset:offset + count])
        offset += count
    return result


def top3(values: Sequence[float]) -> float:
    if len(values) < 3:
        raise ValueError("at least 3 values required")
    return sum(sorted(values, reverse=True)[:3]) / 3.0


class GraspDetector:
    """Baseline, freshness and sustained thumb plus two-finger contact.

    The caller owns the task lifecycle and must stop actions on `fault` or
    `confirmed`. Empty threshold permits baseline/replay only.
    """

    def __init__(self, *, hand: str = "right", threshold: float | None = None,
                 baseline_seconds: float = 1.0, dwell_seconds: float = 0.150,
                 stale_seconds: float = 0.200, task_timeout: float = 20.0):
        if hand not in ("left", "right"):
            raise ValueError("hand must be left or right")
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
            raise ValueError("threshold must be a finite positive number")
        self.hand = hand
        self.threshold = threshold
        self.baseline_seconds = baseline_seconds
        self.dwell_seconds = dwell_seconds
        self.stale_seconds = stale_seconds
        self.task_timeout = task_timeout
        self.state = "BASELINING"
        self.baseline_frames: list[dict[str, tuple[float, ...]]] = []
        self.baseline_start: float | None = None
        self.baseline: dict[str, tuple[float, ...]] = {}
        self.last_received: float | None = None
        self.last_source_stamp_ns = 0
        self.armed_at: float | None = None
        self.candidate_since: float | None = None
        self.candidate_count = 0
        self.last_score = 0.0
        self.fault_reason = ""

    def _fault(self, reason: str) -> str:
        self.state = "SENSOR_UNAVAILABLE" if reason != "task_timeout" else "TIMEOUT"
        self.fault_reason = reason
        return self.state

    def feed(self, frame: PressureFrame) -> str:
        if self.state in ("GRASP_CONFIRMED", "SENSOR_UNAVAILABLE", "TIMEOUT"):
            return self.state
        if self.last_received is not None and frame.received_at <= self.last_received:
            return self._fault("non_monotonic_receive_time")
        if (frame.source_stamp_ns and self.last_source_stamp_ns
                and frame.source_stamp_ns <= self.last_source_stamp_ns):
            return self._fault("non_monotonic_source_stamp")
        if (self.last_received is not None
                and frame.received_at - self.last_received > self.stale_seconds
                and self.state != "BASELINING"):
            return self._fault("sample_gap")
        try:
            parts = split_tactile(frame, self.hand)
        except ValueError as exc:
            return self._fault(str(exc))
        self.last_received = frame.received_at
        if frame.source_stamp_ns:
            self.last_source_stamp_ns = frame.source_stamp_ns
        if all(v == 0 for section in parts.values() for v in section):
            return self._fault("all_zero_tactile")

        if self.state == "BASELINING":
            if self.baseline_start is None:
                self.baseline_start = frame.received_at
            self.baseline_frames.append(parts)
            if frame.received_at - self.baseline_start >= self.baseline_seconds:
                self.baseline = {
                    name: tuple(statistics.median(sample[name][i]
                                                  for sample in self.baseline_frames)
                                for i in range(len(parts[name])))
                    for name in parts
                }
                self.state = "READY"
            return self.state

        if self.state == "READY":
            return self.state
        if self.armed_at is not None and frame.received_at - self.armed_at > self.task_timeout:
            return self._fault("task_timeout")
        section_scores = {}
        for name in FINGERS:
            delta = [max(0.0, x - b) for x, b in zip(parts[name], self.baseline[name])]
            section_scores[name] = top3(delta)
        opponents = sorted((section_scores[n] for n in FINGERS[1:]), reverse=True)
        self.last_score = min(section_scores["thumb"], opponents[1])
        if self.threshold is None:
            return self.state
        if self.last_score >= self.threshold:
            if self.candidate_since is None:
                self.candidate_since = frame.received_at
                self.candidate_count = 1
            else:
                self.candidate_count += 1
            self.state = "CONTACT_CANDIDATE"
            if (self.candidate_count >= 3
                    and frame.received_at - self.candidate_since >= self.dwell_seconds):
                self.state = "GRASP_CONFIRMED"
        else:
            self.candidate_since = None
            self.candidate_count = 0
            self.state = "ARMED"
        return self.state

    def arm(self, now: float) -> None:
        if self.threshold is None:
            raise ValueError("threshold is blank; motion remains disabled")
        if self.state != "READY" or self.last_received is None:
            raise RuntimeError("baseline and live pressure required")
        if now - self.last_received > 0.100:
            raise RuntimeError("pressure sample is stale")
        self.armed_at = now
        self.state = "ARMED"

    def tick(self, now: float) -> str:
        if self.state in ("ARMED", "CONTACT_CANDIDATE"):
            if self.last_received is None or now - self.last_received > self.stale_seconds:
                return self._fault("sensor_stale")
            if self.armed_at is not None and now - self.armed_at > self.task_timeout:
                return self._fault("task_timeout")
        return self.state
