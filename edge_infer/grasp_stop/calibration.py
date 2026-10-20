"""Calibrated right-hand bottle profile from the 2026-10-20 robot trials."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    threshold: float | None
    contact_mode: str
    dwell_seconds: float
    task_timeout_seconds: float


# calib_right_empty_001: empty close stayed at zero.
# wbt_readonly_001: both grasp windows exceed 20 for >3.5 s; the second
# exceeds 40 for only ~0.51 s. 1.5 s at 20 confirms inside both close windows.
# The first labeled VLA start-to-contact interval was ~111 s, so 20 s would
# time out before contact; 180 s includes that observed interval with margin.
RIGHT_BOTTLE_V1 = Calibration(
    threshold=20.0,
    contact_mode="thumb_index",
    dwell_seconds=1.5,
    task_timeout_seconds=180.0,
)

LEGACY_UNCALIBRATED = Calibration(
    threshold=None,
    contact_mode="thumb_two_fingers",
    dwell_seconds=0.150,
    task_timeout_seconds=20.0,
)

PROFILE_NAMES = ("right_bottle_v1", "none")


def resolve_profile(profile: str, hand: str, *, threshold: float | None,
                    contact_mode: str | None, dwell_seconds: float | None,
                    task_timeout_seconds: float | None) -> Calibration:
    if profile == "right_bottle_v1":
        if hand != "right":
            raise ValueError("right_bottle_v1 is calibrated for the right hand only")
        base = RIGHT_BOTTLE_V1
    elif profile == "none":
        base = LEGACY_UNCALIBRATED
    else:
        raise ValueError(f"unknown calibration profile: {profile}")
    return Calibration(
        threshold=base.threshold if threshold is None else threshold,
        contact_mode=base.contact_mode if contact_mode is None else contact_mode,
        dwell_seconds=base.dwell_seconds if dwell_seconds is None else dwell_seconds,
        task_timeout_seconds=(base.task_timeout_seconds if task_timeout_seconds is None
                              else task_timeout_seconds),
    )
