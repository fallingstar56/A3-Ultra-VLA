"""Shared dexterous-hand opening expansion for real-robot inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Joint order within each 10-DoF hand:
#   thumb_rotation, thumb_wiggle, thumb_bent, index_wiggle, index_bent,
#   middle_bent, ring_wiggle, ring_bent, pinky_wiggle, pinky_bent.
LEFT_THUMB_FIXED_INDICES = (0, 1)
RIGHT_THUMB_FIXED_INDICES = (10, 11)
LEFT_HAND_BENT_INDICES = (2, 4, 5, 7, 9)
RIGHT_HAND_BENT_INDICES = (12, 14, 15, 17, 19)

# Fixed thumb rotation/wiggle active-joint poses from episode_000000.parquet
# frame 0: observation.state[20:22] for left and [30:32] for right. Thumb
# bent remains controlled by each side's opening value.
LEFT_THUMB_FIXED_POSE = np.array([-0.22324707, 1.63422358], dtype=np.float32)
RIGHT_THUMB_FIXED_POSE = np.array([0.22437012, -1.43246341], dtype=np.float32)

# The model's opening is index_bent_1 in radians, whose fully closed value is
# 1.48. Use its normalized ratio to drive all five bent joints on each side.
HAND_OPENING_REF_CLOSE = 1.48
LEFT_HAND_BENT_CLOSE = np.array([-0.8416, 1.48, 1.48, 1.48, 1.48], dtype=np.float32)
RIGHT_HAND_BENT_CLOSE = np.array([0.8416, 1.48, 1.48, 1.48, 1.48], dtype=np.float32)


@dataclass(frozen=True)
class HandOpeningMapping:
    """Checkpoint-owned initial hand pose for single-DoF expansion."""

    initial_hand_rad: np.ndarray

    def __post_init__(self) -> None:
        hand = np.asarray(self.initial_hand_rad, dtype=np.float32).reshape(-1)
        if hand.size != 20:
            raise ValueError(
                f"hand mapping initial_hand_rad must be 20D, got {hand.size}D"
            )
        object.__setattr__(self, "initial_hand_rad", hand.copy())

    def to_metadata(self) -> dict[str, list[float]]:
        """Return a msgpack-friendly representation for remote RTC clients."""
        return {"initial_hand_rad": self.initial_hand_rad.tolist()}

    @classmethod
    def from_metadata(cls, value: object) -> "HandOpeningMapping | None":
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            return cls(np.asarray(value, dtype=np.float32))
        if not isinstance(value, dict):
            raise ValueError(
                "hand_mapping metadata must contain a 20D active-joint sample"
            )
        for key in ("initial_hand_rad", "activejointpos", "hand_rad"):
            if key in value:
                return cls(np.asarray(value[key], dtype=np.float32))
        raise ValueError(
            "hand_mapping metadata must contain initial_hand_rad or activejointpos"
        )


def load_hand_opening_mapping(parquet_path: str | Path) -> HandOpeningMapping:
    """Load the first hand active-joint pose from a checkpoint parquet."""
    import pandas as pd

    path = Path(parquet_path)
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"hand mapping parquet is empty: {path}")
    row = frame.iloc[0]
    if "hand_mapping.initial_hand_rad" in row.index:
        hand = np.asarray(row["hand_mapping.initial_hand_rad"], dtype=np.float32).reshape(-1)
    elif "observation.state" in row.index:
        state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
        if state.size < 40:
            raise ValueError(
                f"observation.state in {path} is {state.size}D; expected at least 40D"
            )
        hand = state[20:40]
    else:
        raise ValueError(
            f"{path} has neither hand_mapping.initial_hand_rad nor observation.state"
        )
    return HandOpeningMapping(hand)


def expand_hand_opening_to_hand20(
    opening: np.ndarray,
    current_hand: np.ndarray | None,
    mapping: HandOpeningMapping | None = None,
) -> np.ndarray:
    """Expand ``[..., 2]`` left/right opening values to a 20-DoF hand chunk.

    Each side's index-bend value is normalized by its fully closed value and
    then scaled to all five bent joints. With ``mapping``, non-driven joints
    and thumb rotation/wiggle use the checkpoint's first-frame pose. Without
    it, the live ``current_hand`` and legacy fixed thumb constants are used.
    """
    opening = np.asarray(opening, dtype=np.float32)
    if opening.ndim < 1 or opening.shape[-1] != 2:
        raise ValueError(f"hand opening must have shape (..., 2), got {opening.shape}")

    if mapping is not None:
        initial = mapping.initial_hand_rad
        left_thumb = initial[list(LEFT_THUMB_FIXED_INDICES)]
        right_thumb = initial[list(RIGHT_THUMB_FIXED_INDICES)]
    else:
        initial = np.asarray(current_hand, dtype=np.float32).reshape(-1)
        if initial.size != 20:
            raise ValueError(f"current hand must be 20D, got {initial.size}D")
        left_thumb = LEFT_THUMB_FIXED_POSE
        right_thumb = RIGHT_THUMB_FIXED_POSE

    hand = np.broadcast_to(initial, (*opening.shape[:-1], 20)).copy()
    ratio = np.clip(opening / HAND_OPENING_REF_CLOSE, 0.0, 1.0)
    hand[..., LEFT_THUMB_FIXED_INDICES] = left_thumb
    hand[..., RIGHT_THUMB_FIXED_INDICES] = right_thumb
    hand[..., LEFT_HAND_BENT_INDICES] = ratio[..., 0, None] * LEFT_HAND_BENT_CLOSE
    hand[..., RIGHT_HAND_BENT_INDICES] = ratio[..., 1, None] * RIGHT_HAND_BENT_CLOSE
    return hand


def expand_right_opening_to_hand20(
    opening_scalar: float,
    current_hand: np.ndarray,
) -> list[float]:
    """Expand one right-hand opening value using the existing A2 mapping."""
    hand = np.asarray(current_hand, dtype=np.float32).reshape(-1).copy()
    if hand.size < 20:
        hand = np.concatenate([hand, np.zeros(20 - hand.size, dtype=np.float32)])
    for idx in RIGHT_HAND_BENT_INDICES:
        hand[idx] = float(opening_scalar)
    return hand[:20].tolist()
