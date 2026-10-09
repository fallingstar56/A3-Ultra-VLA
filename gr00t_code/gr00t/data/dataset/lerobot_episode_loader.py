#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import get_frames_by_indices


def _quat_to_rot6d(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Convert a batch of quaternions to 6D rotation representation.

    The 6D rotation is the first two rows of the 3x3 rotation matrix flattened,
    which is a continuous representation friendly for regression.

    Args:
        quat: Array of quaternions with shape (..., 4).
        order: Either "xyzw" (scipy native) or "wxyz".

    Returns:
        Array of 6D rotations with shape (..., 6).
    """
    quat = np.asarray(quat, dtype=np.float64)
    flat = quat.reshape(-1, 4)
    if order.lower() == "wxyz":
        # scipy expects xyzw, reorder
        flat = np.stack([flat[:, 1], flat[:, 2], flat[:, 3], flat[:, 0]], axis=-1)
    elif order.lower() != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {order}")
    # Normalize to avoid numerical issues
    norm = np.linalg.norm(flat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    flat = flat / norm
    rot_mat = Rotation.from_quat(flat).as_matrix()  # (N, 3, 3)
    # First two rows flattened -> 6D
    rot6d = rot_mat[:, :2, :].reshape(-1, 6)
    out_shape = quat.shape[:-1] + (6,)
    return rot6d.reshape(out_shape).astype(np.float32)


def _apply_index_slice(array: np.ndarray, group_info: dict) -> np.ndarray:
    """Slice a raw data array according to a modality-meta entry.

    Supports three forms:
      - ``indices``: explicit list of column indices (arbitrary order).
      - ``start`` / ``end``: contiguous slice ``[start, end)``.

    Args:
        array: Raw 1-D array for a single step.
        group_info: Dict from modality.json describing the group.

    Returns:
        Sliced 1-D array.
    """
    if "indices" in group_info:
        return np.asarray(array)[np.asarray(group_info["indices"], dtype=np.int64)]
    start_idx = group_info["start"]
    end_idx = group_info["end"]
    return np.asarray(array)[start_idx:end_idx]

# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_SUBTASK_OVERRIDES_FILENAME = "subtasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language", "mask"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        modality_meta_path: str | Path | None = None,
        stats_dir: str | Path | None = None,
        tasks_dir: str | Path | None = None,
        subtask_conditioned: bool = False,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon

        Args:
            modality_meta_path: Optional absolute path to a shared ``modality.json``
                file. If provided, this file is used INSTEAD of
                ``<dataset_path>/meta/modality.json`` for every dataset. Useful
                when training with multiple datasets that share the same
                modality layout but you don't want to duplicate the file under
                each dataset's ``meta/`` dir. All other meta files
                (info.json / episodes.jsonl / tasks.jsonl / stats.json) are
                still read from each dataset's own ``meta/`` dir.
            stats_dir: Optional directory that stores per-dataset stats
                outside the dataset's own ``meta/`` dir. When set, this
                loader reads ``<stats_dir>/<dataset_name>/stats.json`` and
                ``<stats_dir>/<dataset_name>/relative_stats.json`` instead
                of ``<dataset_path>/meta/{stats,relative_stats}.json``.
                Useful when the dataset directories are read-only.
            tasks_dir: Optional directory that stores per-dataset
                ``tasks.jsonl`` outside the dataset's own ``meta/`` dir.
                When set, this loader reads
                ``<tasks_dir>/<dataset_name>/tasks.jsonl`` instead of
                ``<dataset_path>/meta/tasks.jsonl``. Useful when the
                dataset directories are read-only, or when you want to
                override the task descriptions without touching the
                original dataset. When ``None`` (default), the original
                per-dataset ``meta/tasks.jsonl`` is used.
            subtask_conditioned: Append ``Subtask: <text>`` to the global
                task loaded through ``task_index``. The subtask is selected
                by the current frame's ``[start, end)`` interval in
                ``episodes.jsonl``.
        """
        self.dataset_path = Path(dataset_path)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.modality_meta_path = (
            Path(modality_meta_path) if modality_meta_path else None
        )
        self.stats_dir = Path(stats_dir) if stats_dir else None
        self.tasks_dir = Path(tasks_dir) if tasks_dir else None
        self.subtask_conditioned = bool(subtask_conditioned)

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(
            modality_configs
        )
        if self.subtask_conditioned:
            if "language" not in self.modality_configs:
                raise ValueError("subtask_conditioned requires a language modality")
            language_key = self.modality_configs["language"].modality_keys[0]
            if not language_key.startswith("annotation."):
                raise ValueError(
                    "subtask_conditioned requires an annotation language key "
                    "backed by tasks.jsonl"
                )
            annotation_key = language_key.removeprefix("annotation.")
            original_key = self.modality_meta["annotation"][annotation_key].get(
                "original_key", language_key
            )
            if original_key != "task_index":
                raise ValueError(
                    "subtask_conditioned requires the global language annotation "
                    "to use task_index"
                )

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        # Load episode metadata (one episode per line)
        episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
        with open(episodes_path, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        # Load task descriptions and create mapping. If ``tasks_dir`` was
        # provided, read from <tasks_dir>/<dataset_name>/tasks.jsonl instead
        # of the dataset's own meta/ dir. This mirrors the ``stats_dir``
        # mechanism and lets multiple read-only datasets keep overridden
        # task descriptions in a writable location. If the external file
        # doesn't exist yet, we seed it by copying the dataset's own
        # meta/tasks.jsonl so subsequent runs can edit it in place.
        if self.tasks_dir is not None:
            tasks_path = self.tasks_dir / self.dataset_path.name / LEROBOT_TASKS_FILENAME
            if not tasks_path.exists():
                fallback = meta_dir / LEROBOT_TASKS_FILENAME
                if not fallback.exists():
                    raise FileNotFoundError(
                        f"External tasks.jsonl not found at {tasks_path} "
                        f"and no fallback at {fallback} "
                        f"(tasks_dir={self.tasks_dir}, dataset={self.dataset_path.name})."
                    )
                tasks_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fallback, tasks_path)
                logging.info(
                    "tasks.jsonl seeded for dataset %s: copied %s -> %s",
                    self.dataset_path.name,
                    fallback,
                    tasks_path,
                )
        else:
            tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            # Skip blank lines (trailing newlines from editors, etc.) so a
            # stray empty line at EOF doesn't crash json.loads("").
            tasks_data = [json.loads(line) for line in f if line.strip()]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Optional per-dataset subtask wording overrides live beside the
        # external tasks.jsonl. This keeps read-only episodes.jsonl untouched
        # while allowing prompt-only wording improvements.
        self.subtask_text_overrides: dict[str, str] = {}
        if self.tasks_dir is not None and self.subtask_conditioned:
            subtask_overrides_path = (
                tasks_path.parent / LEROBOT_SUBTASK_OVERRIDES_FILENAME
            )
            if subtask_overrides_path.exists():
                with open(subtask_overrides_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        override = json.loads(line)
                        source_text = str(override["source_text"]).strip()
                        replacement_text = str(override["text"]).strip()
                        if not source_text or not replacement_text:
                            raise ValueError(
                                f"Invalid subtask override in {subtask_overrides_path}: "
                                f"{override!r}"
                            )
                        if source_text in self.subtask_text_overrides:
                            raise ValueError(
                                f"Duplicate source_text {source_text!r} in "
                                f"{subtask_overrides_path}"
                            )
                        self.subtask_text_overrides[source_text] = replacement_text

        # Load modality structure information. If a shared modality.json path
        # was provided, read from there instead of the dataset-local one so
        # multiple datasets can share a single modality definition.
        if self.modality_meta_path is not None:
            modality_path = self.modality_meta_path
            if not modality_path.exists():
                raise FileNotFoundError(
                    f"Shared modality.json does not exist: {modality_path}"
                )
        else:
            modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        # Resolve dataset statistics paths but defer reading until something
        # actually asks for stats (see ``_ensure_stats_loaded``). open-loop eval
        # and other consumers that only need raw trajectories shouldn't need
        # stats.json to exist.
        if self.stats_dir is not None:
            ext_dir = self.stats_dir / self.dataset_path.name
            self._stats_path = ext_dir / "stats.json"
            self._relative_stats_path = ext_dir / "relative_stats.json"
        else:
            self._stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
            self._relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        self._stats: dict | None = None

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.mask_path_pattern = self.info_meta.get("mask_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    @property
    def stats(self) -> dict:
        """Lazily load per-column stats from ``stats.json`` (+ ``relative_stats.json``).

        Only consumers that actually call ``get_dataset_statistics()`` or touch
        ``self.stats`` pay the cost of requiring the file on disk. open-loop
        evaluation, which reads raw unnormalized actions directly, doesn't.
        """
        if self._stats is None:
            if not self._stats_path.exists():
                raise FileNotFoundError(
                    f"{self._stats_path} does not exist for {self.dataset_path}, "
                    f"please use gr00t/data/stats.py to generate it"
                )
            with open(self._stats_path, "r") as f:
                self._stats = json.load(f)
            if self._relative_stats_path.exists():
                with open(self._relative_stats_path, "r") as f:
                    self._stats["relative_action"] = json.load(f)
        return self._stats

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid
        """
        # Filter out any modalities not handled by the dataset loader.
        unknown_modalities = [
            m for m in modality_configs if m not in ALLOWED_MODALITIES
        ]
        if unknown_modalities:
            logging.debug(
                f"Skipping modalities not supported by dataset loader: {unknown_modalities}"
            )
            modality_configs = {
                k: v for k, v in modality_configs.items() if k in ALLOWED_MODALITIES
            }
        for modality in modality_configs:
            if modality == "language":
                # Language modality has special constraints.
                # Some embodiments (e.g. OXE_DROID) define multiple language keys for
                # training-time augmentation. At inference we only use the first key.
                assert (
                    len(modality_configs[modality].modality_keys) >= 1
                ), "Language modality must have at least one key"
                if len(modality_configs[modality].modality_keys) > 1:
                    logging.warning(
                        f"Language modality has {len(modality_configs[modality].modality_keys)} keys, "
                        f"only the first key will be used: {modality_configs[modality].modality_keys[0]}"
                    )
                    modality_configs[modality] = ModalityConfig(
                        delta_indices=modality_configs[modality].delta_indices,
                        modality_keys=[modality_configs[modality].modality_keys[0]],
                        sin_cos_embedding_keys=modality_configs[
                            modality
                        ].sin_cos_embedding_keys,
                        mean_std_embedding_keys=modality_configs[
                            modality
                        ].mean_std_embedding_keys,
                        action_configs=(
                            modality_configs[modality].action_configs[:1]
                            if modality_configs[modality].action_configs is not None
                            else None
                        ),
                    )
                assert modality_configs[modality].delta_indices == [
                    0
                ], "Only single timestep is supported for language modality"

        # Build mapping from config video keys to dataset modality_meta video keys.
        # This handles the case where the model's pretrained config uses different
        # video key names than the dataset's modality.json (e.g., N1.6 vs N1.7 naming).
        self._video_key_mapping: dict[str, str] = {}
        if "video" in modality_configs and "video" in self.modality_meta:
            config_keys = modality_configs["video"].modality_keys
            meta_keys = list(self.modality_meta["video"].keys())
            needs_mapping = any(
                k not in self.modality_meta["video"] for k in config_keys
            )
            if needs_mapping:
                assert len(config_keys) == len(meta_keys), (
                    f"Cannot auto-map video keys: config has {len(config_keys)} keys "
                    f"{config_keys} but dataset modality meta has {len(meta_keys)} keys "
                    f"{meta_keys}. Counts must match for positional mapping."
                )
                for config_key, meta_key in zip(config_keys, meta_keys):
                    self._video_key_mapping[config_key] = meta_key
                logging.warning(
                    f"Video key mismatch between model config and dataset. "
                    f"Auto-mapping by position: {self._video_key_mapping}"
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Optional per-group temporal shift: if ``time_shift`` is set on a group,
        row ``t`` of the output is sourced from row ``t - time_shift`` of the
        primary column, and the first ``time_shift`` rows fall back to the
        (identically-processed) column named by ``fallback_key``. Lets e.g.
        ``state.body_pos`` be redirected to a delayed copy of the commanded
        action to simulate a control-loop latency at training time.

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        def _process_column(series: pd.Series, group_info: dict) -> pd.Series:
            """Apply the group's slice / rotation transform per element.

            Factored out so we can call it twice when ``time_shift`` is set —
            once on the primary column, once on the fallback column that
            supplies the first ``time_shift`` rows.
            """
            rotation_type = group_info.get("rotation_type", None)
            quat_order = group_info.get("quat_order", "xyzw")

            if not isinstance(series.iloc[0], np.ndarray):
                return series  # strings/scalars pass through unchanged

            if rotation_type == "quat_to_rot6d":
                return series.map(
                    lambda x, gi=group_info, qo=quat_order: _quat_to_rot6d(
                        _apply_index_slice(x, gi), order=qo
                    )
                )
            if rotation_type == "quat_slice":
                # 4D quaternion pass-through: slice the raw quat, reorder to
                # scipy-native xyzw so downstream consumers (both training
                # and eval plotting) see a single, consistent order. The
                # source's ``quat_order`` says how the raw column is stored
                # (typically ``wxyz`` for WBC cmd, ``xyzw`` for IMU); after
                # this step it's ALWAYS xyzw. Model regresses the 4D quat
                # directly — no Gram-Schmidt, no double-cover-fix, no
                # rot6d expansion. Sign hemisphere is NOT canonicalized here;
                # if the raw trajectory happens to hop between q and -q, the
                # model will see the jump. Add canonicalization at the
                # dataset-prep stage if that turns out to hurt training.
                def _slice_and_reorder(x, gi=group_info, qo=quat_order):
                    q = _apply_index_slice(x, gi)
                    if qo.lower() == "wxyz":
                        return np.stack([q[1], q[2], q[3], q[0]], axis=-1).astype(
                            np.float32
                        )
                    return q.astype(np.float32)

                return series.map(_slice_and_reorder)
            if rotation_type == "xyz_quat_to_xyz_rot6d":
                # Composite group: translation (xyz, 3D) + quaternion (4D)
                # in the raw array become translation (3D) + rot6d (6D)
                # concatenated as ``[xyz, rot6d]`` -- the layout
                # expected by ``ActionFormat.XYZ_ROT6D``.
                xyz_info = {
                    k: group_info[k]
                    for k in ("indices", "start", "end")
                    if k in group_info
                }
                if "translation_indices" in group_info:
                    xyz_info = {"indices": group_info["translation_indices"]}
                elif "translation_start" in group_info:
                    xyz_info = {
                        "start": group_info["translation_start"],
                        "end": group_info["translation_end"],
                    }
                rot_info = {}
                if "rotation_indices" in group_info:
                    rot_info = {"indices": group_info["rotation_indices"]}
                elif "rotation_start" in group_info:
                    rot_info = {
                        "start": group_info["rotation_start"],
                        "end": group_info["rotation_end"],
                    }

                def _build(x, xi=xyz_info, ri=rot_info, qo=quat_order):
                    xyz = _apply_index_slice(x, xi).astype(np.float32)
                    rot6d = _quat_to_rot6d(
                        _apply_index_slice(x, ri), order=qo
                    )
                    return np.concatenate([xyz, rot6d], axis=-1)

                return series.map(_build)
            # Plain slice — no rotation conversion.
            return series.map(
                lambda x, gi=group_info: _apply_index_slice(x, gi)
            )

        for group_name in joint_groups:
            if group_name not in modality_info:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )
                continue

            group_info = modality_info[group_name]
            original_key = group_info.get(
                "original_key", DEFAULT_COLUMN_NAMES[modality_type]
            )
            processed = _process_column(df[original_key], group_info)

            # Optional temporal shift: replace row t with the primary column's
            # row t - time_shift, and fill the first time_shift rows from the
            # (identically-processed) fallback column. Only applied when
            # values are arrays — string/scalar columns are shift-inert.
            time_shift = int(group_info.get("time_shift", 0))
            if time_shift > 0 and isinstance(processed.iloc[0], np.ndarray):
                fallback_key = group_info.get("fallback_key")
                if not fallback_key:
                    raise ValueError(
                        f"{modality_type}.{group_name}: 'time_shift={time_shift}' "
                        f"requires 'fallback_key' to supply the first "
                        f"{time_shift} rows (no earlier history exists)."
                    )
                if fallback_key not in df.columns:
                    raise KeyError(
                        f"{modality_type}.{group_name}: fallback_key "
                        f"'{fallback_key}' not found in parquet columns"
                    )
                # ``fallback_overrides`` lets the fallback column be processed
                # with different per-element params than the primary — e.g.
                # sensor IMU stores pelvis quat as xyzw while the WBC cmd
                # stores it as wxyz, and both need to end up as the same
                # 6D rot6d. Any key present in ``fallback_overrides`` shadows
                # the matching field in ``group_info``.
                fallback_group_info = {
                    **group_info,
                    **group_info.get("fallback_overrides", {}),
                }
                fallback_processed = _process_column(
                    df[fallback_key], fallback_group_info
                )
                n = len(processed)
                shift = min(time_shift, n)
                primary_arrays = processed.to_numpy(dtype=object)
                fallback_arrays = fallback_processed.to_numpy(dtype=object)
                shifted = np.empty(n, dtype=object)
                for t in range(n):
                    if t < shift:
                        shifted[t] = fallback_arrays[t]
                    else:
                        shifted[t] = primary_arrays[t - shift]
                processed = pd.Series(shifted, index=processed.index)

            joint_data[group_name] = processed

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data
        """
        # Load raw parquet data using chunking pattern
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()

        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert (
                    subkey in self.modality_meta["annotation"]
                ), f"Key {subkey} not found in language modality"
                original_key = self.modality_meta["annotation"][subkey].get(
                    "original_key", key
                )
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            keys_to_extract = list(self.modality_configs[modality_type].modality_keys)
            # State-only: include reference_only_keys (loaded for RELATIVE action
            # reference lookup but excluded from the encoder concat downstream).
            if modality_type == "state":
                ref_only = getattr(
                    self.modality_configs[modality_type], "reference_only_keys", None
                )
                if ref_only:
                    for k in ref_only:
                        if k not in keys_to_extract:
                            keys_to_extract.append(k)
            joint_groups_df = self._extract_joint_groups(
                original_df,
                keys_to_extract,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[
                    joint_group
                ]

        return loaded_df

    def _load_video_data(
        self, episode_index: int, indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        image_keys = self.modality_configs["video"].modality_keys

        # GR00T_DECODE_SHORT_EDGE (int, e.g. 384): if set, resize every decoded
        # frame so its shorter edge equals this value BEFORE it enters the
        # dataloader cache. Native-resolution cameras (e.g. 2400x1800) blow up
        # per-worker RAM to ~25GB per episode; short-edge=384 cuts that ~50x
        # with negligible signal loss since the model's Resize/Crop target is
        # 224-256 anyway (see gr00t/model/gr00t_n1d7/image_augmentations.py).
        _short_edge_env = os.environ.get("GR00T_DECODE_SHORT_EDGE", "").strip()
        resize_short_edge = int(_short_edge_env) if _short_edge_env else None

        for image_key in image_keys:
            # Resolve the original key used in video file naming.
            # Use the video key mapping if the config key differs from the dataset meta key.
            meta_key = self._video_key_mapping.get(image_key, image_key)
            original_key = self.modality_meta["video"][meta_key].get(
                "original_key", f"observation.images.{meta_key}"
            )
            assert (
                original_key in self.feature_config
            ), f"Original key {original_key} not found in feature config"

            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx,
                video_key=original_key,
                episode_index=episode_index,
            )
            video_path = self.dataset_path / video_filename

            # Decode video frames at specified timestamps
            video_data[image_key] = get_frames_by_indices(
                str(video_path),
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
                resize_short_edge=resize_short_edge,
            )

        return video_data

    def _load_mask_file(self, mask_path: Path, indices: np.ndarray) -> np.ndarray:
        """Load masks from npz/npy file at specified indices."""
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
        suffix = mask_path.suffix.lower()
        if suffix not in {".npz", ".npy"}:
            raise ValueError(f"Only .npz or .npy mask files are supported: {mask_path}")

        if suffix == ".npy":
            masks = np.load(mask_path)
        else:
            npz_data = np.load(mask_path)
            if "arr_0" in npz_data:
                masks = npz_data["arr_0"]
            elif len(npz_data.files) == 1:
                masks = npz_data[npz_data.files[0]]
            else:
                raise ValueError(
                    f"Mask npz must contain a single array or 'arr_0': {mask_path}"
                )

        if masks.ndim == 2:
            masks = masks[None, ...]

        return masks[indices]

    def _load_mask_data(
        self, episode_index: int, indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Load mask data for all configured mask views at specified indices.
        """
        mask_data = {}

        if not self.mask_path_pattern or "mask" not in self.modality_configs:
            return mask_data

        chunk_idx = episode_index // self.chunk_size
        mask_keys = self.modality_configs["mask"].modality_keys

        for mask_key in mask_keys:
            mask_meta = self.modality_meta.get("mask", {}).get(mask_key, {})
            original_key = mask_meta.get("original_key", mask_key)
            mask_filename = self.mask_path_pattern.format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
                mask_key=original_key,
                video_key=original_key,
            )
            mask_path = self.dataset_path / mask_filename
            mask_data[mask_key] = self._load_mask_file(mask_path, indices)

        return mask_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                group_info = self.modality_meta[modality][joint_key]
                # Determine which statistics key to use
                if group_info.get("original_key", None) is not None:
                    stats_key = group_info["original_key"]
                else:
                    stats_key = mapping[modality]

                rotation_type = group_info.get("rotation_type", None)

                if rotation_type == "quat_to_rot6d":
                    # The raw data is a 4D quaternion but we feed a 6D
                    # rotation to the model. We can't derive per-dim stats of
                    # 6D from the raw 4D stats, so use the theoretical bounds
                    # of a unit-rotation-matrix row (each component in
                    # [-1, 1]). These stats are only used for min/max
                    # normalization.
                    dim = 6
                    dataset_statistics[modality][joint_key] = {
                        "mean": [0.0] * dim,
                        "std": [1.0] * dim,
                        "min": [-1.0] * dim,
                        "max": [1.0] * dim,
                        "q01": [-1.0] * dim,
                        "q99": [1.0] * dim,
                    }
                    continue

                if rotation_type == "xyz_quat_to_xyz_rot6d":
                    # Composite: 3D translation stats come from the raw
                    # stats (at the translation indices), 6D rotation stats
                    # are theoretical [-1, 1] bounds.
                    raw_stats = self.stats[stats_key]
                    if "translation_indices" in group_info:
                        xyz_sel = np.asarray(
                            group_info["translation_indices"], dtype=np.int64
                        )
                        xyz_stats = {
                            k: np.asarray(v)[xyz_sel].tolist()
                            for k, v in raw_stats.items()
                        }
                    else:
                        s = group_info["translation_start"]
                        e = group_info["translation_end"]
                        xyz_stats = {
                            k: list(v[s:e]) for k, v in raw_stats.items()
                        }
                    rot_default = {
                        "mean": [0.0] * 6,
                        "std": [1.0] * 6,
                        "min": [-1.0] * 6,
                        "max": [1.0] * 6,
                        "q01": [-1.0] * 6,
                        "q99": [1.0] * 6,
                    }
                    combined = {}
                    for stat_type in xyz_stats.keys():
                        combined[stat_type] = (
                            list(xyz_stats[stat_type]) + list(rot_default[stat_type])
                        )
                    dataset_statistics[modality][joint_key] = combined
                    continue

                # Extract the relevant slice of statistics
                if "indices" in group_info:
                    sel = np.asarray(group_info["indices"], dtype=np.int64)
                    for stat_type in self.stats[stats_key].keys():
                        dataset_statistics[modality][joint_key][stat_type] = (
                            np.asarray(self.stats[stats_key][stat_type])[sel].tolist()
                        )
                else:
                    start_idx, end_idx = group_info["start"], group_info["end"]
                    for stat_type in self.stats[
                        stats_key
                    ].keys():  # mean, std, min, max, q01, q99
                        dataset_statistics[modality][joint_key][stat_type] = self.stats[
                            stats_key
                        ][stat_type][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = random.choice(episode_meta["tasks"])
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = (
                    sub_task["start"],
                    sub_task["end"],
                    sub_task["text"],
                )
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    @staticmethod
    def append_subtask_prompts(
        global_tasks: list[str],
        episode_meta: dict,
        subtask_text_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """Combine tasks.jsonl prompts with exact current-frame subtasks."""
        nframes = len(global_tasks)
        subtask_by_frame: list[str | None] = [None] * nframes
        subtask_text_overrides = subtask_text_overrides or {}

        if "sub_tasks" not in episode_meta:
            raise ValueError(
                f"Episode {episode_meta.get('episode_index')} has no sub_tasks metadata"
            )

        for subtask in episode_meta["sub_tasks"]:
            start = int(subtask["start"])
            end = int(subtask["end"])
            source_text = str(subtask["text"]).strip()
            subtask_text = subtask_text_overrides.get(source_text, source_text)
            if start < 0 or end <= start or not source_text or not subtask_text:
                raise ValueError(
                    f"Invalid subtask in episode {episode_meta.get('episode_index')}: "
                    f"{subtask!r}"
                )

            clipped_start = min(start, nframes)
            clipped_end = min(end, nframes)
            for frame_index in range(clipped_start, clipped_end):
                if subtask_by_frame[frame_index] is not None:
                    raise ValueError(
                        f"Overlapping subtasks in episode "
                        f"{episode_meta.get('episode_index')} at frame {frame_index}"
                    )
                subtask_by_frame[frame_index] = subtask_text

        missing_frame = next(
            (i for i, text in enumerate(subtask_by_frame) if text is None), None
        )
        if missing_frame is not None:
            raise ValueError(
                f"Subtasks do not cover episode {episode_meta.get('episode_index')} "
                f"frame {missing_frame}"
            )

        prompts = []
        for global_task, subtask_text in zip(global_tasks, subtask_by_frame):
            global_task = str(global_task).strip()
            if not global_task:
                raise ValueError(
                    f"Empty global task in episode {episode_meta.get('episode_index')}"
                )
            prompts.append(f"{global_task} Subtask: {subtask_text}")
        return prompts

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(
                    episode_meta, len(df), lang_key
                )
                df["language." + lang_key] = new_languages
            if self.subtask_conditioned:
                language_column = "language." + lang_key
                df[language_column] = self.append_subtask_prompts(
                    df[language_column].tolist(),
                    episode_meta,
                    self.subtask_text_overrides,
                )

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(
                df
            ), f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        # Load synchronized mask data
        mask_data = self._load_mask_data(episode_id, np.arange(actual_length))
        for key in mask_data.keys():
            assert len(mask_data[key]) == len(
                df
            ), f"Mask data for {key} has length {len(mask_data[key])} but dataframe has length {len(df)}"
            df[f"mask.{key}"] = [mask for mask in mask_data[key]]

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
