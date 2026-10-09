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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader


MOTION_QUALITY_TO_ID = {
    "unknown": -1,
    "preferred": 0,
    "unpreferred": 1,
}


def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = True,
) -> VLAStepData:
    step_data = {}

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # Sample timesteps according to delta indices configuration
        indices_to_load = [
            step_index + delta_index for delta_index in config.delta_indices
        ]
        if allow_padding:
            indices_to_load = [
                max(0, min(idx, len(episode_data) - 1)) for idx in indices_to_load
            ]
        keys_to_extract = list(config.modality_keys)
        # State-only: include reference_only_keys so RELATIVE action processing
        # can look them up as state[state_key]. Not fed to state encoder — see
        # processing_gr00t_n1d7 which concats only modality_keys.
        if modality == "state":
            ref_only = getattr(config, "reference_only_keys", None)
            if ref_only:
                for k in ref_only:
                    if k not in keys_to_extract:
                        keys_to_extract.append(k)
        for key in keys_to_extract:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            if modality in ["state", "action"]:
                # Stack arrays for numerical modalities
                step_data[modality][key] = np.vstack(
                    [
                        np.array(modality_data.iloc[i]).astype(np.float32)
                        for i in range(len(modality_data))
                    ]
                )
            else:
                # Keep as lists for other modalities (video, language)
                step_data[modality][key] = modality_data.tolist()

    # Parse extracted data into VLAStepData structure
    video_data = step_data.get("video", {})
    mask_data = step_data.get("mask", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]

    vla_step_data = VLAStepData(
        images=video_data,
        masks=mask_data if mask_data else None,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return vla_step_data


class ShardedSingleStepDataset(ShardedDataset):
    """
    Single-step dataset that creates shards from individual timesteps across episodes.

    This dataset implementation provides step-level data access for VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Splitting episodes into individual timesteps
    3. Organizing timesteps into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    The sharding strategy ensures balanced shard sizes while maintaining randomization
    across episodes and timesteps within episodes. Each shard contains a mix of
    timesteps from different episodes to improve training diversity.

    Key features:
    - Step-level data access (vs episode-level)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]
        bad_segments_path: Optional bad-frame source. Accepts the legacy
            inclusive frame-interval manifest, a stumble-review version-2
            ``<dataset>_all_frames.json`` file, a directory containing those
            per-dataset files, or a comma-separated combination. Any sample
            whose configured temporal indices touch an unpreferred frame is
            removed before sharding.

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = True,
        modality_meta_path: str | Path | None = None,
        stats_dir: str | Path | None = None,
        tasks_dir: str | Path | None = None,
        subtask_conditioned: bool = False,
        bad_segments_path: str | Path | None = None,
        advantage_conditioned: bool = False,
        motion_quality_labels_path: str | Path | None = None,
        advantage_horizon: int = 16,
        advantage_unpreferred_overlap_ratio: float = 1.0 / 3.0,
        advantage_unknown_policy: str = "unconditioned",
        rtc_max_delay: int = 0,
        rtc_delay_decay: float = 0.0,
    ):
        """Initialize single-step dataset with sharding configuration."""
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        self.action_delta_indices = np.asarray(action_delta_indices, dtype=np.int64)

        # Check every temporal frame actually loaded by a sample. For the
        # Sonic-A3 config this is [0..39] from the 40-frame action horizon,
        # plus the current-frame state/video/language indices.
        self.sample_delta_indices = np.asarray(
            sorted(
                {
                    int(delta_index)
                    for config in modality_configs.values()
                    for delta_index in config.delta_indices
                }
            ),
            dtype=np.int64,
        )
        self.bad_segments_path = str(bad_segments_path) if bad_segments_path else None
        self.bad_segments_resolved_path: Path | None = None
        self.bad_frame_intervals = self._load_bad_frame_intervals()

        self.advantage_conditioned = bool(advantage_conditioned)
        self.motion_quality_labels_path = (
            str(motion_quality_labels_path) if motion_quality_labels_path else None
        )
        self.advantage_horizon = int(advantage_horizon)
        self.advantage_unpreferred_overlap_ratio = float(
            advantage_unpreferred_overlap_ratio
        )
        self.advantage_unknown_policy = advantage_unknown_policy
        self.rtc_max_delay = int(rtc_max_delay)
        self.rtc_delay_decay = float(rtc_delay_decay)
        self.advantage_rng = np.random.default_rng(seed + 17_071)
        self._validate_advantage_config()

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            modality_meta_path=modality_meta_path,
            stats_dir=stats_dir,
            tasks_dir=tasks_dir,
            subtask_conditioned=subtask_conditioned,
        )
        self.motion_quality_by_episode = self._load_motion_quality_labels()
        self.unpreferred_segments_by_episode = self._build_unpreferred_segments()

        # Create balanced shards from episode timesteps
        self.shard_dataset()

    def _validate_advantage_config(self) -> None:
        if not self.advantage_conditioned:
            return
        if not self.motion_quality_labels_path:
            raise ValueError(
                "advantage_conditioned=True requires motion_quality_labels_path"
            )
        if self.advantage_horizon <= 0:
            raise ValueError("advantage_horizon must be positive")
        if not 0.0 < self.advantage_unpreferred_overlap_ratio <= 1.0:
            raise ValueError(
                "advantage_unpreferred_overlap_ratio must be in (0, 1]"
            )
        if self.advantage_unknown_policy not in {"unconditioned", "unknown"}:
            raise ValueError(
                "advantage_unknown_policy must be 'unconditioned' or 'unknown'"
            )
        if self.rtc_max_delay < 0:
            raise ValueError("rtc_max_delay must be non-negative")
        largest_delay = max(0, self.rtc_max_delay - 1)
        if largest_delay + self.advantage_horizon > len(self.action_delta_indices):
            raise ValueError(
                "RTC-aligned Advantage Conditioning window does not fit inside "
                f"the action chunk: max_delay_index={largest_delay}, "
                f"advantage_horizon={self.advantage_horizon}, "
                f"action_steps={len(self.action_delta_indices)}"
            )

    def _resolve_motion_quality_file(self) -> Path:
        """Find the label file whose ``dataset`` matches this dataset basename."""
        assert self.motion_quality_labels_path is not None
        dataset_name = Path(self.dataset_path).name
        files: list[Path] = []
        for raw_source in self.motion_quality_labels_path.split(","):
            source = Path(raw_source.strip()).expanduser()
            if not raw_source.strip():
                continue
            if source.is_dir():
                for filename in (
                    f"{dataset_name}_all_frames.json",
                    f"{dataset_name}.json",
                ):
                    candidate = source / filename
                    if candidate.is_file():
                        files.append(candidate)
            elif source.is_file():
                files.append(source)
            else:
                raise FileNotFoundError(f"Motion-quality label source not found: {source}")

        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("dataset") == dataset_name:
                return path
        raise FileNotFoundError(
            f"No motion-quality label JSON for dataset {dataset_name!r} in "
            f"{self.motion_quality_labels_path!r}"
        )

    def _load_motion_quality_labels(self) -> dict[int, np.ndarray]:
        if not self.advantage_conditioned:
            return {}
        label_path = self._resolve_motion_quality_file()
        with label_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("version", 0)) != 2:
            raise ValueError(
                f"Motion-quality labels must use version 2, got {payload.get('version')} "
                f"in {label_path}"
            )

        expected_lengths = {
            int(meta["episode_index"]): int(meta["length"])
            for meta in self.episode_loader.episodes_metadata
        }
        result: dict[int, np.ndarray] = {}
        for episode in payload.get("episodes", []):
            episode_id = int(episode["episode"])
            raw_labels = episode.get("labels", [])
            try:
                codes = np.asarray(
                    [MOTION_QUALITY_TO_ID[str(label)] for label in raw_labels],
                    dtype=np.int8,
                )
            except KeyError as error:
                raise ValueError(
                    f"Unsupported motion-quality label {error.args[0]!r} in "
                    f"{label_path}, episode {episode_id}"
                ) from error
            expected = expected_lengths.get(episode_id)
            if expected is None:
                raise ValueError(
                    f"Label file {label_path} contains unknown episode {episode_id}"
                )
            if len(codes) != expected:
                raise ValueError(
                    f"Frame count mismatch in {label_path}, episode {episode_id}: "
                    f"labels={len(codes)}, dataset={expected}"
                )
            result[episode_id] = codes

        missing = sorted(set(expected_lengths) - set(result))
        if missing:
            raise ValueError(
                f"Motion-quality label file {label_path} is missing episodes: {missing[:10]}"
            )
        print(
            f"[advantage] dataset={Path(self.dataset_path).name}, labels={label_path}, "
            f"episodes={len(result)}, horizon={self.advantage_horizon}, "
            f"unknown_policy={self.advantage_unknown_policy}"
        )
        return result

    def _build_unpreferred_segments(self) -> dict[int, list[tuple[int, int]]]:
        """Precompute inclusive contiguous unpreferred intervals per episode."""
        result: dict[int, list[tuple[int, int]]] = {}
        target = MOTION_QUALITY_TO_ID["unpreferred"]
        for episode_id, codes in self.motion_quality_by_episode.items():
            mask = codes == target
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            result[episode_id] = [
                (int(start), int(end)) for start, end in zip(starts, ends)
            ]
        return result

    def _sample_advantage_rtc_delay(self) -> int:
        """Sample RTC delay before prompt construction so both stay aligned."""
        if self.rtc_max_delay <= 0:
            return 0
        if self.rtc_delay_decay <= 0:
            return int(self.advantage_rng.integers(0, self.rtc_max_delay))
        delays = np.arange(self.rtc_max_delay, dtype=np.float64)
        probs = np.exp(-self.rtc_delay_decay * delays)
        probs /= probs.sum()
        return int(self.advantage_rng.choice(self.rtc_max_delay, p=probs))

    def _apply_motion_quality_condition(
        self,
        vla_step_data: VLAStepData,
        episode_id: int,
        step_index: int,
    ) -> None:
        """Append the quality prompt and retain per-action labels as metadata."""
        if not self.advantage_conditioned:
            return
        delay = self._sample_advantage_rtc_delay()
        frame_indices = step_index + self.action_delta_indices
        frame_codes = self.motion_quality_by_episode[episode_id][frame_indices]
        window = frame_codes[delay : delay + self.advantage_horizon]
        advantage_has_unpreferred = np.any(
            window == MOTION_QUALITY_TO_ID["unpreferred"]
        )

        # The prompt normally describes the first advantage_horizon actions
        # after RTC's frozen prefix. Do not silently call a chunk preferred,
        # however, when a substantial part of a later unpreferred segment is
        # still supervised by the postfix loss. The denominator is the length
        # of that complete global unpreferred segment, not the chunk length.
        postfix_frame_indices = frame_indices[delay:]
        max_unpreferred_overlap_ratio = 0.0
        for segment_start, segment_end in self.unpreferred_segments_by_episode.get(
            episode_id, []
        ):
            overlap_frames = int(
                np.count_nonzero(
                    (postfix_frame_indices >= segment_start)
                    & (postfix_frame_indices <= segment_end)
                )
            )
            segment_frames = segment_end - segment_start + 1
            max_unpreferred_overlap_ratio = max(
                max_unpreferred_overlap_ratio,
                overlap_frames / segment_frames,
            )
        substantial_late_overlap = (
            max_unpreferred_overlap_ratio
            > self.advantage_unpreferred_overlap_ratio
        )

        if advantage_has_unpreferred or substantial_late_overlap:
            label = "unpreferred"
        elif np.any(window == MOTION_QUALITY_TO_ID["unknown"]):
            label = "unknown"
        else:
            label = "preferred"

        should_prompt = label != "unknown" or self.advantage_unknown_policy == "unknown"
        if should_prompt:
            task_text = (vla_step_data.text or "").strip()
            suffix = f"Motion quality: {label}."
            vla_step_data.text = f"{task_text} {suffix}".strip()

        vla_step_data.metadata.update(
            {
                "motion_quality_ids": frame_codes.astype(np.int64),
                "motion_quality_label_id": np.int64(MOTION_QUALITY_TO_ID[label]),
                "motion_quality_label": label,
                "motion_quality_max_unpreferred_overlap_ratio": np.float32(
                    max_unpreferred_overlap_ratio
                ),
                "rtc_delay": np.int64(delay),
            }
        )

    @staticmethod
    def _parse_bad_intervals(
        dataset_name: str,
        source_path: Path,
        episodes: dict[str, list[dict]],
    ) -> dict[int, list[tuple[int, int]]]:
        result: dict[int, list[tuple[int, int]]] = {}
        for episode, intervals in episodes.items():
            parsed = []
            for interval in intervals:
                start = int(interval["start_frame"])
                end = int(interval["end_frame"])
                if start < 0 or end < start:
                    raise ValueError(
                        f"Invalid bad-frame interval for {dataset_name} "
                        f"episode {episode} in {source_path}: [{start}, {end}]"
                    )
                parsed.append((start, end))
            result[int(episode)] = parsed
        return result

    def _resolve_bad_segment_files(self) -> list[Path]:
        """Resolve file, directory, or comma-separated bad-frame sources."""
        assert self.bad_segments_path is not None
        dataset_name = Path(self.dataset_path).name
        files: list[Path] = []
        for raw_source in self.bad_segments_path.split(","):
            if not raw_source.strip():
                continue
            source = Path(raw_source.strip()).expanduser()
            if source.is_dir():
                for filename in (
                    f"{dataset_name}_all_frames.json",
                    f"{dataset_name}.json",
                ):
                    candidate = source / filename
                    if candidate.is_file():
                        files.append(candidate)
            elif source.is_file():
                files.append(source)
            else:
                raise FileNotFoundError(f"Bad-frame source not found: {source}")
        return list(dict.fromkeys(files))

    def _load_quality_bad_intervals(
        self,
        payload: dict,
        source_path: Path,
        dataset_name: str,
    ) -> dict[int, list[tuple[int, int]]]:
        """Extract inclusive unpreferred segments from an all-frame export."""
        if int(payload.get("version", 0)) != 2:
            raise ValueError(
                f"Bad-frame quality labels must use version 2, got "
                f"{payload.get('version')} in {source_path}"
            )

        episodes: dict[str, list[dict]] = {}
        for episode in payload.get("episodes", []):
            episode_id = int(episode["episode"])
            segments = episode.get("segments")
            if not isinstance(segments, list):
                raise ValueError(
                    f"Bad-frame quality labels in {source_path}, episode "
                    f"{episode_id} must contain a segments list"
                )
            bad_segments = []
            for segment in segments:
                label = str(segment.get("label", ""))
                if label not in MOTION_QUALITY_TO_ID:
                    raise ValueError(
                        f"Unsupported motion-quality label {label!r} in "
                        f"{source_path}, episode {episode_id}"
                    )
                if label == "unpreferred":
                    bad_segments.append(segment)
            episodes[str(episode_id)] = bad_segments
        return self._parse_bad_intervals(dataset_name, source_path, episodes)

    def _load_bad_frame_intervals(self) -> dict[int, list[tuple[int, int]]]:
        """Load inclusive bad-frame intervals for this dataset basename."""
        if self.bad_segments_path is None:
            return {}

        dataset_name = Path(self.dataset_path).name
        for source_path in self._resolve_bad_segment_files():
            with source_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if "datasets" in payload or "interval_semantics" in payload:
                if payload.get("interval_semantics") != "inclusive_frame_indices":
                    raise ValueError(
                        "Bad-segment manifest must use "
                        "interval_semantics='inclusive_frame_indices'"
                    )
                dataset_entry = payload.get("datasets", {}).get(dataset_name)
                if dataset_entry is None:
                    continue
                result = self._parse_bad_intervals(
                    dataset_name,
                    source_path,
                    dataset_entry.get("episodes", {}),
                )
                source_format = "interval-manifest"
            elif payload.get("dataset") == dataset_name:
                result = self._load_quality_bad_intervals(
                    payload,
                    source_path,
                    dataset_name,
                )
                source_format = "all-frame-labels"
            elif "dataset" in payload:
                continue
            else:
                raise ValueError(
                    f"Unsupported bad-frame JSON format in {source_path}"
                )

            self.bad_segments_resolved_path = source_path
            interval_count = sum(len(intervals) for intervals in result.values())
            print(
                f"[bad-segments] dataset={dataset_name}, source={source_path}, "
                f"format={source_format}, intervals={interval_count}"
            )
            return result

        print(
            f"[bad-segments] No matching manifest entry or all-frame label JSON "
            f"for {dataset_name} in {self.bad_segments_path}; no samples filtered."
        )
        return {}

    def get_valid_step_indices(self, episode_index: int) -> np.ndarray:
        """Return starts whose complete temporal sample avoids bad frames.

        A start is rejected if any frame loaded through any modality's
        ``delta_indices`` intersects an inclusive bad-frame interval. This
        makes the filter horizon-aware: with action deltas [0..39], a bad
        frame at ``b`` also invalidates starts ``b-39`` through ``b``.
        """
        effective_length = self.get_effective_episode_length(episode_index)
        starts = np.arange(effective_length, dtype=np.int64)
        if effective_length == 0 or not self.bad_frame_intervals:
            return starts

        episode_meta = self.episode_loader.episodes_metadata[episode_index]
        episode_id = int(episode_meta["episode_index"])
        intervals = self.bad_frame_intervals.get(episode_id, [])
        if not intervals:
            return starts

        original_length = self.episode_loader.get_episode_length(episode_index)
        bad_frames = np.zeros(original_length, dtype=bool)
        for start, end in intervals:
            clipped_start = max(0, start)
            clipped_end = min(original_length - 1, end)
            if clipped_start <= clipped_end:
                bad_frames[clipped_start : clipped_end + 1] = True

        touches_bad_frame = np.zeros(effective_length, dtype=bool)
        for delta_index in self.sample_delta_indices:
            loaded_indices = starts + delta_index
            if self.allow_padding:
                loaded_indices = np.clip(loaded_indices, 0, original_length - 1)
                touches_bad_frame |= bad_frames[loaded_indices]
            else:
                in_bounds = (loaded_indices >= 0) & (
                    loaded_indices < original_length
                )
                touches_bad_frame[in_bounds] |= bad_frames[
                    loaded_indices[in_bounds]
                ]
        return starts[~touches_bad_frame]

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode timesteps across shards.

        The sharding process:
        1. Shuffle episode order for randomization
        2. Split each episode into multiple sub-sequences based on sampling rate
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and timesteps)
        - Reproducible sharding based on seed
        """
        shuffled_episode_indices = self.rng.permutation(
            len(self.episode_loader.episode_lengths)
        )
        num_splits = int(1 / self.episode_sampling_rate)

        assert (
            len(shuffled_episode_indices) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"

        # Calculate the valid starts once. The exclusion is applied before
        # shuffling/splitting, so an invalid horizon can never enter a shard.
        self.valid_step_indices_by_episode = {
            int(ep_idx): self.get_valid_step_indices(int(ep_idx))
            for ep_idx in shuffled_episode_indices
        }
        total_unfiltered_steps = int(
            np.sum(
                [
                    self.get_effective_episode_length(int(idx))
                    for idx in shuffled_episode_indices
                ]
            )
        )
        total_steps = int(
            np.sum(
                [
                    len(self.valid_step_indices_by_episode[int(idx)])
                    for idx in shuffled_episode_indices
                ]
            )
        )
        assert total_steps > 0, (
            f"No valid timesteps remain for dataset {self.dataset_path} after "
            f"applying bad-segment manifest {self.bad_segments_path}"
        )
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute episode sub-sequences across shards
        for ep_idx in shuffled_episode_indices:
            # Split episode timesteps into multiple sub-sequences
            step_indices = self.valid_step_indices_by_episode[int(ep_idx)].copy()
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                # Assign to shard with minimum current length (greedy balancing)
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # Validate shard creation
        assert all(
            shard_lengths[i] > 0 for i in range(num_shards)
        ), "All shards must have length greater than 0"

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        if self.bad_segments_path is not None:
            interval_count = sum(
                len(intervals) for intervals in self.bad_frame_intervals.values()
            )
            offset_summary = (
                f"{int(self.sample_delta_indices.min())}.."
                f"{int(self.sample_delta_indices.max())} "
                f"({len(self.sample_delta_indices)} offsets)"
            )
            print(
                f"[bad-segments] intervals={interval_count}, "
                f"horizon_offsets={offset_summary}, "
                f"removed_starts={total_unfiltered_steps - total_steps}, "
                f"kept_starts={total_steps}/{total_unfiltered_steps}"
            )
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_datapoint(
        self,
        episode_data: pd.DataFrame,
        episode_index: int,
        step_index: int,
    ) -> dict:
        """
        Extract and process a single timestep from episode data.

        Converts raw episode data into a VLAStepData structure and applies
        the configured processor to create model-ready inputs.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract

        Returns:
            Processed datapoint ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
        """
        assert (
            self.processor is not None
        ), "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        episode_id = int(
            self.episode_loader.episodes_metadata[int(episode_index)]["episode_index"]
        )
        self._apply_motion_quality_condition(vla_step_data, episode_id, int(step_index))
        # Apply processor to convert to model inputs
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """
        Load and process all timesteps in a specific shard.

        Loads the required episodes and extracts all timesteps assigned to this shard,
        applying the configured processor to each timestep.

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            # Load episode data once per episode in shard
            episode_data = self.episode_loader[ep_idx]
            for step_index in step_indices:
                datapoints.append(
                    self.get_datapoint(episode_data, int(ep_idx), int(step_index))
                )
        return datapoints

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
