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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from gr00t.data.types import ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios.  For convenience the *legacy*
    single-dataset fields still exist; if ``datasets`` is non-empty they take
    precedence.
    """

    # Path to the dataset root directory (can be strings or dicts for complex configs)
    dataset_paths: List[Any]

    # Robot embodiment identifier (e.g. "gr1", "franka")
    embodiment_tag: Optional[str] = None

    # Relative sampling probability (will be normalised across the list)
    mix_ratio: float = 1.0

    dataset_type: str = "physical_embodiment"

    # Optional validation dataset path for open-loop evaluation
    # If not provided, falls back to dataset_paths for evaluation
    val_dataset_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration (supports single or multiple datasets)."""

    # Leave empty by default for backwards-compatibility with the original
    # single-dataset workflow.  Users can supply one or more configs via CLI or
    # YAML when they need mixing.
    datasets: List[SingleDatasetConfig] = field(default_factory=list)

    # Modality configs
    # There are three sources of modality configs:
    # 1. Default modality configs in code: gr00t/configs/data/embodiment_configs.py
    # 2. Modality configs supplied through command line: --data.modality_configs (although rare and inconvenient)
    # 1 and 2 are unified through `config.data.modality_configs`.
    # 3. modality configs saved in the pretrained checkpoint.
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        default_factory=lambda: MODALITY_CONFIGS
    )

    # Sharded dataset configuration
    download_cache: bool = False
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)

    # Override statistics from the pretrained checkpoint
    override_pretraining_statistics: bool = True

    # General task / mode config (shared across datasets)
    mode: str = "single_turn"
    random_chop: float = 0.0
    mock_dataset_mode: bool = (
        False  # if True, cache the first datapoint of each dataset and always return one of them to simulate best-case dataloading
    )

    # Data loading
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str = "fork"  # Options: "fork", "spawn", and "forkserver"
    allow_padding: bool = True

    # If set, every dataset uses this file as its ``modality.json`` instead
    # of ``<dataset_path>/meta/modality.json``. Useful when training with
    # multiple datasets that share the same modality layout but you don't
    # want to duplicate the file under each dataset's ``meta/`` dir. Other
    # meta files (info.json / episodes.jsonl / tasks.jsonl / stats.json)
    # are still read from each dataset's own ``meta/`` dir.
    modality_meta_path: str | None = None

    # If set, per-dataset ``stats.json`` and ``relative_stats.json`` are
    # written to and read from ``<stats_dir>/<dataset_name>/`` instead of
    # the dataset's own ``meta/`` dir. This lets you train on read-only
    # datasets by diverting generated stats to a writable location. Each
    # dataset still gets its OWN stats files (grouped under its directory
    # name); the final global stats are still merged in memory at load time.
    stats_dir: str | None = None

    # If set, per-dataset ``tasks.jsonl`` is read from
    # ``<tasks_dir>/<dataset_name>/tasks.jsonl`` instead of
    # ``<dataset_path>/meta/tasks.jsonl``. Useful for read-only datasets
    # or when you want to override task descriptions without modifying
    # the original dataset. All other meta files still come from each
    # dataset's own ``meta/`` dir. Optional sibling ``subtasks.jsonl`` files
    # can override prompt wording while preserving episodes.jsonl.
    tasks_dir: str | None = None

    # Append the current frame's ``sub_tasks`` text from episodes.jsonl to the
    # global tasks.jsonl instruction as ``Subtask: ...``.
    subtask_conditioned: bool = False

    # Optional bad-frame source: a legacy interval manifest, a version-2
    # all-frame quality-label JSON, a directory containing per-dataset label
    # files, or comma-separated sources. ShardedSingleStepDataset removes every
    # sample whose temporal deltas touch an unpreferred frame.
    bad_segments_path: str | None = None

    # Optional Advantage Conditioning labels. This path is intentionally
    # separate from bad_segments_path: bad segments alter the sampling index,
    # while motion-quality labels retain samples and add a language condition.
    advantage_conditioned: bool = False
    motion_quality_labels_path: str | None = None
    advantage_horizon: int = 16
    advantage_unpreferred_overlap_ratio: float = 1.0 / 3.0
    advantage_unknown_policy: str = "unconditioned"

    # RTC sampling normally happens inside the action head. With Advantage
    # Conditioning enabled the dataset samples the same delay first, so the
    # prompt can describe [t + delay, t + delay + advantage_horizon).
    rtc_max_delay: int = 0
    rtc_delay_decay: float = 0.0

    # Optional path to a user-provided modality config .py file. Stored here
    # (in addition to FinetuneConfig) so worker processes spawned for
    # parallel stats generation can re-import it to register custom
    # embodiment tags like NEW_EMBODIMENT. Not used by the main training
    # path (which calls ``load_modality_config`` in ``launch_finetune.py``
    # before instantiating DataConfig).
    modality_config_path: str | None = None

    # Number of parallel processes used for the *one-time* per-dataset stats
    # generation (stats.json / relative_stats.json). 0 / 1 = serial (original
    # behaviour). Only effective when multiple datasets are missing cached
    # stats; already-cached datasets are skipped regardless. Each worker
    # re-imports the modality config, so keep this modest (<= #CPU cores,
    # and <= #uncached datasets).
    stats_num_workers: int = 0

    # Subsample ratio for the dataset
    subsample_ratio: float = 1.0

    # DP Image Config
    image_crop_size: List[int] = field(default_factory=lambda: [244, 244])
    image_target_size: List[int] = field(default_factory=lambda: [224, 224])
    video_backend: str = "torchcodec"
