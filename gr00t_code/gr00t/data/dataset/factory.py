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

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier

logger = logging.getLogger(__name__)


def _ensure_modality_config_registered(modality_config_path: str | None) -> None:
    """Re-register a user-provided modality config inside the current process.

    The MODALITY_CONFIGS dict is populated as a side-effect of importing the
    config module (via ``register_modality_config``). When we spawn worker
    processes for parallel stats generation, those workers don't re-run
    ``launch_finetune.py::load_modality_config`` automatically, so we need
    to import the module here so the NEW_EMBODIMENT tag is recognised.
    No-op when called in the parent process (module already imported) or
    when no custom config was supplied.
    """
    if not modality_config_path:
        return
    import importlib
    import sys
    from pathlib import Path

    path = Path(modality_config_path)
    if not (path.exists() and path.suffix == ".py"):
        return
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    importlib.import_module(path.stem)


def _gen_stats_for_one_dataset(
    dataset_path: str,
    embodiment_tag_value: str,
    modality_meta_path: str | None,
    stats_dir: str | None,
    modality_config_path: str | None,
    tasks_dir: str | None = None,
) -> str:
    """Worker entrypoint: generate stats.json + relative_stats.json for a
    single dataset. Returns the dataset_path on success so the caller can
    surface progress.
    """
    _ensure_modality_config_registered(modality_config_path)
    generate_stats(dataset_path, stats_dir=stats_dir)
    generate_rel_stats(
        dataset_path,
        EmbodimentTag(embodiment_tag_value),
        modality_meta_path=modality_meta_path,
        stats_dir=stats_dir,
        tasks_dir=tasks_dir,
    )
    return dataset_path


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def _generate_all_stats(
        self,
        dataset_paths: list[str],
        embodiment_tag_value: str,
    ) -> None:
        """Generate stats for every dataset, optionally in parallel.

        Honours ``config.data.stats_num_workers``. Parallelises across
        datasets using ``ProcessPoolExecutor``; each worker is short-lived
        and processes exactly one dataset. Workers re-import the user's
        modality config so embodiment-tag lookup works.
        """
        modality_meta_path = self.config.data.modality_meta_path
        stats_dir = self.config.data.stats_dir
        tasks_dir = getattr(self.config.data, "tasks_dir", None)
        modality_config_path = getattr(
            self.config.data, "modality_config_path", None
        )
        # The modality config path isn't on DataConfig today; fall back to
        # the finetune-level copy if present on the overall Config object.
        if modality_config_path is None:
            modality_config_path = getattr(
                self.config, "modality_config_path", None
            )

        num_workers = int(getattr(self.config.data, "stats_num_workers", 0) or 0)
        num_workers = min(num_workers, len(dataset_paths))

        if num_workers <= 1:
            for dataset_path in tqdm(
                dataset_paths,
                total=len(dataset_paths),
                desc="Generating stats (serial)",
            ):
                _gen_stats_for_one_dataset(
                    dataset_path=dataset_path,
                    embodiment_tag_value=embodiment_tag_value,
                    modality_meta_path=modality_meta_path,
                    stats_dir=stats_dir,
                    modality_config_path=modality_config_path,
                    tasks_dir=tasks_dir,
                )
            return

        logger.info(
            "Parallel stats generation: %d datasets with %d workers",
            len(dataset_paths),
            num_workers,
        )
        # Use "spawn" to avoid inheriting CUDA contexts / fork-unsafe state
        # from the training process.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as pool:
            futures = [
                pool.submit(
                    _gen_stats_for_one_dataset,
                    dataset_path=p,
                    embodiment_tag_value=embodiment_tag_value,
                    modality_meta_path=modality_meta_path,
                    stats_dir=stats_dir,
                    modality_config_path=modality_config_path,
                    tasks_dir=tasks_dir,
                )
                for p in dataset_paths
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating stats (parallel)",
            ):
                # Re-raise worker exceptions in the parent so failures don't
                # silently leave partial/missing stats behind.
                fut.result()

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert (
            self.config.training.eval_strategy == "no"
        ), "Sharded dataset does not support evaluation sets"

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            embodiment_tag = dataset_spec.embodiment_tag
            assert embodiment_tag is not None, "Embodiment tag is required"
            assert (
                self.config.data.mode == "single_turn"
            ), "Only single turn mode is supported"

            # Step 1: generate stats for every dataset (rank 0 only; other
            # ranks wait on the barrier below). Parallelised across
            # datasets when stats_num_workers > 1.
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    self._generate_all_stats(
                        list(dataset_spec.dataset_paths),
                        embodiment_tag,
                    )
            else:
                self._generate_all_stats(
                    list(dataset_spec.dataset_paths),
                    embodiment_tag,
                )
            barrier()

            # Step 2: instantiate the per-dataset loaders. This must happen
            # AFTER stats are on disk (LeRobotEpisodeLoader reads them in
            # __init__).
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    modality_meta_path=self.config.data.modality_meta_path,
                    stats_dir=self.config.data.stats_dir,
                    tasks_dir=getattr(self.config.data, "tasks_dir", None),
                    subtask_conditioned=getattr(
                        self.config.data, "subtask_conditioned", False
                    ),
                    bad_segments_path=getattr(
                        self.config.data, "bad_segments_path", None
                    ),
                    advantage_conditioned=getattr(
                        self.config.data, "advantage_conditioned", False
                    ),
                    motion_quality_labels_path=getattr(
                        self.config.data, "motion_quality_labels_path", None
                    ),
                    advantage_horizon=getattr(
                        self.config.data, "advantage_horizon", 16
                    ),
                    advantage_unpreferred_overlap_ratio=getattr(
                        self.config.data,
                        "advantage_unpreferred_overlap_ratio",
                        1.0 / 3.0,
                    ),
                    advantage_unknown_policy=getattr(
                        self.config.data,
                        "advantage_unknown_policy",
                        "unconditioned",
                    ),
                    rtc_max_delay=getattr(self.config.data, "rtc_max_delay", 0),
                    rtc_delay_decay=getattr(
                        self.config.data, "rtc_delay_decay", 0.0
                    ),
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
