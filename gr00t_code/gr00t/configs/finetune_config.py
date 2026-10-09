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

# Finetune config used for single node post-training.
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path(s) to the dataset root directory containing trajectory data for fine-tuning.

    Supports a single path, or multiple paths separated by commas (','). All paths
    are treated as the SAME embodiment and merged into one dataset group with
    mix_ratio=1.0. Example: "/data/ds_a,/data/ds_b,/data/ds_c".
    """

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    backbone_model_path: str = "/mnt/lichaojie/models/Cosmos-Reason2-2B"
    """Path to the VLM backbone (e.g. Cosmos-Reason2-2B) used by the N1.7 model.
    Overrides the ``model_name`` baked into the base model's config.json, which
    may point to a path that only exists on the machine where the base model
    was originally trained."""

    modality_meta_path: str | None = None
    """Optional absolute path to a shared ``modality.json`` file. When set, all
    datasets in ``--dataset_path`` will use this file as their modality meta
    instead of reading ``<dataset_path>/meta/modality.json`` for each one.
    Useful when training on multiple datasets with identical modality layout.
    All other meta files (info.json / episodes.jsonl / tasks.jsonl / stats.json)
    are still read per-dataset. If None, keeps the original behaviour."""

    stats_dir: str | None = None
    """Optional directory to store/load per-dataset ``stats.json`` and
    ``relative_stats.json``. When set, each dataset's stats live under
    ``<stats_dir>/<dataset_dirname>/`` instead of ``<dataset>/meta/``.
    Necessary when dataset directories are read-only. If None, keeps the
    original LeRobot behaviour (write stats into each dataset's meta/ dir)."""

    tasks_dir: str | None = None
    """Optional directory to load per-dataset ``tasks.jsonl`` from.
    When set, each dataset's task descriptions are read from
    ``<tasks_dir>/<dataset_dirname>/tasks.jsonl`` instead of
    ``<dataset>/meta/tasks.jsonl``. Useful when dataset directories are
    read-only, or when you want to override task descriptions without
    modifying the original dataset. If None, keeps the original
    per-dataset ``meta/tasks.jsonl`` behaviour. Missing file raises
    FileNotFoundError (no silent fallback). When subtask conditioning is
    enabled, an optional sibling ``subtasks.jsonl`` can override subtask
    wording without modifying ``episodes.jsonl``."""

    subtask_conditioned: bool = False
    """Append ``Subtask: <text>`` to the global task prompt using the current
    frame's ``[start, end)`` interval from ``meta/episodes.jsonl``. The global
    task still comes from ``tasks.jsonl`` (including ``tasks_dir`` overrides)."""

    bad_segments_path: str | None = None
    """Optional bad-frame source. Accepts a legacy inclusive-interval manifest,
    comma-separated version-2 all-frame quality-label JSON files, or a directory
    containing ``<dataset>_all_frames.json`` files. Training starts are removed
    whenever any configured temporal index touches an unpreferred frame."""

    advantage_conditioned: bool = False
    """Append a frame-label-derived motion-quality condition to the language
    instruction. This is independent from ``bad_segments_path``: enabling it
    does not remove any sample from the training index."""

    motion_quality_labels_path: str | None = None
    """Comma-separated frame-label JSON files, or a directory containing
    ``<dataset>_all_frames.json`` / ``<dataset>.json``. Each file must use the
    stumble-review version-2 per-episode ``labels`` format."""

    advantage_horizon: int = 16
    """Number of post-delay action frames used to choose preferred versus
    unpreferred. Any unpreferred frame makes the condition unpreferred."""

    advantage_unpreferred_overlap_ratio: float = 1.0 / 3.0
    """Additional full-postfix rule: if loss-bearing actions cover more than
    this fraction of any contiguous unpreferred segment, mark the sample
    unpreferred even when that segment starts after ``advantage_horizon``."""

    advantage_unknown_policy: str = "unconditioned"
    """How a window containing unknown labels is prompted: ``unconditioned``
    omits the quality suffix, while ``unknown`` appends
    ``Motion quality: unknown.``"""

    stats_num_workers: int = 0
    """Parallel workers used for the *one-time* stats generation across
    datasets. 0 / 1 means serial (original behaviour). Useful when training
    on many datasets for the first time — stats generation is CPU+IO bound
    and benefits from parallelism. Subsequent runs skip cached stats."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Training-Time RTC (real-time chunking, Black et al. arXiv:2512.05964) ---
    rtc_max_delay: int = 0
    """
    Maximum inference delay to simulate during training. 0 disables RTC (default).
    When > 0, training samples a per-batch delay d in [0, rtc_max_delay) and
    conditions the action expert on the first d ground-truth actions as a clean
    prefix, computing flow-matching loss only on the postfix.
    Pick this to match the policy's expected wall-clock inference latency in
    controller steps (e.g. 200ms latency on a 50Hz policy → 10).
    """

    rtc_delay_decay: float = 0.0
    """
    Distribution over delays. 0.0 = uniform (matches the pi torch reference).
    > 0 = exponentially decreasing weights ∝ exp(-rtc_delay_decay * d), as in
    the paper (Section V.A: "exponentially decreasing weights, as we found
    that higher delays need less training supervision").
    """


    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up.
    Only takes effect when warmup_steps == 0. If warmup_steps > 0, that value
    wins and this ratio is ignored (see transformers.TrainingArguments.
    get_warmup_steps). Setting both to 0 disables warmup — lr starts at peak."""

    warmup_steps: int = 0
    """Absolute number of warmup steps. When > 0 this OVERRIDES warmup_ratio
    (opposite of what the training_config.py comment historically said).
    0 = fall through to warmup_ratio."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""
