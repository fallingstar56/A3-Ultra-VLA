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

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import FINETUNE_ONLY_TAGS, POSTTRAIN_TAGS, EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData
from gr00t.eval.real_robot.hand_mapping import load_hand_opening_mapping

from .policy import BasePolicy, PolicyWrapper


# Default topic path for the raw VLA token chunk stream. Placeholder — the
# real deployment topic (e.g. /ta/whole_body_command or a per-embodiment
# equivalent) should be set via Gr00tPolicy.set_token_chunk_topic() or via
# the CLI in eval scripts. Kept as a module-level constant so both the
# GR00T inference server and the robot-side receiver can import the same
# default and stay in sync until it's swapped for the real name.
DEFAULT_VLA_TOKEN_CHUNK_TOPIC = "/vla/token_chunk"


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type.
                Accepts an EmbodimentTag enum or a string (resolved case-insensitively).
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        model_dir = Path(model_path)
        self.model_dir = model_dir

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation.
        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root.  Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            # Map raw checkpoint tag values to user-friendly enum names where possible.
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(
                        f"  {enum_name:30s} (--embodiment-tag {enum_name})"
                    )
                else:
                    supported_lines.append(
                        f"  {tag_value:30s} (internal, no public enum)"
                    )
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model. "
                    f"See the example READMEs for how to finetune and download checkpoints."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots. "
                    f"Use it with launch_finetune.py, not with the base model directly."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert (
            len(language_delta_indices) == 1
        ), "Only one language delta index is supported"
        self.language_key = language_keys[0]

        # Token-chunk mode config. `output_mode="token_chunk"` makes get_action
        # return the raw normalized chunk (plus pooled VLA + state features
        # and residual metadata) so a downstream chunk-based transport can
        # forward the whole thing to the robot for 20/30Hz→60Hz interpolation
        # and 60Hz residual compensation. See send_token_chunk_for_robot().
        self._token_chunk_topic = DEFAULT_VLA_TOKEN_CHUNK_TOPIC
        self._token_chunk_seq_id = 0

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert (
                modality in observation
            ), f"Observation must contain a '{modality}' key"
            assert isinstance(
                observation[modality], dict
            ), f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            assert (
                video_key in observation["video"]
            ), f"Video key '{video_key}' must be in observation"

            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert (
                    len(observation["video"][video_key]) == bs
                ), f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_video, np.ndarray
            ), f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert (
                batched_video.dtype == np.uint8
            ), f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert (
                batched_video.ndim == 5
            ), f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(
                self.modality_configs["video"].delta_indices
            ), f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"

            # Verify channel dimension is 3 (RGB images)
            assert (
                batched_video.shape[-1] == 3
            ), f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Check that the expected state key exists in the observation
            # (must happen before indexing — see video validation above)
            assert (
                state_key in observation["state"]
            ), f"State key '{state_key}' must be in observation"

            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert (
                    len(observation["state"][state_key]) == bs
                ), f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_state, np.ndarray
            ), f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"

            # Verify dtype is float32 (standard for continuous state values)
            assert (
                batched_state.dtype == np.float32
            ), f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                batched_state.ndim == 3
            ), f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(
                self.modality_configs["state"].delta_indices
            ), f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"

        # Reference-only state keys (e.g. an IMU orientation used as the
        # RELATIVE-action reference frame but NOT fed into the encoder) must
        # still be present in observation["state"] — decode_action looks them
        # up at unapply time. Fail loudly here rather than in a downstream
        # np.stack KeyError.
        _ref_only = getattr(
            self.modality_configs["state"], "reference_only_keys", None
        ) or []
        for state_key in _ref_only:
            assert state_key in observation["state"], (
                f"Reference-only state key '{state_key}' must be in observation "
                f"— it is used as the RELATIVE-action reference frame at decode "
                f"time. Provide it alongside the other state keys."
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Check that the expected language key exists in the observation
            # (must happen before indexing — see video validation above)
            assert (
                language_key in observation["language"]
            ), f"Language key '{language_key}' must be in observation"

            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert (
                    len(observation["language"][language_key]) == bs
                ), f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(
                batched_language, list
            ), f"Language key '{language_key}' must be a list. Got {type(batched_language)}"

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(
                    self.modality_configs["language"].delta_indices
                ), f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(
                    batch_item, list
                ), f"Language batch item must be a list. Got {type(batch_item)}"

                # Current implementation expects exactly one language instruction per timestep
                assert (
                    len(batch_item) == 1
                ), f"Language batch item must have exactly one item. Got {len(batch_item)}"

                # Verify the instruction itself is a string
                assert isinstance(
                    batch_item[0], str
                ), f"Language batch item must be a string. Got {type(batch_item[0])}"

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(
                vla_step_data.states
            )  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [
                {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
            ]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Train-time RTC (arXiv:2512.05964): if the caller passes a normalized
        # model-space action prefix in ``options``, inject it into the batch so
        # Gr00tN1d7ActionHead.get_action() can pin it at flow-matching t=1 and
        # denoise only the postfix. The prefix comes back from a previous call's
        # info["action_pred_normalized"] and is (B, action_horizon, action_dim).
        rtc_options = options or {}
        rtc_prefix = rtc_options.get("action_prefix")
        if rtc_prefix is not None:
            prefix_t = rtc_prefix
            if isinstance(prefix_t, np.ndarray):
                prefix_t = torch.from_numpy(np.asarray(prefix_t, dtype=np.float32))
            prefix_t = prefix_t.to(
                device=self.model.device, dtype=torch.bfloat16
            )
            # Inject under the standard key that action_head.prepare_input reads.
            # Nested under "inputs" to match the collator's output layout
            # (Gr00tN1d7DataCollator returns {"inputs": {...}}).
            if "inputs" in collated_inputs and isinstance(collated_inputs["inputs"], dict):
                collated_inputs["inputs"]["action"] = prefix_t
            else:
                collated_inputs["action"] = prefix_t

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs, options=options)
        normalized_action = model_pred["action_pred"].float()
        # Uncompensated + delta are None on older ckpts / when residual head
        # is off. `.get` on BatchFeature falls back to dict semantics.
        normalized_action_uncomp = model_pred.get("action_pred_uncompensated", None)
        action_residual = model_pred.get("action_residual", None)
        if normalized_action_uncomp is not None:
            normalized_action_uncomp = normalized_action_uncomp.float()
        if action_residual is not None:
            action_residual = action_residual.float()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        _state_cfg = self.modality_configs["state"]
        _state_lookup_keys = list(_state_cfg.modality_keys)
        # Include reference_only_keys so RELATIVE->absolute decode can look up
        # its reference frame in state (see StateActionProcessor.unapply_action).
        _ref_only = getattr(_state_cfg, "reference_only_keys", None) or []
        for _k in _ref_only:
            if _k not in _state_lookup_keys:
                _state_lookup_keys.append(_k)
        for k in _state_lookup_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        # Train-time RTC clients need the normalized (model-space) chunk to feed
        # back as `action_prefix` on the next call. Detach + numpy so it survives
        # msgpack serialization over the ZMQ PolicyClient/Server bridge.
        info = {
            "action_pred_normalized": normalized_action.detach().cpu().numpy(),
        }
        if normalized_action_uncomp is not None:
            info["action_pred_uncompensated_normalized"] = (
                normalized_action_uncomp.detach().cpu().numpy()
            )
        if action_residual is not None:
            info["action_residual_normalized"] = (
                action_residual.detach().cpu().numpy()
            )

        # ------------------------------------------------------------------
        # Token-chunk output mode.
        #
        # When the caller asks for output_mode="token_chunk" we bundle the
        # raw normalized chunk with pooled conditioning + a chunk envelope
        # (chunk_id, source_hz, target_interp_hz, topic path, timestamps)
        # for downstream 20/30→60Hz interpolation + 60Hz residual injection
        # on the robot. Nothing is decoded — the robot side decodes only
        # after residual is applied so the delta operates in normalized
        # (model-space) units, which is the space the residual head was
        # trained in.
        # ------------------------------------------------------------------
        output_mode = (options or {}).get("output_mode", "action")
        if output_mode == "token_chunk":
            import time as _time
            info["token_chunk"] = self._build_token_chunk_envelope(
                normalized_action=normalized_action,
                normalized_action_uncomp=normalized_action_uncomp,
                action_residual=action_residual,
                backbone_features=model_pred.get("backbone_features", None),
                state_features=model_pred.get("state_features", None),
                source_hz=float((options or {}).get("source_hz", 20.0)),
                target_interp_hz=float((options or {}).get("target_interp_hz", 60.0)),
                topic=(options or {}).get("token_chunk_topic", self._token_chunk_topic),
                wall_time=_time.time(),
            )
        return casted_action, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(
                action_arr, np.ndarray
            ), f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"

            # Verify dtype is float32 (standard for continuous actions)
            assert (
                action_arr.dtype == np.float32
            ), f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                action_arr.ndim == 3
            ), f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(
                self.modality_configs["action"].delta_indices
            ), f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def get_rtc_metadata(self) -> dict[str, Any]:
        """Return everything a train-time-RTC client needs to build prefixes.

        The client (see gr00t/eval/real_robot/A2/infer_a2_rtc.py) has no direct
        access to the checkpoint's action stats or the modality's ABSOLUTE /
        RELATIVE tags. To do the cross-chunk delta-frame reanchor (paper
        arXiv:2512.05964v2, Sec III on absolute/relative actions) it needs the
        per-flat-dim normalization scale AND a mask of which dims are relative.

        Returns a plain-python dict (msgpack-friendly). Values are numpy where
        the client will want vectorised math; lists where the client will
        iterate structurally. Fields:
            action_horizon:    len(delta_indices) — H
            action_keys:       ordered modality keys, e.g. ["hand_joint", "arm_joint"]
            action_dims:       {key: D_key}
            action_reps:       {key: "ABSOLUTE" | "RELATIVE"}
            action_state_key:  {key: state_key | None} — RELATIVE dims reanchor
                               against this state key, which may differ from the
                               action key name (e.g. sonic-a3 ``body`` →
                               ``body_pos``). None for ABSOLUTE keys / when the
                               config has no action_configs. Clients fall back
                               to the action key name when absent/None.
            action_norm:       {key: {"kind": "meanstd"|"minmax", ...params}}
            state_keys:        ordered state modality keys
            state_dims:        {key: D_key}
            reference_only_keys: state keys the loader materialises but the
                               encoder does NOT concat — clients must still
                               feed them so decode_action can look up R_ref
                               for RELATIVE-rot6d unapply (e.g. sonic-a3
                               ``pelvis_orient6d``).
            video_keys:        ordered video modality keys — clients map these
                               abstract names to their own camera streams.
            language_keys:     ordered language modality keys — usually a
                               single ``annotation.human.task_description``.
            rtc_max_delay:     int (0 == checkpoint not trained with train-time RTC)
            use_relative_action: bool (StateActionProcessor flag)
            hand_mapping:      {"initial_hand_rad": 20D list} loaded from
                               experiment_cfg/launch/hand_mapping_sample.parquet,
                               or None for legacy checkpoints.
            hand_mapping_sample: compatibility alias for hand_mapping.
        """
        emb_tag = self.embodiment_tag.value
        action_cfg = self.modality_configs["action"]
        state_cfg = self.modality_configs["state"]
        action_horizon = len(action_cfg.delta_indices)

        # Per-key rep tag. Older configs may not have action_configs — treat
        # everything as absolute in that case (no cross-chunk reanchor needed).
        action_reps: dict[str, str] = {}
        action_state_key: dict[str, str | None] = {}
        action_format: dict[str, str] = {}
        action_type: dict[str, str] = {}
        if getattr(action_cfg, "action_configs", None):
            for key, cfg in zip(action_cfg.modality_keys, action_cfg.action_configs):
                action_reps[key] = str(cfg.rep).split(".")[-1]  # enum → "ABSOLUTE"
                # RELATIVE dims reanchor against cfg.state_key (defaults to the
                # action key name when None). Surface it so clients can map an
                # action key whose state reference has a different name — e.g.
                # sonic-a3 trains action ``body`` as RELATIVE w/ state_key=
                # ``body_pos``. ABSOLUTE keys get None.
                action_state_key[key] = (
                    cfg.state_key if cfg.state_key else None
                )
                # Format + type so RTC reanchor can dispatch joint-vector math
                # (DEFAULT → linear shift) vs SO(3) composition (ROT6D →
                # R_rel_new = R_new^-1 @ R_prev @ R_rel_prev). Without this a
                # RELATIVE rot6d key would get the vector-shift path and its
                # absolute pose would jump on every chunk swap.
                action_format[key] = str(cfg.format).split(".")[-1]  # "ROT6D"|"DEFAULT"|...
                action_type[key] = str(cfg.type).split(".")[-1]      # "EEF"|"NON_EEF"
        else:
            for key in action_cfg.modality_keys:
                action_reps[key] = "ABSOLUTE"
                action_state_key[key] = None
                action_format[key] = "DEFAULT"
                action_type[key] = "NON_EEF"

        # Grab the exact norm params the processor uses on the return path
        # (unapply_action). This picks up the meanstd-vs-minmax choice and any
        # relative_action override already baked in.
        sap = self.processor.state_action_processor
        action_norm: dict[str, dict[str, Any]] = {}
        action_dims: dict[str, int] = {}
        meanstd_keys = action_cfg.mean_std_embedding_keys or set()
        for key in action_cfg.modality_keys:
            params = sap.norm_params[emb_tag]["action"][key]
            action_dims[key] = int(params["dim"])
            if key in meanstd_keys:
                action_norm[key] = {
                    "kind": "meanstd",
                    "mean": np.asarray(params["mean"], dtype=np.float32),
                    "std": np.asarray(params["std"], dtype=np.float32),
                }
            else:
                action_norm[key] = {
                    "kind": "minmax",
                    "min": np.asarray(params["min"], dtype=np.float32),
                    "max": np.asarray(params["max"], dtype=np.float32),
                }

        state_dims: dict[str, int] = {}
        for key in state_cfg.modality_keys:
            params = sap.norm_params[emb_tag]["state"][key]
            state_dims[key] = int(params["dim"])

        rtc_max_delay = int(getattr(self.model.config, "rtc_max_delay", 0) or 0)

        video_cfg = self.modality_configs["video"]
        language_cfg = self.modality_configs["language"]
        reference_only_keys = list(
            getattr(state_cfg, "reference_only_keys", None) or []
        )

        hand_mapping = None
        hand_mapping_sample = (
            self.model_dir
            / "experiment_cfg"
            / "launch"
            / "hand_mapping_sample.parquet"
        )
        if hand_mapping_sample.exists():
            try:
                hand_mapping = load_hand_opening_mapping(
                    hand_mapping_sample
                ).to_metadata()
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load checkpoint hand mapping sample "
                    f"{hand_mapping_sample}: {exc}"
                ) from exc

        return {
            "action_horizon": action_horizon,
            "action_keys": list(action_cfg.modality_keys),
            "action_dims": action_dims,
            "action_reps": action_reps,
            "action_state_key": action_state_key,
            "action_format": action_format,
            "action_type": action_type,
            "action_norm": action_norm,
            "state_keys": list(state_cfg.modality_keys),
            "state_dims": state_dims,
            "reference_only_keys": reference_only_keys,
            "video_keys": list(video_cfg.modality_keys),
            "language_keys": list(language_cfg.modality_keys),
            "rtc_max_delay": rtc_max_delay,
            "use_relative_action": bool(getattr(sap, "use_relative_action", False)),
            "hand_mapping": hand_mapping,
            "hand_mapping_sample": hand_mapping,
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        self._token_chunk_seq_id = 0
        return {}

    # ----------------------------------------------------------------------
    # Token-chunk mode
    # ----------------------------------------------------------------------

    def set_token_chunk_topic(self, topic: str) -> None:
        """Override the default ``/vla/token_chunk`` placeholder topic.

        Meant to be called once during setup so every ``get_action(options=
        {"output_mode": "token_chunk"})`` bakes the real deployment topic
        into its envelope. The robot-side receiver reads the same field to
        subscribe to the correct ROS topic.
        """
        self._token_chunk_topic = str(topic)

    def get_token_chunk_topic(self) -> str:
        return self._token_chunk_topic

    def _build_token_chunk_envelope(
        self,
        *,
        normalized_action: torch.Tensor,
        normalized_action_uncomp: torch.Tensor | None,
        action_residual: torch.Tensor | None,
        backbone_features: torch.Tensor | None,
        state_features: torch.Tensor | None,
        source_hz: float,
        target_interp_hz: float,
        topic: str,
        wall_time: float,
    ) -> dict[str, Any]:
        """Serialize the raw chunk plus what the robot-side runner needs.

        Only pool + fp32 the features that the caller actually asked to
        include (``include_vla_feature`` / ``include_state_feature`` in
        options); the pooled fp32 vectors are still small (< a few kB)
        but we keep the flag so a chunk over an ROS wire can stay lean.

        The returned dict is msgpack-friendly:
            {
              "topic":            str, placeholder path unless overridden
              "chunk_id":         int, monotonic per Gr00tPolicy instance
              "wall_time":        float, seconds since epoch at inference
              "source_hz":        float, chunk-index advance rate on robot
              "target_interp_hz": float, robot-side interpolation output rate
              "action_horizon":   int
              "action_dim":       int
              "tokens":           np.float32 (B, H, D) normalized action
              "tokens_uncomp":    np.float32 (B, H, D) or None
              "residual":         np.float32 (B, H, D) or None
              "vla_feature":      np.float32 (B, D_vla) pooled or None
              "state_feature":    np.float32 (B, D_state) pooled or None
            }
        """
        seq_id = self._token_chunk_seq_id
        self._token_chunk_seq_id += 1

        tokens_np = normalized_action.detach().cpu().numpy().astype(np.float32)
        tokens_uncomp_np = (
            normalized_action_uncomp.detach().cpu().numpy().astype(np.float32)
            if normalized_action_uncomp is not None
            else None
        )
        residual_np = (
            action_residual.detach().cpu().numpy().astype(np.float32)
            if action_residual is not None
            else None
        )

        def _pool_fp32(x: torch.Tensor | None) -> np.ndarray | None:
            if x is None:
                return None
            if x.dim() == 3:
                x = x.mean(dim=1)
            return x.detach().float().cpu().numpy().astype(np.float32)

        vla_feature_np = _pool_fp32(backbone_features)
        state_feature_np = _pool_fp32(state_features)

        B, H, D = tokens_np.shape
        return {
            "topic": topic,
            "chunk_id": int(seq_id),
            "wall_time": float(wall_time),
            "source_hz": float(source_hz),
            "target_interp_hz": float(target_interp_hz),
            "action_horizon": int(H),
            "action_dim": int(D),
            "tokens": tokens_np,
            "tokens_uncomp": tokens_uncomp_np,
            "residual": residual_np,
            "vla_feature": vla_feature_np,
            "state_feature": state_feature_np,
        }


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert (
            len(self.policy.modality_configs["language"].delta_indices) == 1
        ), "Only one language delta index is supported"

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert (
                parsed_key in observation
            ), f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_video, np.ndarray
            ), f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert (
                batched_video.dtype == np.uint8
            ), f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert (
                batched_video.ndim == 5
            ), f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(
                modality_configs["video"].delta_indices
            ), f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"

            # Verify channel dimension is 3 (RGB images)
            assert (
                batched_video.shape[-1] == 3
            ), f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert (
                parsed_key in observation
            ), f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(
                batched_state, np.ndarray
            ), f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"

            # Verify dtype is float32 (standard for continuous state values)
            assert (
                batched_state.dtype == np.float32
            ), f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                batched_state.ndim == 3
            ), f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(
                modality_configs["state"].delta_indices
            ), f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"

        # Reference-only state keys must also be present (see check_observation
        # for the same guard on the nested-observation path).
        _ref_only = getattr(
            modality_configs["state"], "reference_only_keys", None
        ) or []
        for state_key in _ref_only:
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, (
                f"Reference-only state key '{parsed_key}' must be in observation "
                f"— it is used as the RELATIVE-action reference frame at decode "
                f"time. Provide it alongside the other state keys."
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if (
                language_key == "task"
                and "annotation.human.coarse_action" in observation
            ):
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert (
                language_key in observation
            ), f"Language key '{language_key}' must be in observation"

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(
                batched_language, (tuple, list)
            ), f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"

            # Verify each batch item is a string
            assert isinstance(
                batched_language[0], str
            ), f"Language batch item must be a string. Got {type(batched_language[0])}"

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            keys_to_read = list(self.policy.modality_configs[modality].modality_keys)
            # State-only: also read reference_only_keys — they need to reach
            # decode_action for RELATIVE-action absolute reconstruction.
            if modality == "state":
                _ref_only = getattr(
                    self.policy.modality_configs[modality],
                    "reference_only_keys",
                    None,
                ) or []
                for _k in _ref_only:
                    if _k not in keys_to_read:
                        keys_to_read.append(_k)
            for key in keys_to_read:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if (
                        key == "task"
                        and "annotation.human.coarse_action" in observation
                    ):
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(
                action_arr, np.ndarray
            ), f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"

            # Verify dtype is float32 (standard for continuous actions)
            assert (
                action_arr.dtype == np.float32
            ), f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"

            # Verify shape has 3 dimensions: (B, T, D)
            assert (
                action_arr.ndim == 3
            ), f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(
                modality_configs["action"].delta_indices
            ), f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()

    def get_rtc_metadata(self) -> dict[str, Any]:
        """Passthrough for train-time RTC clients (see Gr00tPolicy.get_rtc_metadata)."""
        return self.policy.get_rtc_metadata()
