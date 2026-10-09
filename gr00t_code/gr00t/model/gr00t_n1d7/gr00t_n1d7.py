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
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.residual_head import ResidualHead

logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # Optional residual head. See gr00t/model/modules/residual_head.py for
        # the design. Off by default; when on the checkpoint gains a small
        # (~1M param) MLP whose zero-init output leaves inference unchanged
        # until the head is trained.
        self.residual_head: ResidualHead | None = None
        if getattr(config, "residual_head_enabled", False):
            out_horizon = (
                config.residual_output_horizon
                if getattr(config, "residual_output_horizon", None) is not None
                else config.action_horizon
            )
            state_dim = (
                self.input_embedding_dim
                if getattr(config, "residual_head_use_state", True)
                else None
            )
            vla_dim = (
                config.backbone_embedding_dim
                if getattr(config, "residual_head_use_vla_feature", True)
                else None
            )
            self.residual_head = ResidualHead(
                action_dim=self.action_dim,
                output_horizon=out_horizon,
                hidden_dim=config.residual_head_hidden_dim,
                cond_dim=config.residual_head_cond_dim,
                state_feature_dim=state_dim,
                vla_feature_dim=vla_dim,
                history_len=config.residual_head_history_len,
                delta_bound=config.residual_head_delta_bound,
                step_embed_dim=config.residual_head_step_embed_dim,
                num_hidden_layers=config.residual_head_num_layers,
            )
            logger.info(
                "Residual head enabled: %d trainable params (bound=%.3f, out_horizon=%d)",
                self.residual_head.num_trainable_params(),
                config.residual_head_delta_bound,
                out_horizon,
            )

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        # residual_head is orthogonal to the three flags above; keep it
        # trainable by default (was already set requires_grad=True above)
        # unless the caller asked to freeze the whole VLA (residual-only
        # training). In that case the training launcher will call
        # ``freeze_vla_for_residual_only`` after model construction.
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def freeze_vla_for_residual_only(self):
        """Freeze every parameter EXCEPT the residual head. Called by the
        residual-head training path when config.residual_freeze_vla=True.

        Callable on Gr00tN1d7 top-level too; see the wrapper below.
        """
        for p in self.parameters():
            p.requires_grad = False
        if self.residual_head is not None:
            for p in self.residual_head.parameters():
                p.requires_grad = True
            trainable = self.residual_head.num_trainable_params()
            logger.info(
                "Frozen VLA action head, only residual head trainable (%d params).",
                trainable,
            )
        else:
            logger.warning(
                "freeze_vla_for_residual_only called but residual_head is None — "
                "nothing is trainable."
            )

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _sample_rtc_delay(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample inference-delay ``d`` per batch element for training-time RTC.

        ``d`` is drawn from ``{0, ..., rtc_max_delay - 1}``. If
        ``rtc_delay_decay > 0`` the probability of delay ``d`` is proportional
        to ``exp(-rtc_delay_decay * d)`` (exponentially decreasing weights, as
        in the paper); otherwise the distribution is uniform.
        """
        max_delay = self.config.rtc_max_delay
        decay = float(getattr(self.config, "rtc_delay_decay", 0.0) or 0.0)
        if decay <= 0.0:
            return torch.randint(
                low=0, high=max_delay, size=(batch_size,), device=device, dtype=torch.long
            )
        d = torch.arange(max_delay, device=device, dtype=torch.float32)
        probs = torch.exp(-decay * d)
        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=batch_size, replacement=True)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype
        )

        bsize, ah, _ = actions.shape
        rtc_max_delay = int(getattr(self.config, "rtc_max_delay", 0) or 0)
        rtc_active = self.training and rtc_max_delay > 0

        if rtc_active:
            # Training-time RTC (Black et al. arXiv:2512.05964).
            #
            # Sample per-batch delay d ~ {0..rtc_max_delay-1}; the first d
            # actions become a "clean prefix" (flow-matching time = 1, i.e.
            # x_t = ground-truth actions). The remaining "postfix" tokens get
            # the standard sampled time. The model is told the per-token time
            # via adaLN-zero in the DiT and the sinusoidal timestep embedding
            # in the action encoder, and the loss is computed only on postfix
            # tokens.
            provided_delay = action_input.get("rtc_delay", None)
            if provided_delay is None:
                delay = self._sample_rtc_delay(bsize, actions.device)
            else:
                delay = provided_delay.to(device=actions.device, dtype=torch.long).reshape(-1)
                if delay.numel() != bsize:
                    raise ValueError(
                        f"rtc_delay batch has {delay.numel()} values, expected {bsize}"
                    )
                if torch.any(delay < 0) or torch.any(delay >= rtc_max_delay):
                    raise ValueError(
                        "rtc_delay values must be in "
                        f"[0, rtc_max_delay={rtc_max_delay})"
                    )
            prefix_mask = (
                torch.arange(ah, device=actions.device)[None, :] < delay[:, None]
            )  # (B, ah)
            # t_per_token: prefix=1 (clean / matches the ground-truth action),
            # postfix=sampled scalar t.
            t_per_token = torch.where(
                prefix_mask,
                torch.ones((), device=actions.device, dtype=t.dtype),
                t[:, None].expand(bsize, ah),
            )  # (B, ah)

            t_expanded = t_per_token[:, :, None]  # (B, ah, 1)
            noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions
            velocity = actions - noise

            # Discretize per-token time for action encoder and DiT.
            t_discretized = (t_per_token * self.num_timestep_buckets).long()  # (B, ah)
            # Build per-token timestep for the full DiT sequence (state token
            # then action tokens). The state token reuses the sampled scalar t
            # so that with delay==0 the behavior matches the non-RTC path.
            state_t_discretized = (t * self.num_timestep_buckets).long()  # (B,)
            dit_timestep = torch.cat(
                [state_t_discretized[:, None], t_discretized], dim=1
            )  # (B, 1+ah)
        else:
            t = t[:, None, None]  # (B,1,1) for broadcast
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()  # (B,)
            dit_timestep = t_discretized  # (B,)
            prefix_mask = None

        action_features = self.action_encoder(
            noisy_trajectory, t_discretized, embodiment_id
        )

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=dit_timestep,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=dit_timestep,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        if rtc_active:
            # Per paper Algorithm 1: mask loss so only postfix tokens contribute.
            postfix_mask = (~prefix_mask)[:, :, None].to(action_loss.dtype)  # (B, ah, 1)
            action_loss = action_loss * postfix_mask
            denom = (action_mask * postfix_mask).sum() + 1e-6
        else:
            denom = action_mask.sum() + 1e-6
        loss = action_loss.sum() / denom

        # ------------------------------------------------------------------
        # Residual head loss (joint or residual-only training).
        # ------------------------------------------------------------------
        residual_loss = None
        if self.residual_head is not None:
            # Base action chunk to correct = one Euler integration step from
            # noise using pred_velocity, matching what inference produces
            # at t=0. Cheaper than running the whole denoising loop and
            # already teaches the head to compensate the base-VLA output.
            with torch.no_grad():
                base_pred = noise + pred_velocity.detach()
            delta = self._compute_residual_delta(
                base_action_chunk=base_pred,
                state_features=state_features,
                vl_embeds=vl_embeds,
                history_action=action_input.get("history_action", None),
            )
            corrected = base_pred + delta.to(dtype=base_pred.dtype)
            residual_target = actions  # ground truth
            residual_mse = (
                F.mse_loss(corrected, residual_target, reduction="none") * action_mask
            )
            if rtc_active:
                residual_mse = residual_mse * postfix_mask
                r_denom = (action_mask * postfix_mask).sum() + 1e-6
            else:
                r_denom = action_mask.sum() + 1e-6
            residual_loss = residual_mse.sum() / r_denom
            # Simple sum — the residual head is small, if the caller wants
            # a weight they can multiply outside. Freezing the VLA is the
            # standard knob for residual-only training.
            loss = loss + residual_loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "residual_loss": residual_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert (
            current_T == self.config.state_history_length
        ), "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(
            data={"backbone_features": vl_embeds, "state_features": state_features}
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        # Train-time RTC inference (Black et al. arXiv:2512.05964): the model
        # was trained to consume a clean action prefix at per-token flow time
        # = 1, so at inference we pin those positions to ground-truth across
        # every Euler step and pass per-token times to the network. This is a
        # drop-in replacement for the inference-time RTC branch below.
        rtc_mode = (options or {}).get("rtc_mode", "inference_time")
        if rtc_mode == "train_time" and "action" in action_input:
            return self._sample_actions_with_prefix(
                vl_embeds=vl_embeds,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                action_prefix=action_input["action"],
                options=options,
                noise=actions,
            )

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = (
                options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            )
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(
                self.num_inference_timesteps
            )  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(
                actions, timesteps_tensor, embodiment_id
            )
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        # Optional post-VLA residual correction. Config flag decides whether
        # the returned ``action_pred`` already folds in the delta or whether
        # the caller wants both variants exposed (e.g. for eval curves).
        residual_options = options or {}
        use_residual = residual_options.get("use_residual", None)
        if use_residual is None:
            use_residual = bool(
                self.residual_head is not None
                and getattr(self.config, "residual_apply_in_get_action", True)
            )
        delta = None
        actions_uncomp = actions
        if use_residual and self.residual_head is not None:
            delta = self._compute_residual_delta(
                base_action_chunk=actions,
                state_features=state_features,
                vl_embeds=vl_embeds,
                history_action=action_input.get("history_action", None),
            )
            actions = actions + delta.to(dtype=actions.dtype)

        return BatchFeature(
            data={
                "action_pred": actions,
                "action_pred_uncompensated": actions_uncomp,
                "action_residual": delta,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def _sample_actions_with_prefix(
        self,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_prefix: torch.Tensor,
        options: dict[str, Any] | None,
        noise: torch.Tensor,
    ) -> BatchFeature:
        """Train-time RTC inference: pin a clean action prefix and denoise the postfix.

        Expects ``options`` to contain ``rtc_delay`` (alias: ``rtc_overlap_steps``)
        — the number of action positions, ``d``, taken from the previous chunk's
        tail. With ``rtc_overlap_steps`` the prefix is read from the last
        ``d`` actions of ``action_prefix`` (matching the existing inference-time
        RTC convention in this file); with ``rtc_delay`` (or no alias) it is read
        from the first ``d`` positions of ``action_prefix`` directly.
        """
        assert options is not None, "options is required for train-time RTC"
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        ah = self.config.action_horizon

        if "rtc_delay" in options:
            delay = int(options["rtc_delay"])
            prefix_actions = action_prefix[:, :delay, :].to(noise.dtype)
        elif "rtc_overlap_steps" in options:
            # Reuse the convention from the inference-time RTC branch: the
            # caller passes the previous chunk and we splice its tail.
            delay = int(options["rtc_overlap_steps"])
            action_horizon_before_padding = int(
                options.get("action_horizon", action_prefix.shape[1])
            )
            prefix_actions = action_prefix[
                :,
                action_horizon_before_padding - delay : action_horizon_before_padding,
                :,
            ].to(noise.dtype)
        else:
            raise AssertionError(
                "train-time RTC requires `rtc_delay` or `rtc_overlap_steps` in options"
            )

        assert 0 <= delay <= ah, f"delay {delay} must be in [0, action_horizon={ah}]"

        prefix_mask = torch.arange(ah, device=device)[None, :] < delay  # (1, ah)
        prefix_mask = prefix_mask.expand(batch_size, ah)

        # Pad the (possibly shorter) prefix actions out to the full horizon so
        # we can `where`-blend without index gymnastics.
        action_prefix_full = torch.zeros_like(noise)
        if delay > 0:
            action_prefix_full[:, :delay, :] = prefix_actions

        x_t = torch.where(prefix_mask[:, :, None], action_prefix_full, noise)
        dt = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)  # 0, 1/N, 2/N, ...
            t_cont_postfix = torch.full(
                size=(batch_size, ah),
                fill_value=t_cont,
                device=device,
                dtype=torch.float32,
            )
            t_per_token = torch.where(
                prefix_mask,
                torch.ones((), device=device, dtype=t_cont_postfix.dtype),
                t_cont_postfix,
            )  # (B, ah)
            t_disc_per_token = (t_per_token * self.num_timestep_buckets).long()

            # State token reuses the postfix time so behavior matches the
            # non-RTC inference path when delay==0.
            state_t = torch.full(
                size=(batch_size, 1),
                fill_value=int(t_cont * self.num_timestep_buckets),
                device=device,
                dtype=torch.long,
            )
            dit_timestep = torch.cat([state_t, t_disc_per_token], dim=1)  # (B, 1+ah)

            action_features = self.action_encoder(
                x_t, t_disc_per_token, embodiment_id
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)

            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=dit_timestep,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=dit_timestep,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler step only on postfix; prefix stays pinned to ground-truth.
            x_t = torch.where(
                prefix_mask[:, :, None],
                action_prefix_full,
                x_t + dt * pred_velocity,
            )

        # Same residual gate as the non-RTC branch; RTC prefix positions are
        # already pinned to ground truth so applying delta there is
        # meaningless — we mask it out.
        residual_options = options or {}
        use_residual = residual_options.get("use_residual", None)
        if use_residual is None:
            use_residual = bool(
                self.residual_head is not None
                and getattr(self.config, "residual_apply_in_get_action", True)
            )
        delta = None
        x_t_uncomp = x_t
        if use_residual and self.residual_head is not None:
            delta = self._compute_residual_delta(
                base_action_chunk=x_t,
                state_features=state_features,
                vl_embeds=vl_embeds,
                history_action=None,
            )
            delta = delta.to(dtype=x_t.dtype)
            # Zero delta on prefix positions (they're pinned).
            delta = torch.where(prefix_mask[:, :, None], torch.zeros_like(delta), delta)
            x_t = x_t + delta

        return BatchFeature(
            data={
                "action_pred": x_t,
                "action_pred_uncompensated": x_t_uncomp,
                "action_residual": delta,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    # ------------------------------------------------------------------
    # Residual head helpers
    # ------------------------------------------------------------------

    def _pool_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Mean-pool along the sequence dimension so ResidualHead gets a
        fixed-shape conditioning vector regardless of backbone seq_len."""
        if feat is None:
            return None
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    def _compute_residual_delta(
        self,
        base_action_chunk: torch.Tensor,
        state_features: torch.Tensor | None,
        vl_embeds: torch.Tensor | None,
        history_action: torch.Tensor | None,
    ) -> torch.Tensor:
        """Public-ish wrapper. Handles pooling of encoder outputs and
        respects the config gates for which streams the head sees."""
        assert self.residual_head is not None, "residual_head is disabled"
        state_feat = None
        vla_feat = None
        if getattr(self.config, "residual_head_use_state", True):
            state_feat = self._pool_features(state_features)
        if getattr(self.config, "residual_head_use_vla_feature", True):
            vla_feat = self._pool_features(vl_embeds)
        # Cast conditioning streams to the head's parameter dtype (usually
        # fp32) so bf16 backbones don't stall the head in low precision.
        target_dtype = next(self.residual_head.parameters()).dtype
        if state_feat is not None:
            state_feat = state_feat.to(dtype=target_dtype)
        if vla_feat is not None:
            vla_feat = vla_feat.to(dtype=target_dtype)
        if history_action is not None:
            history_action = history_action.to(dtype=target_dtype)
        delta = self.residual_head(
            base_action_chunk=base_action_chunk.to(dtype=target_dtype),
            state_feature=state_feat,
            vla_feature=vla_feat,
            history_action=history_action,
        )
        return delta

    @torch.no_grad()
    def compute_residual_delta_from_features(
        self,
        base_action_chunk: torch.Tensor,
        *,
        state_features: torch.Tensor | None = None,
        vla_features: torch.Tensor | None = None,
        history_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference-only shortcut: caller has already cached VLA state and
        vla features, wants a fresh delta for a possibly re-interpolated
        chunk. Used by the 60Hz robot-side runner to re-invoke the head at
        every 60Hz tick against the interpolated waypoint while keeping the
        conditioning frozen from the last VLA inference."""
        assert self.residual_head is not None, "residual_head is disabled"
        target_dtype = next(self.residual_head.parameters()).dtype
        if state_features is not None:
            state_features = self._pool_features(state_features).to(dtype=target_dtype)
        if vla_features is not None:
            vla_features = self._pool_features(vla_features).to(dtype=target_dtype)
        return self.residual_head(
            base_action_chunk=base_action_chunk.to(dtype=target_dtype),
            state_feature=state_features,
            vla_feature=vla_features,
            history_action=history_action,
        )

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if (
        "Cosmos-Reason2" in config.model_name
        or "Qwen3-VL" in config.model_name
    ):
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        # Residual-only training: caller can pass ``residual_freeze_vla=True``
        # on the config and everything except the residual head becomes
        # frozen. Joint training leaves flags untouched.
        if getattr(config, "residual_freeze_vla", False):
            self.action_head.freeze_vla_for_residual_only()
            # Freeze backbone as well.
            for p in self.backbone.parameters():
                p.requires_grad = False
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])[
                "inputs"
            ]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(
        self, inputs: dict, options: dict[str, Any] | None = None
    ) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs, options
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
