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

"""Small residual head that predicts a bounded, additive correction to a base
VLA action chunk.

Design goals:
- Zero-init the final layer so an untrained head is a no-op (safe fallback).
- Bound the output with tanh so a runaway residual can't destabilize control.
- Feature-in / delta-out API that works both as an inline module inside
  Gr00tN1d7 (training + inference-side compensation) and as a standalone
  60Hz runner on the robot (see robotinterface/RoboInterface).
- Fixed-shape output `(B, output_horizon, action_dim)`. At deployment time
  the caller sets `output_horizon` to the 60Hz interpolated chunk length
  (e.g. 3× the VLA chunk if VLA outputs at 20Hz and the robot runs at 60Hz).

The head is intentionally small (~1M params) — the previous A3/G1 sim2sim
prototype confirmed that a 512→256 MLP with tanh-bound is enough to visibly
compensate lag/rate errors in the base VLA chunk within a few hundred
samples of training.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualHead(nn.Module):
    """Predict per-step correction delta given base action + conditioning features.

    Inputs (all except ``base_action_chunk`` optional; the head consumes only
    the streams present in ``feature_dims``):
        base_action_chunk:  (B, output_horizon, action_dim)
            The chunk we're correcting. When operating at 60Hz on the robot
            this is the already-interpolated chunk; when training end-to-end
            this is the VLA's predicted chunk before residual is applied.
        state_feature:      (B, state_feature_dim)
            Pooled state encoding.
        vla_feature:        (B, vla_feature_dim)
            Pooled VLA backbone features (or last-layer of the action head)
            used as "VLA hidden" cache on the robot side.
        history_action:     (B, history_len, action_dim)
            Recent executed actions (optional; helps smooth transitions
            across chunk boundaries).

    Output:
        delta: (B, output_horizon, action_dim)
            Bounded by ``delta_bound`` via tanh. Last layer is zero-init so
            an untrained head returns exactly 0.
    """

    def __init__(
        self,
        action_dim: int,
        output_horizon: int,
        hidden_dim: int = 512,
        cond_dim: int = 256,
        state_feature_dim: int | None = None,
        vla_feature_dim: int | None = None,
        history_len: int = 0,
        delta_bound: float = 0.3,
        step_embed_dim: int = 32,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.output_horizon = output_horizon
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.state_feature_dim = state_feature_dim
        self.vla_feature_dim = vla_feature_dim
        self.history_len = history_len
        self.delta_bound = float(delta_bound)
        self.step_embed_dim = step_embed_dim

        # Conditioning encoders (each optional). Missing inputs are silently
        # replaced with zeros at forward time so the head is robust to callers
        # that skip a stream (e.g. robot deployment without vla_feature).
        encoders: dict[str, nn.Module] = {}
        if state_feature_dim is not None and state_feature_dim > 0:
            encoders["state"] = nn.Sequential(
                nn.Linear(state_feature_dim, cond_dim),
                nn.SiLU(),
            )
        if vla_feature_dim is not None and vla_feature_dim > 0:
            encoders["vla"] = nn.Sequential(
                nn.Linear(vla_feature_dim, cond_dim),
                nn.SiLU(),
            )
        if history_len > 0:
            encoders["history"] = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(history_len * action_dim, cond_dim),
                nn.SiLU(),
            )
        self.cond_encoders = nn.ModuleDict(encoders)
        # Bias vector — used when no conditioning stream is available so the
        # residual pathway still has a well-defined input.
        self.cond_bias = nn.Parameter(torch.zeros(cond_dim))

        # Fixed sinusoidal step embedding for the output horizon. Broadcast
        # to every batch. Not learned so training is stable across horizon
        # changes and this can be recomputed on the fly at deployment when
        # the caller wants a longer 60Hz chunk than seen in training.
        self.register_buffer(
            "_step_embed", self._make_step_embed(output_horizon, step_embed_dim)
        )

        per_step_in = action_dim + step_embed_dim + cond_dim
        layers: list[nn.Module] = []
        d_in = per_step_in
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.SiLU())
            d_in = hidden_dim
        self.mlp = nn.Sequential(*layers)
        # Zero-init the final projection so an untrained head is identity.
        self.delta_out = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    @staticmethod
    def _make_step_embed(horizon: int, dim: int) -> torch.Tensor:
        """Sinusoidal positional embedding indexed by step position."""
        half = dim // 2
        pos = torch.arange(horizon, dtype=torch.float32)
        # log-spaced frequencies akin to Transformer sinusoidal PE
        div = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        angles = pos[:, None] * div[None, :]  # (H, half)
        emb = torch.zeros(horizon, dim)
        emb[:, :half] = torch.sin(angles)
        emb[:, half : 2 * half] = torch.cos(angles)
        return emb

    def build_conditioning(
        self,
        state_feature: torch.Tensor | None = None,
        vla_feature: torch.Tensor | None = None,
        history_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate + sum available conditioning streams to (B, cond_dim)."""
        parts: list[torch.Tensor] = []
        if "state" in self.cond_encoders and state_feature is not None:
            if state_feature.dim() == 3:
                state_feature = state_feature.mean(dim=1)
            parts.append(self.cond_encoders["state"](state_feature))
        if "vla" in self.cond_encoders and vla_feature is not None:
            if vla_feature.dim() == 3:
                vla_feature = vla_feature.mean(dim=1)
            parts.append(self.cond_encoders["vla"](vla_feature))
        if "history" in self.cond_encoders and history_action is not None:
            parts.append(self.cond_encoders["history"](history_action))
        if not parts:
            # Nothing supplied — fall back to the learned bias so the head
            # still produces something coherent (delta will still be zero
            # from the zero-init output layer at t=0 of training).
            return self.cond_bias.unsqueeze(0)
        return sum(parts) + self.cond_bias

    def forward(
        self,
        base_action_chunk: torch.Tensor,
        *,
        state_feature: torch.Tensor | None = None,
        vla_feature: torch.Tensor | None = None,
        history_action: torch.Tensor | None = None,
        step_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the bounded delta with the same shape as ``base_action_chunk``.

        Args:
            base_action_chunk: (B, H_out, action_dim). H_out may differ from
                ``self.output_horizon`` — the step embedding is recomputed
                on the fly in that case.
            step_indices: optional (H_out,) or (B, H_out) integer positions
                into the 60Hz grid. Lets the caller run the head incrementally
                (one 60Hz step at a time) while keeping consistent PE.
        """
        assert base_action_chunk.dim() == 3, (
            f"base_action_chunk must be (B, H, D), got {tuple(base_action_chunk.shape)}"
        )
        B, H_out, D = base_action_chunk.shape
        assert D == self.action_dim, (
            f"action_dim mismatch: got {D}, expected {self.action_dim}"
        )

        cond = self.build_conditioning(
            state_feature=state_feature,
            vla_feature=vla_feature,
            history_action=history_action,
        )  # (B or 1, cond_dim)
        if cond.shape[0] == 1 and B > 1:
            cond = cond.expand(B, -1)
        cond_seq = cond.unsqueeze(1).expand(B, H_out, cond.shape[-1])

        if step_indices is None:
            if H_out == self.output_horizon:
                step_emb = self._step_embed
            else:
                step_emb = self._make_step_embed(H_out, self.step_embed_dim).to(
                    device=base_action_chunk.device, dtype=base_action_chunk.dtype
                )
            step_emb = step_emb.unsqueeze(0).expand(B, H_out, self.step_embed_dim)
        else:
            if step_indices.dim() == 1:
                step_indices = step_indices.unsqueeze(0).expand(B, -1)
            # On-the-fly PE lookup — works even for indices beyond the buffer.
            step_emb = self._make_step_embed(
                int(step_indices.max().item()) + 1, self.step_embed_dim
            ).to(device=base_action_chunk.device, dtype=base_action_chunk.dtype)
            step_emb = step_emb[step_indices]  # (B, H_out, step_embed_dim)

        step_emb = step_emb.to(dtype=base_action_chunk.dtype)
        cond_seq = cond_seq.to(dtype=base_action_chunk.dtype)
        per_step_in = torch.cat([base_action_chunk, step_emb, cond_seq], dim=-1)
        hidden = self.mlp(per_step_in)
        delta = self.delta_out(hidden)
        delta = torch.tanh(delta) * self.delta_bound
        return delta

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualHeadOutput:
    """Small carrier so callers can plumb the raw delta + the compensated
    action chunk out of Gr00tN1d7ActionHead without polluting BatchFeature.

    Not a dataclass to avoid pulling in dataclasses/pydantic at import time
    on inference-only paths (Orin/Thor)."""

    __slots__ = ("delta", "action_pred", "action_pred_uncompensated")

    def __init__(
        self,
        delta: torch.Tensor,
        action_pred: torch.Tensor,
        action_pred_uncompensated: torch.Tensor,
    ):
        self.delta = delta
        self.action_pred = action_pred
        self.action_pred_uncompensated = action_pred_uncompensated
