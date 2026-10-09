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
Calculate dataset statistics for LeRobot datasets.

Usage:
    python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag <embodiment_tag>
    python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag <embodiment_tag> --modality-config-path <config.py>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations.
    modality_config_path: Optional path to a .py config file for custom embodiment tags not in the built-in registry.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import (
    EndEffectorActionChunk,
    JointActionChunk,
)
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.types import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
from gr00t.data.utils import to_json_serializable

LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"
# Bare filenames used when stats are stored outside the dataset's ``meta/``
# directory (e.g., under a user-specified ``stats_dir``).
STATS_BASENAME = "stats.json"
REL_STATS_BASENAME = "relative_stats.json"


def _resolve_stats_paths(
    dataset_path: Path, stats_dir: Path | str | None
) -> tuple[Path, Path]:
    """Return (stats_path, relative_stats_path) to read/write for this dataset.

    If ``stats_dir`` is given, stats live under
    ``<stats_dir>/<dataset_name>/{stats.json,relative_stats.json}`` so a
    single external directory can hold stats for multiple read-only datasets.
    Otherwise the canonical LeRobot location under ``<dataset>/meta/`` is used.
    """
    if stats_dir is not None:
        out_dir = Path(stats_dir) / dataset_path.name
        return (out_dir / STATS_BASENAME, out_dir / REL_STATS_BASENAME)
    return (
        dataset_path / LE_ROBOT_STATS_FILENAME,
        dataset_path / LE_ROBOT_REL_STATS_FILENAME,
    )


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(
    stats_path: Path | str, features: list[str]
):
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(
    dataset_path: Path | str,
    stats_dir: Path | str | None = None,
):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        info = json.load(f)
    le_features = info["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)

    stats_path, _ = _resolve_stats_paths(dataset_path, stats_dir)
    if check_stats_validity(stats_path, lowdim_features):
        return

    # Honor info.json["data_path"] (same as LeRobotEpisodeLoader); fall back to legacy default.
    data_path_pattern = info.get("data_path", LE_ROBOT_DATA_FILENAME)
    glob_pattern = re.sub(r"\{[^}]*\}", "*", data_path_pattern)
    parquet_files = list(dataset_path.glob(glob_pattern))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    # --- Vectorized helpers ---------------------------------------------
    @staticmethod
    def _rot6d_to_matrix_batch(rot6d: np.ndarray) -> np.ndarray:
        """Convert (..., 6) rot6d to (..., 3, 3) rotation matrix.

        Uses Gram-Schmidt on the first two rows, third row is their cross
        product. Fully vectorized. Mirrors
        ``EndEffectorPose._rot6d_to_matrix`` but batched.
        """
        rot6d = np.asarray(rot6d, dtype=np.float64)
        # (..., 2, 3)
        pair = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
        row1 = pair[..., 0, :]
        row2 = pair[..., 1, :]

        row1 = row1 / np.linalg.norm(row1, axis=-1, keepdims=True).clip(min=1e-12)
        # Remove projection of row2 onto row1
        dot = np.sum(row1 * row2, axis=-1, keepdims=True)
        row2 = row2 - dot * row1
        row2 = row2 / np.linalg.norm(row2, axis=-1, keepdims=True).clip(min=1e-12)
        row3 = np.cross(row1, row2)
        # Stack into (..., 3, 3); rows are the basis vectors.
        rot_mat = np.stack([row1, row2, row3], axis=-2)
        return rot_mat

    @staticmethod
    def _matrix_to_rot6d_batch(rot_mat: np.ndarray) -> np.ndarray:
        """Convert (..., 3, 3) rotation matrix to (..., 6) rot6d."""
        return rot_mat[..., :2, :].reshape(*rot_mat.shape[:-2], 6)

    @staticmethod
    def _rotvec_to_matrix_batch(rotvec: np.ndarray) -> np.ndarray:
        """Convert (..., 3) rotation vectors to (..., 3, 3) matrices via
        Rodrigues' formula, fully vectorized."""
        rotvec = np.asarray(rotvec, dtype=np.float64)
        theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)  # (..., 1)
        # Avoid div-by-zero; where theta≈0, rotation is identity.
        safe_theta = np.where(theta < 1e-12, 1.0, theta)
        k = rotvec / safe_theta  # (..., 3)
        kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
        zeros = np.zeros_like(kx)
        K = np.stack(
            [
                np.stack([zeros, -kz, ky], axis=-1),
                np.stack([kz, zeros, -kx], axis=-1),
                np.stack([-ky, kx, zeros], axis=-1),
            ],
            axis=-2,
        )  # (..., 3, 3)
        I = np.broadcast_to(np.eye(3), K.shape).copy()
        theta_ = theta[..., None]  # (..., 1, 1)
        sin_t = np.sin(theta_)
        cos_t = np.cos(theta_)
        rot_mat = I + sin_t * K + (1 - cos_t) * (K @ K)
        # Where theta≈0, force identity (avoids tiny nonzero noise).
        near_zero = (theta < 1e-12)[..., None]  # (..., 1, 1)
        rot_mat = np.where(near_zero, I, rot_mat)
        return rot_mat

    @staticmethod
    def _matrix_to_rotvec_batch(rot_mat: np.ndarray) -> np.ndarray:
        """Convert (..., 3, 3) rotation matrices to (..., 3) rotation vectors.

        Uses the analytic axis-angle formula with stable branch selection.
        """
        # trace = R[0,0] + R[1,1] + R[2,2]
        trace = (
            rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]
        )
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # (..., )
        # Axis components from skew-symmetric part.
        rx = rot_mat[..., 2, 1] - rot_mat[..., 1, 2]
        ry = rot_mat[..., 0, 2] - rot_mat[..., 2, 0]
        rz = rot_mat[..., 1, 0] - rot_mat[..., 0, 1]
        axis = np.stack([rx, ry, rz], axis=-1)  # (..., 3)
        sin_theta = np.sin(theta)[..., None]
        # Where sin_theta is tiny, axis from skew is ~zero; theta≈0 → rotvec=0.
        safe_sin = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)
        rotvec = axis / (2.0 * safe_sin) * theta[..., None]
        rotvec = np.where(
            np.abs(sin_theta) < 1e-8, np.zeros_like(rotvec), rotvec
        )
        return rotvec

    @classmethod
    def _build_homogeneous_from_array_batch(
        cls, data: np.ndarray, action_format: ActionFormat
    ) -> np.ndarray:
        """Vectorized EndEffectorPose.from_action_format -> 4x4 matrices.

        ``data`` may be any shape ending in D; output shape is ``data.shape[:-1] + (4,4)``.
        """
        data = np.asarray(data, dtype=np.float64)
        leading = data.shape[:-1]
        if action_format == ActionFormat.XYZ_ROT6D:
            xyz = data[..., :3]
            rot_mat = cls._rot6d_to_matrix_batch(data[..., 3:])
        elif action_format == ActionFormat.XYZ_ROTVEC:
            xyz = data[..., :3]
            rot_mat = cls._rotvec_to_matrix_batch(data[..., 3:])
        elif action_format == ActionFormat.ROT6D:
            # Pure rotation, zero translation — reuses the same SE(3) code path
            # as XYZ_ROT6D for relative composition. Translation stays zero
            # through matrix multiplication, so the inverse mapping cleanly
            # returns just the rot6d.
            xyz = np.zeros(leading + (3,), dtype=np.float64)
            rot_mat = cls._rot6d_to_matrix_batch(data)
        elif action_format == ActionFormat.DEFAULT:
            # Already 4x4 homogeneous, flattened.
            return data.reshape(*leading, 4, 4)
        else:
            raise ValueError(f"Unsupported ActionFormat: {action_format}")

        H = np.zeros(leading + (4, 4), dtype=np.float64)
        H[..., :3, :3] = rot_mat
        H[..., :3, 3] = xyz
        H[..., 3, 3] = 1.0
        return H

    @classmethod
    def _homogeneous_to_array_batch(
        cls, H: np.ndarray, action_format: ActionFormat
    ) -> np.ndarray:
        """Vectorized inverse of ``_build_homogeneous_from_array_batch``."""
        xyz = H[..., :3, 3]
        rot_mat = H[..., :3, :3]
        if action_format == ActionFormat.XYZ_ROT6D:
            rot = cls._matrix_to_rot6d_batch(rot_mat)
            return np.concatenate([xyz, rot], axis=-1)
        elif action_format == ActionFormat.XYZ_ROTVEC:
            rot = cls._matrix_to_rotvec_batch(rot_mat)
            return np.concatenate([xyz, rot], axis=-1)
        elif action_format == ActionFormat.ROT6D:
            # Rotation-only inverse: drop the translation column (which stayed
            # zero throughout since the forward mapping wrote it as zero).
            return cls._matrix_to_rot6d_batch(rot_mat)
        elif action_format == ActionFormat.DEFAULT:
            return H.reshape(*H.shape[:-2], 16)
        else:
            raise ValueError(f"Unsupported ActionFormat: {action_format}")

    @classmethod
    def _compute_relative_eef_batch(
        cls,
        absolute_actions: np.ndarray,
        reference_states: np.ndarray,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Fully vectorized EEF relative-chunking computation.

        Args:
            absolute_actions: (N, H, D) absolute action chunks.
            reference_states: (N, D) per-chunk reference frame.
            action_format: Action format (xyz+rot6d, xyz+rotvec, or default).

        Returns:
            (N, H, D) relative action chunks (same layout / format).
        """
        H_actions = cls._build_homogeneous_from_array_batch(
            absolute_actions, action_format
        )  # (N, H, 4, 4)
        H_ref = cls._build_homogeneous_from_array_batch(
            reference_states, action_format
        )  # (N, 4, 4)
        # T_relative = inv(T_ref) @ T_action, broadcasted over the H axis.
        H_ref_inv = np.linalg.inv(H_ref)  # (N, 4, 4)
        H_rel = np.einsum("nij,nhjk->nhik", H_ref_inv, H_actions)
        return cls._homogeneous_to_array_batch(H_rel, action_format)


    def __init__(
        self,
        dataset_path: Path | str,
        embodiment_tag: EmbodimentTag,
        action_key: str,
        modality_meta_path: Path | str | None = None,
        stats_dir: Path | str | None = None,
        tasks_dir: Path | str | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        # Check action config
        assert (
            action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        )
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(
            action_key
        )
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value][
            "action"
        ]
        self.action_config = action_configs[idx]
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value][
                "action"
            ].delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        state_modality_keys = MODALITY_CONFIGS[embodiment_tag.value][
            "state"
        ].modality_keys
        state_reference_only_keys = (
            MODALITY_CONFIGS[embodiment_tag.value]["state"].reference_only_keys or []
        )
        assert (
            state_key in state_modality_keys
            or state_key in state_reference_only_keys
        ), (
            f"state_key '{state_key}' for RELATIVE action '{action_key}' must be listed "
            f"either in state.modality_keys or state.reference_only_keys; got "
            f"modality_keys={state_modality_keys} reference_only_keys={state_reference_only_keys}"
        )
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )
        self.loader = LeRobotEpisodeLoader(
            dataset_path,
            self.modality_configs,
            modality_meta_path=modality_meta_path,
            stats_dir=stats_dir,
            tasks_dir=tasks_dir,
        )

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        # ``df[...].values`` returns an object array of per-row arrays; stack
        # them into a dense float array so the vectorized path can slice by
        # fancy indexing.
        state_data = np.stack(list(df[state_key].values)).astype(np.float64)
        action_data = np.stack(list(df[action_key].values)).astype(np.float64)

        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        if usable_length <= 0:
            return []
        action_delta_indices = np.array(
            self.modality_configs["action"].delta_indices
        )
        state_offset = self.modality_configs["state"].delta_indices[-1]

        # Build batched references and action chunks.
        # state_inds: (usable_length,) ; action_inds: (usable_length, H)
        step_idx = np.arange(usable_length)
        state_inds = state_offset + step_idx
        action_inds = action_delta_indices[None, :] + step_idx[:, None]

        ref_states = state_data[state_inds]  # (N, D)
        actions = action_data[action_inds]  # (N, H, D)

        if self.action_config.type == ActionType.EEF:
            action_format = self.action_config.format
            rel = self._compute_relative_eef_batch(
                actions, ref_states, action_format
            )
            return [rel.astype(np.float32)]
        elif self.action_config.type == ActionType.NON_EEF:
            if self.action_config.format == ActionFormat.ROT6D:
                # Pure 6D rotation: route through the same batched SE(3) code
                # path as XYZ_ROT6D (zero-translation build/inverse handled
                # inside _build_homogeneous_from_array_batch /
                # _homogeneous_to_array_batch).
                rel = self._compute_relative_eef_batch(
                    actions, ref_states, ActionFormat.ROT6D
                )
                return [rel.astype(np.float32)]
            # JointPose subtraction is plain element-wise subtraction.
            rel = actions - ref_states[:, None, :]
            return [rel.astype(np.float32)]
        else:
            raise ValueError(f"Unknown ActionType: {self.action_config.type}")

    def __len__(self) -> int:
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
    modality_meta_path: Path | str | None = None,
    stats_dir: Path | str | None = None,
    tasks_dir: Path | str | None = None,
) -> dict:
    loader = RelativeActionLoader(
        dataset_path,
        embodiment_tag,
        group_key,
        modality_meta_path=modality_meta_path,
        stats_dir=stats_dir,
        tasks_dir=tasks_dir,
    )
    trajectories: list[np.ndarray] = []
    for episode_id in tqdm(
        range(len(loader)), desc=f"Loading trajectories for key {group_key}"
    ):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id))
    if not trajectories:
        raise ValueError(
            f"No trajectories produced for key {group_key}; dataset may be too short"
        )
    # Each entry is either (N_chunks, H, D) from the vectorized EEF path,
    # or (H, D) from the legacy per-step path. Normalize to (N_chunks, H, D).
    normalized = []
    for t in trajectories:
        if t.ndim == 2:
            normalized.append(t[None, ...])
        else:
            normalized.append(t)
    all_traj = np.concatenate(normalized, axis=0)  # (N, H, D)
    return {
        "max": np.max(all_traj, axis=0),
        "min": np.min(all_traj, axis=0),
        "q01": np.quantile(all_traj, 0.01, axis=0),
        "q99": np.quantile(all_traj, 0.99, axis=0),
        "mean": np.mean(all_traj, axis=0),
        "std": np.std(all_traj, axis=0),
    }


def generate_rel_stats(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    modality_meta_path: Path | str | None = None,
    stats_dir: Path | str | None = None,
    tasks_dir: Path | str | None = None,
) -> None:
    dataset_path = Path(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(
            action_config.modality_keys, action_config.action_configs
        )
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    _, stats_path = _resolve_stats_paths(dataset_path, stats_dir)
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    for action_key in sorted(action_keys):
        if action_key in stats:
            continue
        print(
            f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}"
        )
        stats[action_key] = calculate_stats_for_key(
            dataset_path,
            embodiment_tag,
            action_key,
            modality_meta_path=modality_meta_path,
            stats_dir=stats_dir,
            tasks_dir=tasks_dir,
        )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    modality_config_path: str | None = None,
    modality_meta_path: Path | str | None = None,
    stats_dir: Path | str | None = None,
    tasks_dir: Path | str | None = None,
):
    """Generate dataset statistics.

    Args:
        dataset_path: Path to the dataset.
        embodiment_tag: Embodiment tag for modality configurations.
        modality_config_path: Optional path to a .py modality config file. Required for custom
            embodiment tags not in the built-in MODALITY_CONFIGS registry.
        modality_meta_path: Optional path to a SHARED ``modality.json`` used
            instead of ``<dataset_path>/meta/modality.json``. Must match the
            one used at training time if you pre-generate stats offline.
        stats_dir: Optional directory to store stats outside the dataset's
            meta/ dir. When set, writes to
            ``<stats_dir>/<dataset_name>/{stats.json,relative_stats.json}``.
        tasks_dir: Optional directory to load ``tasks.jsonl`` from instead of
            the dataset's own meta/ dir. When set, reads from
            ``<tasks_dir>/<dataset_name>/tasks.jsonl``.
    """
    if modality_config_path is not None:
        import importlib
        import sys

        config_path = Path(modality_config_path)
        if config_path.exists() and config_path.suffix == ".py":
            sys.path.append(str(config_path.parent))
            importlib.import_module(config_path.stem)
            print(f"Loaded modality config: {config_path}")
        else:
            raise FileNotFoundError(
                f"Modality config path does not exist or is not a .py file: {modality_config_path}"
            )
    generate_stats(dataset_path, stats_dir=stats_dir)
    generate_rel_stats(
        dataset_path,
        embodiment_tag,
        modality_meta_path=modality_meta_path,
        stats_dir=stats_dir,
        tasks_dir=tasks_dir,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
