"""LeRobot parquet 数据加载器 (a3 布局, 跟 a2 完全不同)。

A3 parquet action 175D, 跟 a2 的 119D 布局**不兼容**, 字段索引也不同。
本文件只处理 a3 布局; a2 那套放在 a2_server.replay 里, 互不依赖。

A3 默认 action 索引 (来自实测 info.json):
  hand/position           [ 0, 20)  20D  双手 actuator (0..4096)
  hand/activejointpos     [20, 40)  20D  双手主动关节弧度 (经 OmnihandCtrl 转 actuator)
  hand/effort             [40, 60)  20D  (回放不用)
  end/orientation         [60, 68)   8D  双臂 [lq_xyzw, rq_xyzw]
  end/position            [68, 74)   6D  双臂 [lp, rp]
  arm/effort              [74, 88)  14D  (回放不用)
  arm/position            [88,102)  14D  双臂关节角弧度
  arm/velocity            [102,116) 14D  (回放不用)
  leg/effort              [116,128) 12D  (回放不用)
  leg/position            [128,140) 12D  腿部关节角 (回放暂不发, 只读)
  leg/velocity            [140,152) 12D  (回放不用)
  velocity/angular        [152]     1D
  velocity/forward        [153]     1D
  velocity/lateral        [154]     1D
  waist/effort            [155,158)  3D  (回放不用)
  waist/position          [158,162)  4D  [yaw, roll, pitch, height]   <- 注意 4D 含 height
  waist/velocity          [162,165)  3D  (回放不用)
  head/effort             [165,167)  2D  (回放不用)
  head/position           [167,169)  2D  [shake, nod]
  head/velocity           [169,171)  2D  (回放不用)
  gripper/position        [171,173)  2D  AgiClaw actuator (0..4096)
  gripper/effort          [173,175)  2D  (回放不用)

下游 (a3_server.replay) 拿到这些 ndarray 直接 POST 到 server 的 HTTP 端点。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# pandas 仅在 load_parquet_action 内部需要 (parquet 解析); 模块顶层延后 import
# 让只用 detect_control_mode / find_*_json 的调用方 (replay --help / 选模式 A H5)
# 在没装 pandas 的环境里也能 import 本模块。


# ==================== meta 默认值 (A3 布局) ====================
_DEFAULT_META = {
    "hand_position":         {"start":   0, "end":  20},   # actuator 直接
    "hand_activejointpos":   {"start":  20, "end":  40},   # 弧度 (OmnihandCtrl 转)
    "end_orientation":       {"start":  60, "end":  68},   # [lq_xyzw, rq_xyzw]
    "end_position":          {"start":  68, "end":  74},   # [lp, rp]
    "arm_joint":             {"start":  88, "end": 102},   # 14D 弧度
    "leg_position":          {"start": 128, "end": 140},
    "loco_angular":          {"start": 152, "end": 153},
    "loco_forward":          {"start": 153, "end": 154},
    "loco_lateral":          {"start": 154, "end": 155},
    "waist_position":        {"start": 158, "end": 162},   # 4D [yaw, roll, pitch, height]
    "head_position":         {"start": 167, "end": 169},   # 2D [shake, nod]
    "gripper_position":      {"start": 171, "end": 173},   # 2D
    # quat_order: 默认 xyzw, 可以被 modality.json 覆盖
    "_eef_quat_order":       "xyzw",
}

# info.json::features.action.field_descriptions 的 key -> meta 中的 key
_INFO_KEY_MAP = {
    "action/hand/position":             ("hand_position", "range"),
    "action/hand/activejointpos":       ("hand_activejointpos", "range"),
    "action/end/orientation":           ("end_orientation", "range"),
    "action/end/position":              ("end_position", "range"),
    "action/arm/position":              ("arm_joint", "range"),
    "action/leg/position":              ("leg_position", "range"),
    "action/velocity/angular_velocity": ("loco_angular", "range"),
    "action/velocity/forward_velocity": ("loco_forward", "range"),
    "action/velocity/lateral_velocity": ("loco_lateral", "range"),
    "action/waist/position":            ("waist_position", "range"),
    "action/head/position":             ("head_position", "range"),
    "action/gripper/position":          ("gripper_position", "range"),
}


def _find_dataset_meta(parquet_path: Path, name: str) -> Optional[Path]:
    """Walk up from the parquet to locate ``<dataset>/meta/<name>``."""
    cur = parquet_path.resolve().parent
    for _ in range(4):
        candidate = cur / "meta" / name
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def find_modality_json(parquet_path: Path) -> Optional[Path]:
    return _find_dataset_meta(parquet_path, "modality.json")


def find_info_json(parquet_path: Path) -> Optional[Path]:
    return _find_dataset_meta(parquet_path, "info.json")


def _load_action_meta(
    modality_json_path: Optional[Path],
    info_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return ``{key: group_info}`` assembled from defaults + info.json + modality.json."""
    meta: Dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v)
                            for k, v in _DEFAULT_META.items()}

    if info_json_path is not None and info_json_path.exists():
        with open(info_json_path, "r") as f:
            info = json.load(f)
        action_feat = info.get("features", {}).get("action", {})
        fdesc = action_feat.get("field_descriptions", {})
        for info_key, (meta_key, _mode) in _INFO_KEY_MAP.items():
            if info_key not in fdesc:
                continue
            idxs = fdesc[info_key].get("indices")
            if not idxs:
                continue
            # range 形式即可 (info.json 总是连续 indices); 兼容散列
            if list(idxs) == list(range(int(idxs[0]), int(idxs[-1]) + 1)):
                meta[meta_key] = {"start": int(idxs[0]), "end": int(idxs[-1]) + 1}
            else:
                meta[meta_key] = {"indices": [int(x) for x in idxs]}

    if modality_json_path is not None and modality_json_path.exists():
        with open(modality_json_path, "r") as f:
            raw = json.load(f)
        for key, entry in raw.get("action", {}).items():
            meta[key] = entry
    return meta


def _slice_by_info(flat: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    """Slice ``(T, D_total)`` by a meta group dict."""
    if "indices" in info:
        return flat[:, np.asarray(info["indices"], dtype=np.int64)]
    if "start" in info and "end" in info:
        return flat[:, int(info["start"]):int(info["end"])]
    raise KeyError(
        f"meta group missing indices/start-end: keys={list(info.keys())}"
    )


def load_parquet_action(
    parquet_path: Path,
    modality_json_path: Optional[Path],
    info_json_path: Optional[Path],
    control_mode: str,
    parts: set,
    hand_kind: str = "hand",
    hand_source: str = "auto",
) -> Dict[str, np.ndarray]:
    """Load a single-episode parquet and slice it into replay-ready arrays.

    Args:
        control_mode: 'joint' 或 'eef'
        parts: {'arm', 'hand', 'waist', 'head', 'loco'} 的子集
        hand_kind: 'hand' (O10 20D) 或 'gripper' (AgiClaw 2D)
        hand_source: 'auto' / 'position' (直接 actuator) / 'activejointpos' (弧度)
                      'auto' = 优先 position (省一次 OmnihandCtrl 转换)

    Returns dict keys (按需填充):
      - arm_joint:  (N, 14) 弧度
      - eef_14d:    (N, 14) [lp(3), rp(3), lq_xyzw(4), rq_xyzw(4)]
      - hand_actuator: (N, 20) actuator 0..4096   <- hand_kind=hand
      - hand_radians:  (N, 20) 弧度 (调用方需用 OmnihandCtrl 转 actuator)
      - gripper:    (N, 2) actuator               <- hand_kind=gripper
      - waist_4d:   (N, 4) [yaw, roll, pitch, height]
      - head:       (N, 2) [shake, nod]
      - loco_fwd / loco_lat / loco_ang: (N,) 标量
    """
    import pandas as pd  # 延后 import: H5 模式 (replay --help / 模式 A) 不需要 pandas
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        raise KeyError(
            f"parquet {parquet_path} has no 'action' column; got {list(df.columns)}"
        )
    flat = np.vstack([np.asarray(a, dtype=np.float32) for a in df["action"]])
    meta = _load_action_meta(modality_json_path, info_json_path)

    out: Dict[str, np.ndarray] = {}

    if "arm" in parts:
        if control_mode == "joint":
            out["arm_joint"] = _slice_by_info(flat, meta["arm_joint"]).astype(np.float32)
        else:
            # A3 EEF: end/position 6D [lp, rp], end/orientation 8D [lq, rq]
            ep = _slice_by_info(flat, meta["end_position"])
            eo = _slice_by_info(flat, meta["end_orientation"])
            if ep.shape[1] != 6 or eo.shape[1] != 8:
                raise ValueError(
                    f"A3 EEF 期望 end/position=6D, end/orientation=8D; "
                    f"实际 ep={ep.shape[1]}D, eo={eo.shape[1]}D"
                )
            lp, rp = ep[:, :3], ep[:, 3:6]
            lq, rq = eo[:, :4], eo[:, 4:8]
            quat_order = str(meta.get("_eef_quat_order", "xyzw")).lower()
            if quat_order == "wxyz":
                lq = np.stack([lq[:, 1], lq[:, 2], lq[:, 3], lq[:, 0]], axis=-1)
                rq = np.stack([rq[:, 1], rq[:, 2], rq[:, 3], rq[:, 0]], axis=-1)
            out["eef_14d"] = np.hstack([lp, rp, lq, rq]).astype(np.float32)

    if "hand" in parts:
        if hand_kind == "gripper":
            if "gripper_position" not in meta:
                raise KeyError(
                    "hand_kind=gripper 但 parquet meta 没有 gripper_position 字段"
                )
            out["gripper"] = _slice_by_info(flat, meta["gripper_position"]).astype(np.float32)
        else:
            # hand_source: auto 优先 position (actuator), 没字段则回退 activejointpos
            use_pos = (hand_source == "position") or (
                hand_source == "auto" and "hand_position" in meta
            )
            if use_pos and "hand_position" in meta:
                out["hand_actuator"] = _slice_by_info(flat, meta["hand_position"]).astype(np.float32)
            elif "hand_activejointpos" in meta:
                out["hand_radians"] = _slice_by_info(flat, meta["hand_activejointpos"]).astype(np.float32)
            else:
                raise KeyError(
                    "hand_kind=hand 但 parquet meta 没有 hand_position 或 hand_activejointpos"
                )

    if "waist" in parts and "waist_position" in meta:
        wp = _slice_by_info(flat, meta["waist_position"]).astype(np.float32)
        if wp.shape[1] != 4:
            raise ValueError(
                f"A3 waist 期望 4D [yaw, roll, pitch, height], 实际 {wp.shape[1]}D"
            )
        out["waist_4d"] = wp

    if "head" in parts and "head_position" in meta:
        out["head"] = _slice_by_info(flat, meta["head_position"]).astype(np.float32)

    if "loco" in parts:
        for out_key, meta_key in (
            ("loco_fwd", "loco_forward"),
            ("loco_lat", "loco_lateral"),
            ("loco_ang", "loco_angular"),
        ):
            if meta_key in meta:
                out[out_key] = _slice_by_info(flat, meta[meta_key]).reshape(-1).astype(np.float32)

    return out


def detect_control_mode(
    modality_json_path: Optional[Path],
    info_json_path: Optional[Path],
) -> str:
    """Auto-pick 'eef' vs 'joint' based on modality.json action keys.

    info.json 同时存 joint+eef 字段 (描述布局), 唯有 modality.json 反映模型
    实际驱动的视图。modality.json 的 action 键里出现 left_eef/right_eef 或
    eef 字样 -> 选 'eef', 否则选 'joint'。没 modality.json 时回退 'joint'。
    """
    if modality_json_path is not None and modality_json_path.exists():
        with open(modality_json_path, "r") as f:
            raw = json.load(f)
        action_keys = set(raw.get("action", {}).keys())
        if any("eef" in k for k in action_keys):
            return "eef"
        if "arm_joint" in action_keys:
            return "joint"
    return "joint"
