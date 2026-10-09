"""A3 机器人端 replay (ROS-native, 自己 publish + subscribe).

之前是 HTTP 客户端依赖 a3_server :5050 + 主机端 30Hz GET 录 state. 改成
直接当 ROS 节点 — publisher / subscriber 都用 ROS clock 时间戳, 没有 HTTP
延迟, 也不需要 server 在跑。

三种回放模式:

  ┌────┬───────────┬──────────────────────────┬──────────────┐
  │ 模式│ 数据源     │ arm 怎么发                │ pnc_arm mode │
  ├────┼───────────┼──────────────────────────┼──────────────┤
  │  A  │ H5         │ publish_arm_raw (mc, 不插值) │ passive      │
  │  B  │ parquet    │ publish_arm_target           │ passive      │
  │     │            │ (InterpolationPublisher 30→150Hz) │              │
  │  C  │ parquet    │ publish_arm_interpolate      │ trajectory   │
  │     │            │ (PncArmInterpolateChannel 机上插值) │              │
  └────┴───────────┴──────────────────────────┴──────────────┘

约束:
  - H5 + --use-onboard-interp -> 报错 (H5 已是密集轨迹, 再插会破坏波形)
  - --control-mode eef -> 强制模式 C (mc 通道没有 EEF)

用法 (ADU 上执行, 不需要先启 server):
  ros2 run a3_server replay --file-path /path/to/aligned_joints.h5
  ros2 run a3_server replay --parquet-path /path/to/episode_000000.parquet
  ros2 run a3_server replay --parquet-path ... --use-onboard-interp
  ros2 run a3_server replay --parquet-path ... --control-mode eef

跑完后, cmd_records 和 state_records 保存到 /agibot/replay_records.npz
(覆盖, 主机端 UI 用 scp 拉回画图)。
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
    QoSDurabilityPolicy, QoSLivelinessPolicy,
)
from sensor_msgs.msg import JointState

from a3_server.hand_ctrl import OmnihandCtrl
from a3_server.joint_config import HandKind
from a3_server.parquet_loader import (
    detect_control_mode, find_info_json, find_modality_json,
    load_parquet_action,
)
from a3_server.pnc_arm_mode import set_pnc_arm_mode
from a3_server.replay_node import ReplayNode

try:
    import h5py
except ImportError:
    h5py = None


REMOTE_RECORDS_PATH = "/agibot/replay_records.npz"  # 跑完保存到这里, UI scp 拉回


# ==================== hand_kind 自动探测 (ROS subscriber 拿 frame_id) ====================

def detect_hand_kind_via_topic(timeout_sec: float = 5.0) -> str:
    """订阅 /motion/control/hand_joint_state 等几秒, 拿 header.frame_id.

    底层 mc 守护进程开机就发这个 topic, 跟 server 是否启动无关。
    跟 detect_hand_kind_remote.sh 同思路, 只是这里用 rclpy 直接订阅。

    返回 "hand" / "gripper"; 探测失败 raise RuntimeError。
    """
    sub_node = rclpy.create_node("hand_kind_detect")
    detected: Dict[str, Optional[str]] = {"kind": None}

    def cb(msg):
        if detected["kind"] is not None:
            return
        f = (msg.header.frame_id or "").strip()
        if "O10Hand" in f:
            detected["kind"] = "hand"
        elif "AgiClaw" in f:
            detected["kind"] = "gripper"

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        liveliness=QoSLivelinessPolicy.AUTOMATIC,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    sub_node.create_subscription(
        JointState, "/motion/control/hand_joint_state", cb, qos_profile=qos)

    end = time.time() + timeout_sec
    while time.time() < end and detected["kind"] is None:
        rclpy.spin_once(sub_node, timeout_sec=0.1)

    sub_node.destroy_node()
    if detected["kind"] is None:
        raise RuntimeError(
            f"探测 hand_kind 超时 ({timeout_sec}s), 没收到 hand_joint_state msg; "
            f"mc 守护进程是否在跑? 可用 --hand-kind hand|gripper 跳过探测")
    return detected["kind"]


# ==================== H5 加载 (timestamp 对齐) ====================

def _resample_nearest(arr: np.ndarray, src_ts: np.ndarray,
                       dst_ts: np.ndarray) -> np.ndarray:
    n = len(src_ts)
    if n == 0 or len(dst_ts) == 0:
        shape = (len(dst_ts),) + arr.shape[1:]
        return np.empty(shape, dtype=arr.dtype)
    idx = np.searchsorted(src_ts, dst_ts, side="left")
    idx = np.clip(idx, 0, n - 1)
    idx_prev = np.clip(idx - 1, 0, n - 1)
    use_prev = np.abs(src_ts[idx_prev] - dst_ts) < np.abs(src_ts[idx] - dst_ts)
    idx = np.where(use_prev, idx_prev, idx)
    return arr[idx]


def load_h5_action(file_path: str, parts: set, control_mode: str,
                    hand_kind: str, fps: float = 30.0) -> Dict[str, np.ndarray]:
    """读 A3 H5 字段, 自动检测各 part 行数:

      - 行数都一致 → 直接读, 不重采样
      - 行数不一致 → 用 action/<part>/timestamp (ns int64) nearest-neighbor
        重采样到 [max(t0), min(t-1)] 公共时间窗 + 用户 fps 均匀采样

    A3 H5 字段命名跟 A3 parquet 一致 (action/arm/position 等), 跟 A2 H5
    (action/joint/position) 完全不兼容。
    """
    if h5py is None:
        raise RuntimeError("h5py 未安装, 无法读 H5; pip3 install --user h5py")
    out: Dict[str, np.ndarray] = {}
    with h5py.File(file_path, "r") as f:
        used_part_keys = []
        if "arm" in parts:
            used_part_keys.append("arm" if control_mode == "joint" else "end")
        if "hand" in parts:
            used_part_keys.append("gripper" if hand_kind == "gripper" else "hand")
        if "waist" in parts: used_part_keys.append("waist")
        if "head"  in parts: used_part_keys.append("head")
        if "loco"  in parts: used_part_keys.append("velocity")

        probe = {
            "arm":     "action/arm/position",
            "end":     "action/end/position",
            "hand":    ("action/hand/position" if "action/hand/position" in f
                        else "action/hand/activejointpos"),
            "gripper": "action/gripper/position",
            "waist":   "action/waist/position",
            "head":    "action/head/position",
            "velocity": "action/velocity/forward_velocity",
        }
        shapes = {k: f[probe[k]].shape[0]
                  for k in used_part_keys if probe.get(k) and probe[k] in f}
        need_align = (len(set(shapes.values())) > 1) if shapes else False

        ts_dict: Dict[str, np.ndarray] = {}
        dst_ts: Optional[np.ndarray] = None
        if need_align:
            for k in used_part_keys:
                tkey = f"action/{k}/timestamp"
                if tkey in f:
                    ts_dict[k] = np.asarray(f[tkey][:], dtype=np.int64)
            if ts_dict:
                t_lo = max(ts[0]  for ts in ts_dict.values())
                t_hi = min(ts[-1] for ts in ts_dict.values())
                if t_hi <= t_lo:
                    raise ValueError(
                        f"H5 各 part timestamp 没有公共时间窗 (t_lo={t_lo}, t_hi={t_hi})")
                duration_s = (t_hi - t_lo) / 1e9
                n_frames = max(1, int(round(duration_s * fps)))
                dst_ts = np.linspace(t_lo, t_hi, n_frames).astype(np.int64)
                print(f"  [align] 各 part 行数不一致 ({shapes}), 用 timestamp + "
                       f"fps={fps:.0f} 重采样到 {n_frames} 帧 ({duration_s:.2f}s)",
                       flush=True)
            else:
                print(f"  [align] 各 part 行数不一致 ({shapes}) 但缺 timestamp 字段, "
                       f"按原始数据读取 (replay 主循环会按 min 帧数 truncate)",
                       flush=True)
        else:
            print(f"  [align] 各 part 行数一致 ({shapes}), 跳过对齐", flush=True)

        def _read(field_path: str, part_key: str) -> np.ndarray:
            arr = np.asarray(f[field_path][:], dtype=np.float32)
            if dst_ts is None or part_key not in ts_dict:
                return arr
            src_ts = ts_dict[part_key]
            if len(src_ts) != len(arr):
                return arr
            return _resample_nearest(arr, src_ts, dst_ts)

        if "arm" in parts:
            if control_mode == "joint":
                if "action/arm/position" not in f:
                    raise KeyError("H5 没有 action/arm/position; control-mode=joint 不可用")
                out["arm_joint"] = _read("action/arm/position", "arm")
            else:
                if "action/end/position" not in f or "action/end/orientation" not in f:
                    raise KeyError("H5 没有 action/end/{position,orientation}; control-mode=eef 不可用")
                ep = _read("action/end/position", "end")
                eo = _read("action/end/orientation", "end")
                if ep.shape[1] != 6 or eo.shape[1] != 8:
                    raise ValueError(
                        f"A3 EEF 期望 end/position=6D end/orientation=8D; "
                        f"实际 ep={ep.shape[1]}D eo={eo.shape[1]}D")
                lp, rp = ep[:, :3], ep[:, 3:6]
                lq, rq = eo[:, :4], eo[:, 4:8]
                out["eef_14d"] = np.hstack([lp, rp, lq, rq]).astype(np.float32)

        if "hand" in parts:
            if hand_kind == "gripper":
                if "action/gripper/position" in f:
                    out["gripper"] = _read("action/gripper/position", "gripper")
                else:
                    raise KeyError("hand_kind=gripper 但 H5 没有 action/gripper/position")
            else:
                if "action/hand/position" in f:
                    out["hand_actuator"] = _read("action/hand/position", "hand")
                elif "action/hand/activejointpos" in f:
                    out["hand_radians"] = _read("action/hand/activejointpos", "hand")
                else:
                    raise KeyError(
                        "H5 没有 action/hand/position 也没有 action/hand/activejointpos")

        if "waist" in parts and "action/waist/position" in f:
            wp = _read("action/waist/position", "waist")
            if wp.shape[1] != 4:
                raise ValueError(
                    f"A3 H5 waist 期望 4D [yaw, roll, pitch, height], 实际 {wp.shape[1]}D")
            out["waist_4d"] = wp

        if "head" in parts and "action/head/position" in f:
            out["head"] = _read("action/head/position", "head")

        if "loco" in parts:
            for key, name in (
                ("loco_fwd", "action/velocity/forward_velocity"),
                ("loco_lat", "action/velocity/lateral_velocity"),
                ("loco_ang", "action/velocity/angular_velocity"),
            ):
                if name in f:
                    out[key] = _read(name, "velocity").reshape(-1)
    return out


# ==================== hand 弧度 -> actuator ====================

def hand_radians_to_actuator(hand_rad: np.ndarray, hand_order: str) -> np.ndarray:
    """20D 弧度 -> 20D actuator (固定 left_first: 前 10 左, 后 10 右)."""
    ctrl_left = OmnihandCtrl(hand_type=True)
    ctrl_right = OmnihandCtrl(hand_type=False)
    if hand_order == "left_first":
        ls, rs = slice(0, 10), slice(10, 20)
    else:
        ls, rs = slice(10, 20), slice(0, 10)
    raw = np.zeros_like(hand_rad, dtype=np.float32)
    for i in range(len(hand_rad)):
        raw[i, :10] = ctrl_left.active_joint_pos_to_actuator_input(hand_rad[i, ls].tolist())
        raw[i, 10:] = ctrl_right.active_joint_pos_to_actuator_input(hand_rad[i, rs].tolist())
    return raw


def eef_lp_rp_lq_rq_to_lp_lq_rp_rq(v14: np.ndarray) -> np.ndarray:
    """(N,14) [lp(3), rp(3), lq(4), rq(4)] -> [lp(3), lq(4), rp(3), rq(4)]."""
    out = np.empty_like(v14)
    out[:, 0:3]   = v14[:, 0:3]   # lp
    out[:, 3:7]   = v14[:, 6:10]  # lq
    out[:, 7:10]  = v14[:, 3:6]   # rp
    out[:, 10:14] = v14[:, 10:14] # rq
    return out


# ==================== 保存 records ====================

def save_replay_records(node: ReplayNode, out_path: str) -> Dict[str, int]:
    """把 node.export_records() 写到 npz, 返回每个 part 的帧数信息."""
    rec = node.export_records()
    cmd = rec["cmd"]
    state = rec["state"]
    save_kw: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for part, arr in cmd.items():
        save_kw[f"cmd_{part}"] = arr
        counts[f"cmd_{part}"] = int(len(arr))
    for part, arr in state.items():
        save_kw[f"state_{part}"] = arr
        counts[f"state_{part}"] = int(len(arr))
    if save_kw:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(out_path, **save_kw)
    return counts


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="A3 机器人端 replay (ROS-native, 不依赖 server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
三种模式:
  模式 A (H5, mc passive 透传):
    ros2 run a3_server replay --file-path /path/to/aligned_joints.h5

  模式 B (parquet, server 软件 30->150Hz 插值, mc passive):
    ros2 run a3_server replay --parquet-path /path/to/episode_000000.parquet

  模式 C (parquet, pnc_arm trajectory 机上插值):
    ros2 run a3_server replay --parquet-path /path/to/ep0.parquet --use-onboard-interp
    ros2 run a3_server replay --parquet-path /path/to/ep0.parquet --control-mode eef
""",
    )
    parser.add_argument("--file-path", type=str, default=None,
                        help="H5 文件路径 (与 --parquet-path 二选一)")
    parser.add_argument("--parquet-path", type=str, default=None,
                        help="LeRobot 单 episode parquet 路径 (与 --file-path 二选一)")
    parser.add_argument("--modality-json", type=str, default=None)
    parser.add_argument("--info-json", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default="auto",
                        choices=["auto", "joint", "eef"])
    parser.add_argument("--use-onboard-interp", action="store_true", default=False,
                        help="模式 C: 走 pnc_arm 通道 (parquet 才能用; eef 自动启用)")
    parser.add_argument("--parts", nargs="+", default=["arm", "hand"],
                        choices=["arm", "hand", "waist", "head", "loco"])
    parser.add_argument("--hand-kind", type=str, default="auto",
                        choices=["auto", "hand", "gripper"],
                        help="auto = 订阅 hand_joint_state 拿 frame_id 探测")
    parser.add_argument("--hand-order", type=str, default="left_first",
                        choices=["left_first", "right_first"])
    parser.add_argument("--fps", type=float, default=30.0, help="主循环发送频率")
    parser.add_argument("--skip-mode-switch", action="store_true", default=False,
                        help="跳过自动 pnc_arm 模式切换")
    parser.add_argument("--no-record-state", dest="record_state",
                        action="store_false", default=True,
                        help="不订阅 *_joint_state 录 state (默认录)")
    parser.add_argument("--records-path", type=str, default=REMOTE_RECORDS_PATH,
                        help=f"records.npz 保存路径 (默认 {REMOTE_RECORDS_PATH})")

    args = parser.parse_args()

    if (args.file_path is None) == (args.parquet_path is None):
        parser.error("必须且只能指定 --file-path 或 --parquet-path 之一")

    print(f"[replay_pid] {os.getpid()}", flush=True)

    # 模式判定
    is_h5 = args.file_path is not None
    if is_h5 and args.use_onboard_interp:
        parser.error("H5 数据是逐帧密集轨迹, 不允许 --use-onboard-interp")

    parts = set(args.parts)

    # ---- 启动 rclpy ----
    rclpy.init()

    # ---- 探测 hand_kind ----
    hand_kind = args.hand_kind
    if hand_kind == "auto":
        try:
            hand_kind = detect_hand_kind_via_topic(timeout_sec=5.0)
        except Exception as e:
            print(f"[fatal] {e}")
            rclpy.shutdown()
            sys.exit(1)
    print(f"  hand_kind: {hand_kind}")

    # ---- control-mode ----
    if is_h5:
        control_mode = "joint" if args.control_mode in ("auto", "joint") else "eef"
        if control_mode == "eef":
            rclpy.shutdown()
            parser.error("H5 + control-mode=eef 不支持 (mc 通道无 EEF)")
    else:
        modality_json = (Path(args.modality_json) if args.modality_json
                         else find_modality_json(Path(args.parquet_path)))
        info_json = (Path(args.info_json) if args.info_json
                     else find_info_json(Path(args.parquet_path)))
        if args.control_mode == "auto":
            control_mode = detect_control_mode(modality_json, info_json)
            print(f"  [auto] control-mode={control_mode}")
        else:
            control_mode = args.control_mode

    use_onboard = bool(args.use_onboard_interp) or (control_mode == "eef")
    if is_h5:
        mode_label = "A (H5 raw, passive)"
    else:
        mode_label = ("C (parquet, onboard interp, trajectory)" if use_onboard
                      else "B (parquet, server 软件插值, passive)")

    print("===== A3 Replay (ROS-native) =====")
    print(f"  数据源:      {'H5 ' + args.file_path if is_h5 else 'parquet ' + args.parquet_path}")
    print(f"  模式:        {mode_label}")
    print(f"  control-mode: {control_mode}")
    print(f"  parts:       {sorted(parts)}")
    print(f"  fps:         {args.fps}")
    print(f"  records:     {args.records_path}")

    # ---- 切 pnc_arm 模式 ----
    target_pnc_mode = "trajectory" if use_onboard else "passive"
    if not args.skip_mode_switch:
        print(f"\n切换 pnc_arm -> {target_pnc_mode}")
        ok = set_pnc_arm_mode(target_pnc_mode)
        if not ok:
            print(f"  [警告] pnc_arm 切到 {target_pnc_mode} 失败, 继续执行")
    else:
        print(f"\n[跳过] pnc_arm 模式切换 (--skip-mode-switch); "
              f"当前需为 {target_pnc_mode}")

    # ---- 加载数据 ----
    print("\n加载数据...")
    if is_h5:
        data = load_h5_action(args.file_path, parts, control_mode, hand_kind,
                               fps=float(args.fps))
    else:
        data = load_parquet_action(
            parquet_path=Path(args.parquet_path),
            modality_json_path=modality_json,
            info_json_path=info_json,
            control_mode=control_mode,
            parts=parts,
            hand_kind=hand_kind,
        )

    arm_data = data.get("arm_joint" if control_mode == "joint" else "eef_14d")
    hand_actuator = data.get("hand_actuator")
    hand_radians = data.get("hand_radians")
    gripper_data = data.get("gripper")
    waist_4d = data.get("waist_4d")
    head_data = data.get("head")
    loco_fwd = data.get("loco_fwd")
    loco_lat = data.get("loco_lat")
    loco_ang = data.get("loco_ang")

    if hand_radians is not None and hand_actuator is None:
        hand_actuator = hand_radians_to_actuator(hand_radians, args.hand_order)

    arm_send = arm_data
    if control_mode == "eef" and arm_data is not None and use_onboard:
        arm_send = eef_lp_rp_lq_rq_to_lp_lq_rp_rq(arm_data)

    all_lens = []
    for x in (arm_send, hand_actuator, gripper_data, waist_4d,
              head_data, loco_fwd):
        if x is not None:
            all_lens.append(len(x))
    total = min(all_lens) if all_lens else 0
    print(f"  总帧数: {total}")
    if total == 0:
        print("没有可回放数据, 退出。")
        rclpy.shutdown()
        return

    # ---- 创建 ReplayNode + executor 后台 spin ----
    node = ReplayNode(hand_kind=HandKind(hand_kind), record_state=args.record_state)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 等订阅器初始化
    time.sleep(0.5)

    # ---- 主循环 ----
    print(f"\n开始回放 {total} 帧...")
    interval = 1.0 / args.fps
    arm_flag = 102 if control_mode == "eef" else 2

    try:
        # deadline-based timing: 用绝对节拍, 累积漂移会被自动修正 (个别帧慢的话
        # 下一帧赶上). sleep(interval - elapsed) 每帧 ~1-2ms 调度抖动累积,
        # 1299 帧 fps=60 实测 52.6Hz 就是这么来的。
        t_start = time.perf_counter()
        next_wake = t_start
        for i in range(total):

            # arm
            if arm_send is not None:
                arm_v = arm_send[i]
                if is_h5:                              # 模式 A
                    node.publish_arm_raw(arm_v)
                elif use_onboard:                       # 模式 C
                    node.publish_arm_interpolate(arm_v, flag=arm_flag)
                else:                                   # 模式 B
                    node.publish_arm_target(arm_v)

            # hand / gripper
            if hand_kind == "gripper" and gripper_data is not None:
                cmd = [int(round(x)) for x in gripper_data[i]]
                if is_h5:
                    node.publish_hand_raw(cmd)
                else:
                    node.publish_hand_target(cmd)
            if hand_kind == "hand" and hand_actuator is not None:
                cmd = [int(round(x)) for x in hand_actuator[i]]
                if is_h5:
                    node.publish_hand_raw(cmd)
                else:
                    node.publish_hand_target(cmd)

            # waist
            if waist_4d is not None:
                yaw, roll, pitch, height = (float(x) for x in waist_4d[i])
                node.publish_waist(yaw=yaw, roll=roll, pitch=pitch, height=height)

            # head
            if head_data is not None:
                node.publish_head(float(head_data[i][0]), float(head_data[i][1]))

            # loco
            if loco_fwd is not None and loco_lat is not None and loco_ang is not None:
                node.publish_loco(float(loco_fwd[i]), float(loco_lat[i]),
                                   float(loco_ang[i]))

            # deadline-based sleep: 节拍跟绝对时钟对齐, 不会累积抖动
            next_wake += interval
            now = time.perf_counter()
            sleep_for = next_wake - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -interval:
                # 单帧落后超过一个 interval, 重置基准 (典型: gc 暂停)
                next_wake = now

            if (i + 1) % 30 == 0 or i == total - 1:
                elapsed_now = time.perf_counter() - t_start
                actual_fps = (i + 1) / elapsed_now if elapsed_now > 0 else 0.0
                print(f"[progress] {i + 1}/{total} fps={actual_fps:.1f}", flush=True)

        elapsed_total = time.perf_counter() - t_start
        print(f"\n回放完成: {total} 帧 / {elapsed_total:.2f}s "
              f"({total / max(elapsed_total, 1e-6):.1f} Hz 实测)")

        # 给最后一条 state 一点时间被 callback 抓到
        time.sleep(0.3)

        # ---- 保存 records ----
        try:
            counts = save_replay_records(node, args.records_path)
            print(f"\n[records] 已保存到 {args.records_path}")
            for k, n in sorted(counts.items()):
                print(f"  {k}: {n}")
        except Exception as e:
            print(f"\n[records] 保存失败: {e}")
    finally:
        # 先停 InterpolationPublisher 的 150Hz 后台线程, 再关 rclpy. 否则
        # _interp_loop 会在 rclpy.shutdown 后继续 publish 引发异常 trace.
        try:
            node.interp_pub.stop()
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
