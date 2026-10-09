"""A3 reset_pose — 慢插值复位到初始姿态.

跟 replay_ui.py::_do_reset 等价 (之前那个走 server HTTP), 这里 ROS-native:
  - 订阅 *_joint_state 拿当前 arm/hand state 作为起点
  - 自动按 frame_id 探测 hand_kind (hand=O10 / gripper=AgiClaw)
  - 终点: arm = ARM_RESET_14D (用户给的 14D 弧度)
          hand = VLA_HAND_TEST_OPEN_POS (5 指张开) / gripper = [0, 0] (闭合)
  - 主循环 30Hz × 3s 线性插值, 走 ReplayNode 的 publish_arm_target /
    publish_hand_target (通过 InterpolationPublisher 再叠 5 帧 33ms 平滑)

用法:
  ros2 run a3_server reset_pose
  ros2 run a3_server reset_pose --duration 5.0 --fps 30
  ros2 run a3_server reset_pose --hand-kind hand    # 跳过自动探测
"""

import argparse
import sys
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor

from a3_server.joint_config import HandKind
from a3_server.replay import detect_hand_kind_via_topic
from a3_server.replay_node import ReplayNode


# arm 复位姿态 (14D 弧度, 双臂; 用户实测值, 跟 replay_ui.ARM_RESET_14D 一致)
ARM_RESET_14D = [
    -0.19527023223876938, -0.11566303466796857,  0.0018544795227048994,
     1.2152437594604493,   0.0146250126647951,   0.03240739837646478,
    -0.0031095697021483737,
     0.19465988067626938, -0.12634418701171857, -0.013756811828613102,
     1.2311129000854493,  -0.010962426452636898, 0.03210222259521478,
     0.0031095697021483737,
]
GRIPPER_RESET_2D = [0, 0]


def _get_hand_open_pos() -> list:
    """从 RoboInterface/config.py 拿 VLA_HAND_TEST_OPEN_POS (5 指全伸直 20D actuator)."""
    # 工具脚本部署在 ADU 上, RoboInterface/config.py 不一定在 PYTHONPATH。
    # 启动时尝试几个常见位置, 都没有就硬编码一份 fallback。
    import importlib.util
    candidates = [
        "/agibot/RoboInterface/config.py",
        "/agibot/robotinterface/RoboInterface/config.py",
    ]
    for p in candidates:
        try:
            spec = importlib.util.spec_from_file_location("ri_config", p)
            if spec is None: continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "VLA_HAND_TEST_OPEN_POS"):
                return list(mod.VLA_HAND_TEST_OPEN_POS)
        except Exception:
            continue
    # Fallback: 直接硬编码 (跟 RoboInterface/config.py 同值)
    return [
        2780.0, 1942.0, 3954.0, 3927.0, 3995.0, 3995.0, 4031.0, 3995.0,   17.0, 3995.0,
        2784.0, 2044.0, 3933.0,    0.0, 3995.0, 3995.0,  512.0, 3995.0, 3909.0, 3995.0,
    ]


def _wait_initial_state(node: ReplayNode, timeout_sec: float = 3.0):
    """等 state subscriber 拿到至少一条 arm + hand 反馈, 才能用作插值起点."""
    end = time.time() + timeout_sec
    while time.time() < end:
        with node._lock:
            arm_ok = bool(node.state_records.get("arm"))
            hand_ok = bool(node.state_records.get("hand"))
        if arm_ok and hand_ok:
            return True
        time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="A3 慢插值复位到初始姿态 (ROS-native, 不依赖 server)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="复位过渡总秒数 (默认 3.0s)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="主循环 publish 频率 (默认 30Hz)")
    parser.add_argument("--hand-kind", type=str, default="auto",
                        choices=["auto", "hand", "gripper"])
    args = parser.parse_args()

    rclpy.init()

    # 探测 hand_kind
    hand_kind = args.hand_kind
    if hand_kind == "auto":
        try:
            hand_kind = detect_hand_kind_via_topic(timeout_sec=5.0)
        except Exception as e:
            print(f"[fatal] {e}")
            rclpy.shutdown()
            sys.exit(1)
    print(f"  hand_kind: {hand_kind}")

    # 创建 ReplayNode (复用 publisher / subscriber)
    node = ReplayNode(hand_kind=HandKind(hand_kind), record_state=True)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("等待初始 state ...")
    if not _wait_initial_state(node, timeout_sec=3.0):
        print("[warn] 没拿到 arm/hand state, 仍尝试用零起点插值 (可能突变)")

    with node._lock:
        cur_arm = (list(node.state_records["arm"][-1][1])
                   if node.state_records["arm"] else [0.0] * 14)
        cur_hand = (list(node.state_records["hand"][-1][1])
                    if node.state_records["hand"] else [0.0] * (2 if hand_kind == "gripper" else 20))

    target_arm = np.asarray(ARM_RESET_14D, dtype=np.float64)
    if hand_kind == "hand":
        target_hand = np.asarray(_get_hand_open_pos(), dtype=np.float64)
    else:
        target_hand = np.asarray(GRIPPER_RESET_2D, dtype=np.float64)
    cur_arm_arr = np.asarray(cur_arm, dtype=np.float64)
    cur_hand_arr = np.asarray(cur_hand, dtype=np.float64)
    if len(cur_hand_arr) != len(target_hand):
        print(f"[fatal] hand 维度不匹配: cur={len(cur_hand_arr)} target={len(target_hand)}")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    n_steps = max(2, int(round(args.duration * args.fps)))
    interval = 1.0 / args.fps
    print(f"复位中: {args.duration:.1f}s × {args.fps:.0f}Hz = {n_steps} 帧 ...")

    try:
        for i in range(1, n_steps + 1):
            t0 = time.time()
            alpha = i / n_steps
            arm_v = (1 - alpha) * cur_arm_arr + alpha * target_arm
            hand_v = (1 - alpha) * cur_hand_arr + alpha * target_hand
            node.publish_arm_target([float(x) for x in arm_v])
            node.publish_hand_target([int(round(x)) for x in hand_v])
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
        print(f"复位完成 (arm 14D + {'hand 张开' if hand_kind == 'hand' else 'gripper 闭合'})")
    finally:
        try:
            node.interp_pub.stop()  # 先停 150Hz interp 后台线程再 shutdown
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
