"""ROS 节点 (a3_server.replay 用) — 自己 publish + subscribe, 不依赖 server :5050.

负责:
  - 订阅 5 路 *_joint_state, callback 里直接 append (t_ns, position) 到 state_records,
    频率 = mc 反馈频率 (~150Hz, 比之前 30Hz HTTP 拉密).
  - 创建 publishers (复用 InterpolationPublisher + 加 protobuf 通道 waist/loco/pnc_arm),
    publish 时同步 append (t_ns, values) 到 cmd_records.

时间戳全用 ROS clock (node.get_clock().now().nanoseconds), 没有 HTTP 延迟。

跟 server_node.A3ServerNode 共用 InterpolationPublisher + 同样的 protobuf channel
路径; 但不订阅相机 / TA whole_body / 不开 Flask, 只为 replay 服务。
"""

from datetime import datetime
import threading
from typing import Dict, List, Optional, Tuple

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
    QoSDurabilityPolicy, QoSLivelinessPolicy,
)
from sensor_msgs.msg import JointState

from a3_server.interp_publisher import InterpolationPublisher
from a3_server.joint_config import HandKind, LEG_JOINT_NAMES


def _now_ns(node: Node) -> int:
    return node.get_clock().now().nanoseconds


def _create_qos(depth: int = 1) -> QoSProfile:
    """跟 server_node._create_qos 同款."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        liveliness=QoSLivelinessPolicy.AUTOMATIC,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _fill_pb_header(header_msg) -> None:
    """填 aimdk.protocol.Header (timestamp + frame_id="user_McScript")."""
    now = datetime.utcnow()
    secs = int(now.timestamp())
    nanos = now.microsecond * 1000
    ts = header_msg.timestamp
    if hasattr(ts, "seconds"):
        ts.seconds = secs
        ts.nanos = nanos
    else:
        ts.sec = secs
        ts.nsec = nanos
    if hasattr(header_msg, "control_source"):
        try: header_msg.control_source = 2  # SAFE
        except Exception: pass
    if hasattr(header_msg, "frame_id"):
        try: header_msg.frame_id = "user_McScript"
        except Exception: pass


def _try_load_pb(paths, label: str, logger):
    """按优先顺序尝试 import (mod, name), 返回 class 或 None."""
    for mod_name, cls_name in paths:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except (ImportError, AttributeError):
            continue
    if logger is not None:
        logger.warn(f"无法加载 {label} pb2 (尝试过: {paths})")
    return None


class ReplayNode(Node):
    """A3 replay 用的 ROS 节点 — 自己 publish + 订阅 state."""

    def __init__(self, hand_kind: HandKind, record_state: bool = True):
        super().__init__("a3_replay")
        self.hand_kind = HandKind(hand_kind)
        self.record_state = record_state

        self._lock = threading.Lock()
        # state_records[part] = [(t_ns, [position floats]), ...]
        self.state_records: Dict[str, List[Tuple[int, list]]] = {
            "arm": [], "hand": [], "waist": [], "head": [], "leg": [],
        }
        # cmd_records[part] = [(t_ns, [values]), ...]
        # part: arm / hand / waist / head / loco
        self.cmd_records: Dict[str, List[Tuple[int, list]]] = {
            "arm": [], "hand": [], "waist": [], "head": [], "loco": [],
        }

        qos = _create_qos()

        # ==================== state subscribers ====================
        if record_state:
            self.create_subscription(JointState, "/motion/control/arm_joint_state",
                                      self._arm_state_cb, qos_profile=qos)
            self.create_subscription(JointState, "/motion/control/hand_joint_state",
                                      self._hand_state_cb, qos_profile=qos)
            self.create_subscription(JointState, "/motion/control/waist_joint_state",
                                      self._waist_state_cb, qos_profile=qos)
            self.create_subscription(JointState, "/motion/control/neck_joint_state",
                                      self._head_state_cb, qos_profile=qos)
            try:
                from joint_msgs.msg import JointState as LegJointState  # type: ignore
                _leg_state_msg = LegJointState
            except ImportError:
                _leg_state_msg = JointState
            self.create_subscription(_leg_state_msg, "/motion/control/leg_joint_state",
                                      self._leg_state_cb, qos_profile=qos)

        # ==================== publishers (mc 通道, 走 InterpolationPublisher) ====================
        self.interp_pub = InterpolationPublisher(
            self, target_fps=150.0, interp_steps=5, hand_kind=self.hand_kind)

        # head publisher (neck topic)
        self._neck_pub = self.create_publisher(
            JointState, "/motion/control/neck_joint_command", qos_profile=qos)

        # leg publisher (replay 通常不发 leg, 但留接口)
        try:
            from joint_msgs.msg import JointCommand as LegJointCommand  # type: ignore
            _leg_cmd_msg = LegJointCommand
        except ImportError:
            _leg_cmd_msg = JointState
        self._leg_pub = self.create_publisher(
            _leg_cmd_msg, "/body_drive/leg_joint_command_ros2", qos_profile=qos)

        # ==================== protobuf 通道: waist / loco / pnc_arm interp ====================
        self._RosMsgWrapper = None
        self._WaistChannel = None
        self._LocoChannel = None
        self._PncArmInterpChannel = None
        try:
            from ros2_plugin_proto.msg import RosMsgWrapper  # type: ignore
            self._RosMsgWrapper = RosMsgWrapper
        except ImportError as e:
            self.get_logger().error(f"ros2_plugin_proto 未装, protobuf 通道不可用: {e}")

        log = self.get_logger()
        self._WaistChannel = _try_load_pb(
            paths=[
                ("aimdk.protocol.motion_control.motion.mc_motion_channel_pb2",
                 "MotionControlMoveWaistChannel"),
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "MotionControlMoveWaistChannel"),
                ("aimdk.protocol_pb2", "MotionControlMoveWaistChannel"),
            ], label="腰部", logger=log)
        self._LocoChannel = _try_load_pb(
            paths=[
                ("aimdk.protocol.motion_control.motion.mc_motion_channel_pb2",
                 "MotionControlLocomotionVelocityChannel"),
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "MotionControlLocomotionVelocityChannel"),
                ("aimdk.protocol_pb2", "MotionControlLocomotionVelocityChannel"),
            ], label="行走", logger=log)
        self._PncArmInterpChannel = _try_load_pb(
            paths=[
                ("aimdk.protocol.pnc_arm.pnc_arm_channel_pb2",
                 "PncArmInterpolateChannel"),
            ], label="PncArm 插值", logger=log)

        self._waist_pub = None
        self._loco_pub = None
        self._pnc_arm_interp_pub = None
        if self._RosMsgWrapper is not None:
            if self._WaistChannel is not None:
                self._waist_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel",
                    qos_profile=qos)
            if self._LocoChannel is not None:
                self._loco_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/motion/control/locomotion_velocity/pb_3Aaimdk_2Eprotocol_2EMotionControlLocomotionVelocityChannel",
                    qos_profile=qos)
            if self._PncArmInterpChannel is not None:
                self._pnc_arm_interp_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/pnc_arm/motion/interpolate/pb_3Aaimdk_2Eprotocol_2EPncArmInterpolateChannel",
                    qos_profile=qos)

        self.get_logger().info(
            f"ReplayNode 初始化完成 (hand_kind={self.hand_kind.value}, record_state={record_state})")

    # ==================== state callbacks ====================
    # 频率 = mc 发布频率 (~150Hz), 直接每条都 append, 比 30Hz HTTP 拉更密。
    def _arm_state_cb(self, msg):
        with self._lock:
            self.state_records["arm"].append((_now_ns(self), list(msg.position)))

    def _hand_state_cb(self, msg):
        with self._lock:
            self.state_records["hand"].append((_now_ns(self), list(msg.position)))

    def _waist_state_cb(self, msg):
        with self._lock:
            self.state_records["waist"].append((_now_ns(self), list(msg.position)))

    def _head_state_cb(self, msg):
        with self._lock:
            self.state_records["head"].append((_now_ns(self), list(msg.position)))

    def _leg_state_cb(self, msg):
        with self._lock:
            self.state_records["leg"].append((_now_ns(self), list(msg.position)))

    # ==================== publishers (cmd) ====================
    # 每个 publish_* 都同步 append cmd_records (t_ns + values) — 时间戳来自 publish
    # 之前一刻的 ROS clock。

    def publish_arm_raw(self, values) -> None:
        """模式 A: 直接 publish 到 mc 通道, 不插值."""
        self.interp_pub.publish_arm_raw(values)
        with self._lock:
            self.cmd_records["arm"].append((_now_ns(self),
                                              [float(x) for x in values]))

    def publish_arm_target(self, values) -> None:
        """模式 B: server 软件 30→150Hz 插值."""
        self.interp_pub.set_arm_target(values)
        with self._lock:
            self.cmd_records["arm"].append((_now_ns(self),
                                              [float(x) for x in values]))

    def publish_arm_interpolate(self, positions, flag: int = 2) -> bool:
        """模式 C: pnc_arm trajectory 通道 (机上插值).

        positions 14D (joint) / 14D (SE3) / 17D (含腰前 3 维占位).
        flag: 0/1/2 = joint 左/右/双; 100/101/102 = SE3 左/右/双.
        """
        if self._pnc_arm_interp_pub is None or self._PncArmInterpChannel is None:
            self.get_logger().warn("PncArm 插值通道未初始化")
            return False
        try:
            pos = [float(x) for x in positions]
            if len(pos) == 14:
                pos = [0.0, 0.0, 0.0] + pos
            n = len(pos)
            ch = self._PncArmInterpChannel()
            _fill_pb_header(ch.header)
            ch.flag = int(flag)
            ch.positions[:] = pos
            ch.velocities[:] = [0.0] * n
            ch.accelerations[:] = [0.0] * n
            ch.effort[:] = [0.0] * n
            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._pnc_arm_interp_pub.publish(wrapper)
            with self._lock:
                # cmd_records 记原始 14D (不含腰占位), 跟主机端 cmd_data 对齐
                stored = pos[3:] if len(pos) == 17 and len(positions) == 14 else list(positions)
                self.cmd_records["arm"].append(
                    (_now_ns(self), [float(x) for x in stored]))
            return True
        except Exception as e:
            self.get_logger().error(f"PncArm 插值发布失败: {e}")
            return False

    def publish_hand_raw(self, values) -> None:
        self.interp_pub.publish_hand_raw(values)
        with self._lock:
            self.cmd_records["hand"].append((_now_ns(self),
                                               [float(x) for x in values]))

    def publish_hand_target(self, values) -> None:
        self.interp_pub.set_hand_target(values)
        with self._lock:
            self.cmd_records["hand"].append((_now_ns(self),
                                               [float(x) for x in values]))

    def publish_waist(self, *, pitch: float = 0.0, roll: float = 0.0,
                       yaw: float = 0.0, height: float = 0.0) -> bool:
        """通过 protobuf MotionControlMoveWaistChannel 发布."""
        if self._waist_pub is None or self._WaistChannel is None:
            self.get_logger().warn("腰部 publisher 未初始化")
            return False
        try:
            ch = self._WaistChannel()
            _fill_pb_header(ch.header)
            ch.waist_pitch = float(pitch)
            ch.waist_roll = float(roll)
            ch.waist_yaw = float(yaw)
            ch.waist_height = float(height)
            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._waist_pub.publish(wrapper)
            with self._lock:
                # 存 [yaw, roll, pitch, height] 跟 H5/parquet waist_4d 一致
                self.cmd_records["waist"].append(
                    (_now_ns(self), [float(yaw), float(roll), float(pitch), float(height)]))
            return True
        except Exception as e:
            self.get_logger().error(f"腰部发布失败: {e}")
            return False

    def publish_head(self, shake: float, nod: float) -> bool:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "user_McScript"
        msg.name = ["head_yaw_joint", "head_pitch_joint"]
        msg.position = [float(shake), float(nod)]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        self._neck_pub.publish(msg)
        with self._lock:
            self.cmd_records["head"].append(
                (_now_ns(self), [float(shake), float(nod)]))
        return True

    def publish_loco(self, forward: float, lateral: float, angular: float,
                      mode: int = 0) -> bool:
        if self._loco_pub is None or self._LocoChannel is None:
            self.get_logger().warn("行走 publisher 未初始化")
            return False
        try:
            ch = self._LocoChannel()
            _fill_pb_header(ch.header)
            ch.data.mode = int(mode)
            ch.data.forward_velocity = float(forward)
            ch.data.lateral_velocity = float(lateral)
            ch.data.angular_velocity = float(angular)
            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._loco_pub.publish(wrapper)
            with self._lock:
                self.cmd_records["loco"].append(
                    (_now_ns(self), [float(forward), float(lateral), float(angular)]))
            return True
        except Exception as e:
            self.get_logger().error(f"行走发布失败: {e}")
            return False

    # ==================== 工具: 导出 records ====================

    def export_records(self) -> Dict[str, Dict[str, "np.ndarray"]]:
        """导出 cmd / state records 成 ndarray 字典 (供保存 npz / 画图):

        返回:
            {
              "cmd":   {arm: (Nc,Da+1), hand: ..., waist: ..., head: ..., loco: ...},
              "state": {arm: (Ns,Da+1), hand: ..., waist: ..., head: ..., leg: ...},
            }
        每条记录第 0 列是 t_ns (相对于 t_start, 秒), 后面是 values。
        """
        import numpy as np
        with self._lock:
            cmd_snap = {k: list(v) for k, v in self.cmd_records.items()}
            state_snap = {k: list(v) for k, v in self.state_records.items()}

        # 找 t_start: 取所有记录里最早的 t_ns
        all_t = [r[0] for k in cmd_snap for r in cmd_snap[k]]
        all_t += [r[0] for k in state_snap for r in state_snap[k]]
        t_start = min(all_t) if all_t else 0

        def _pack(records: Dict[str, list]) -> Dict[str, "np.ndarray"]:
            out = {}
            for part, rs in records.items():
                if not rs:
                    continue
                D = len(rs[0][1])
                arr = np.zeros((len(rs), D + 1), dtype=np.float32)
                for i, (t_ns, vals) in enumerate(rs):
                    arr[i, 0] = (t_ns - t_start) / 1e9  # 秒, 相对 t_start
                    arr[i, 1:1 + len(vals)] = vals
                out[part] = arr
            return out

        return {"cmd": _pack(cmd_snap), "state": _pack(state_snap),
                 "t_start_ns": t_start}
