#!/usr/bin/env python3
"""
A3 机器人 ROS2 服务节点 (运行在机器人上)。

参考 a2_server.server_node 实现，区别：
  - 不订阅相机：A3 相机走 foxglove_msgs/CompressedVideo (h265)，由上层按需自取，
    不在 server 内拉，避免依赖 PyAV/ffmpeg。
  - 协议层 import 改为 aima_protocol-main 生成的 aimdk.protocol.* 包路径
    （新 layout，按 .proto 子目录拆分），不再用旧的单一 aimdk.protocol_pb2。
  - 新增订阅 /ta/whole_body_command/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel，
    并通过 HTTP /get_ta_whole_body_command 暴露解析后的 dict，方便客户端查询。
  - 末端按 ``--hand-kind hand|gripper`` 切换 O10 灵巧手 / AgiClaw 双指夹爪。

A3U 实测 topic (ros2 topic info):
  arm   cmd: /motion/control/pnc/arm_joint_command  joint_msgs/JointCommand
  hand  cmd: /motion/control/hand_joint_command  sensor_msgs/JointState
  neck  cmd: /motion/control/neck_joint_command  sensor_msgs/JointState
  waist cmd: /motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel
                                                  ros2_plugin_proto/RosMsgWrapper
  leg   cmd: /body_drive/leg_joint_command_ros2  sensor_msgs/JointState
  loco  cmd: /motion/control/locomotion_velocity/pb_3Aaimdk_2Eprotocol_2EMotionControlLocomotionVelocityChannel

构建:
  cd /agibot/ros_server && colcon build
启动:
  source install/setup.bash && ros2 run a3_server server -- --hand-kind hand
"""

import argparse
import time
import io
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
    QoSDurabilityPolicy, QoSLivelinessPolicy,
)
from sensor_msgs.msg import JointState, Image, Imu
from geometry_msgs.msg import PoseArray
import tf2_ros
from cv_bridge import CvBridge
from flask import Flask, jsonify, request, send_file

from a3_server.joint_config import (
    LEG_JOINT_NAMES, WAIST_JOINT_NAMES, HandKind, hand_dim,
)
from a3_server.interp_publisher import InterpolationPublisher


# ==================== A3 相机配置 ====================
# A3 原始流是 /hal/<name>/stream (foxglove_msgs/CompressedVideo, h265)。
# 把 /stream 替换为 /rgb 后, ros2 topic info 显示是 sensor_msgs/msg/Image
# (未压缩, 一般是 bgr8/rgb8), server 端 cv_bridge 转成 BGR ndarray, HTTP
# 用 JPEG 编码下发, 与 A2 的 /get_*_rgb 接口同形。
CAMERA_TOPICS = {
    "head_stereo_left":  "/hal/head_stereo_left_fisheye_camera/rgb",
    "head_stereo_right": "/hal/head_stereo_right_fisheye_camera/rgb",
    "head_left":         "/hal/head_left_fisheye_camera/rgb",
    "head_right":        "/hal/head_right_fisheye_camera/rgb",
    "head_rear":         "/hal/head_rear_fisheye_camera/rgb",
    "chest_front":       "/hal/chest_front_d457_camera/rgb",
    "waist_front":       "/hal/waist_front_d415_camera/rgb",
    "wrist_left":        "/hal/wrist_left_d405_camera/rgb",
    "wrist_right":       "/hal/wrist_right_d405_camera/rgb",
    "armpit_right":      "/hal/armpit_right_fisheye_camera/rgb",
}


# ==================== 工具 ====================

def _create_qos(depth=1):
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        liveliness=QoSLivelinessPolicy.AUTOMATIC,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _fill_pb_header(header_msg):
    """填充 aimdk.protocol.Header 的 timestamp + control_source。

    aima_protocol-main 的 Header 用 aimdk.protocol.Timestamp（自定义 message,
    带 sec/nsec 字段），不是 google.protobuf.Timestamp。

    control_source 用 SAFE (=2), 与 T_LocomotionVelocity / T_WaistMove 等
    用户脚本一致；MANUAL (=1) 部分版本会被 mc 模块拒。
    """
    now = datetime.utcnow()
    secs = int(now.timestamp())
    nanos = now.microsecond * 1000
    ts = header_msg.timestamp
    # 兼容两种字段命名（旧: seconds/nanos，新: sec/nsec）
    if hasattr(ts, "seconds"):
        ts.seconds = secs
        ts.nanos = nanos
    else:
        ts.sec = secs
        ts.nsec = nanos
    if hasattr(header_msg, "control_source"):
        # ControlSource_SAFE = 2
        try:
            header_msg.control_source = 2
        except Exception:
            pass
    if hasattr(header_msg, "frame_id"):
        try:
            header_msg.frame_id = "user_McScript"
        except Exception:
            pass


from google.protobuf.json_format import MessageToDict


def _msg_to_dict(pb_msg):
    """protobuf msg -> dict（保留默认值，repeated/scalar 都展开）。

    用 MessageToDict 而不是手写遍历，避免漏字段；客户端拿到的就是
    .proto 中字段名一一对应的 nested dict。

    `always_print_fields_with_no_presence` 是 protobuf 4.25+ 才有；
    老版本 fallback 到 `including_default_value_fields`。
    """
    try:
        return MessageToDict(
            pb_msg,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        return MessageToDict(
            pb_msg,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
            including_default_value_fields=True,
        )


# ==================== ROS2 节点 ====================

class A3ServerNode(Node):

    def __init__(self, node_name="a3_server", robot_ip="192.168.100.100",
                 hand_kind: HandKind = HandKind.HAND):
        super().__init__(node_name)
        self.robot_ip = robot_ip
        self.hand_kind = HandKind(hand_kind)
        self.cv_bridge = CvBridge()

        # ==================== 状态缓存 ====================
        self.latest_neck_joints = None
        self.latest_arm_joints = None
        self.latest_hand_joints = None
        self.latest_waist_joints = None
        self.latest_leg_joints = None
        self.latest_eef_pose = None

        # 相机：每路缓存最近一帧 BGR ndarray
        self.latest_cameras = {name: None for name in CAMERA_TOPICS}
        self._cam_lock = threading.Lock()

        # TA 全身指令（A3 独有）
        self.latest_ta_whole_body = None       # MessageToDict 后的 dict
        self.latest_ta_whole_body_ts = None    # wall time
        self.latest_ta_whole_body_raw_len = 0  # 原始字节长度
        self._ta_lock = threading.Lock()

        # IMU cache — 订阅 /body_drive/{pelvis,torso}_imu/data (sensor_msgs/Imu).
        # 用于 sonic_a3 modality 里的 pelvis_orient6d / pelvis_gravity state。
        # 每颗 IMU 存: {"orientation_xyzw": [4], "angular_velocity": [3],
        #              "linear_acceleration": [3], "gravity_dir": [3], "timestamp": <float>}
        # gravity_dir 由 orientation 反旋 world-frame [0,0,-1] 到 IMU frame 得到,
        # 训练侧 sonic_a3_full modality.json 的 pelvis_gravity 就是这个。
        self.latest_pelvis_imu = None
        self.latest_torso_imu = None
        self._imu_lock = threading.Lock()

        # ==================== WBC whole-body state cache ====================
        # 第二条 state 链路 (whole_body 模式专用)。与上面分散的 latest_* 缓存
        # (upper_body 链路 + 手) 完全独立: whole_body 部署时客户端调
        # /set_state_source {"source":"whole_body"}, server 才懒创建下面的
        # /wbc/whole_body_state 订阅, 把 leg/waist/head/arm + pelvis/torso IMU
        # 填进这组 latest_wb_* 缓存。手不在此链路 (TaWholeBodyState 不含手),
        # whole_body 用手时仍从分散的 latest_hand_joints 取。upper_body 模式下
        # 不会创建该订阅, 这组缓存保持 None, 互不干扰。
        self.latest_wb_leg_joints = None
        self.latest_wb_waist_joints = None
        self.latest_wb_neck_joints = None   # 由 TaWholeBodyState.head_state 填
        self.latest_wb_arm_joints = None
        self.latest_wb_pelvis_imu = None
        self.latest_wb_torso_imu = None
        self._wb_state_lock = threading.Lock()   # 保护 latest_wb_* 缓存
        self._wb_sub_lock = threading.Lock()     # 保护 _wb_state_sub 懒创建
        self._wb_state_sub = None                # 懒创建的订阅句柄 (None=未启用)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = _create_qos()

        # ==================== 订阅: 相机 (sensor_msgs/Image) ====================
        for name, topic in CAMERA_TOPICS.items():
            # 默认参数捕获 name, 否则闭包都引用循环末值
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._camera_cb(n, msg),
                qos_profile=qos)
        self.get_logger().info(
            f"已订阅 {len(CAMERA_TOPICS)} 路 A3 相机 (sensor_msgs/Image)")

        # ==================== 订阅: 关节 ====================
        self.create_subscription(JointState, "/motion/control/arm_joint_state",
                                 self._arm_cb, qos_profile=qos)
        self.create_subscription(JointState, "/motion/control/hand_joint_state",
                                 self._hand_cb, qos_profile=qos)
        self.create_subscription(PoseArray, "/motion_control/hand_pose_state",
                                 self._eef_pose_cb, qos_profile=qos)

        # ==================== 订阅 & 发布: 头部 (ROS Topic) ====================
        self.create_subscription(JointState, "/motion/control/neck_joint_state",
                                 self._neck_cb, qos_profile=qos)
        self._neck_pub = self.create_publisher(
            JointState, "/motion/control/neck_joint_command", qos_profile=qos)

        # ==================== 订阅: IMU (sensor_msgs/Imu) ====================
        # sonic_a3 用两颗 IMU: pelvis + torso. VLA state 里 pelvis_orient6d /
        # pelvis_gravity 都从这里取。
        self.create_subscription(Imu, "/body_drive/pelvis_imu/data",
                                 lambda m: self._imu_cb("pelvis", m),
                                 qos_profile=qos)
        self.create_subscription(Imu, "/body_drive/torso_imu/data",
                                 lambda m: self._imu_cb("torso", m),
                                 qos_profile=qos)

        # ==================== 订阅: 腰部 state (sensor_msgs/JointState) ====================
        # A3U 实测 /motion/control/waist_joint_state 是 sensor_msgs/JointState,
        # 顺序 [waist_yaw, waist_roll, waist_pitch]; 控制走下面 protobuf 通道。
        self.create_subscription(JointState, "/motion/control/waist_joint_state",
                                 self._waist_cb, qos_profile=qos)

        # ==================== 订阅 & 发布: 腿部 ====================
        # state: /motion/control/leg_joint_state (sensor_msgs/JointState)
        # cmd:   /body_drive/leg_joint_command_ros2 (sensor_msgs/JointState)
        #        — 当前固件版本可能尚未启用; 老固件用 /motion/control/leg_joint_command
        self.create_subscription(JointState, "/motion/control/leg_joint_state",
                                 self._leg_cb, qos_profile=qos)
        self._leg_pub = self.create_publisher(
            JointState, "/body_drive/leg_joint_command_ros2", qos_profile=qos)

        # ==================== protobuf 通道：行走 / 腰部 / TA whole_body ====================
        self._MotionControlLocomotionVelocityChannel = None
        self._MotionControlMoveWaistChannel = None
        self._TaWholeBodyCommandChannel = None
        self._RosMsgWrapper = None

        try:
            from ros2_plugin_proto.msg import RosMsgWrapper
            self._RosMsgWrapper = RosMsgWrapper
        except ImportError as e:
            self.get_logger().error(
                f"ros2_plugin_proto 未安装，所有 protobuf topic 都不可用: {e}")

        # 行走 (MotionControlLocomotionVelocityChannel) — 新版命名
        # 优先找新 AIMA layout (motion_control/motion/), 它才是机器人 mc 守护进程
        # 实际订阅的协议; 老的 mc/motion/ layout 作为 fallback。
        self._MotionControlLocomotionVelocityChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.motion_control.motion.mc_motion_channel_pb2",
                 "MotionControlLocomotionVelocityChannel"),
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "MotionControlLocomotionVelocityChannel"),
                ("aimdk.protocol_pb2", "MotionControlLocomotionVelocityChannel"),
                # 老 a2 命名兼容
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "McLocomotionVelocityChannel"),
                ("aimdk.protocol_pb2", "McLocomotionVelocityChannel"),
            ],
            label="行走",
        )

        # 腰部 (MotionControlMoveWaistChannel) — 新 AIMA layout 字段是扁平的
        # (waist_pitch/waist_roll/waist_yaw/waist_height), 老 mc/motion/ layout
        # 是 McMoveWaistChannel 嵌套 data, 二者 wire 不兼容 — 必须先找新的。
        self._MotionControlMoveWaistChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.motion_control.motion.mc_motion_channel_pb2",
                 "MotionControlMoveWaistChannel"),
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "MotionControlMoveWaistChannel"),
                ("aimdk.protocol_pb2", "MotionControlMoveWaistChannel"),
                ("aimdk.protocol.mc.motion.mc_motion_channel_pb2",
                 "McMoveWaistChannel"),
                ("aimdk.protocol_pb2", "McMoveWaistChannel"),
            ],
            label="腰部",
        )

        # TA whole_body_command (A3 独有，aima_protocol-main 才有)
        self._TaWholeBodyCommandChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.ta.ta_channel_pb2", "TaWholeBodyCommandChannel"),
            ],
            label="TA 全身指令",
        )

        # GR00T 10-frame reference window (whole_body sonic 侧订阅)
        # gr00t sonic-a3 部署走这条: 每次 /send_chunk 收到 whole-body chunk 后,
        # 后台 timer 以 REF_WINDOW_PUBLISH_HZ 频率从当前 played_idx 起截 10 帧
        # 组 TaWholeBodyReferenceWindow, 发到 wire topic。
        # 该消息已从 ta_channel.proto 迁到 wbc_reference_window.proto (避免与
        # upstream ta_channel 符号重复); 保留 ta_channel_pb2 作旧版 fallback。
        self._TaWholeBodyReferenceWindow = self._try_load_pb(
            paths=[
                ("aimdk.protocol.ta.wbc_reference_window_pb2",
                 "TaWholeBodyReferenceWindow"),
                ("aimdk.protocol.ta.ta_channel_pb2", "TaWholeBodyReferenceWindow"),
            ],
            label="TA 10-frame reference window",
        )
        # WBC 全身 state 通道 (whole_body 链路)。订阅在 /set_state_source
        # {"source":"whole_body"} 时懒创建 (enable_wb_state_source), 解析后填
        # latest_wb_* 缓存, 经 HTTP /get_whole_body_state 暴露。
        self._TaWholeBodyStateChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.ta.ta_whole_body_state_pb2",
                 "TaWholeBodyStateChannel"),
            ],
            label="WBC 全身 state 通道",
        )
        # 单帧 TaWholeBodyCommand 也要能构造 (window.frames.add() 返回它)
        # 依赖同一份 pb2 模块的 TaWholeBodyCommand class + TaJointLayout enum
        self._TaWholeBodyCommand = self._try_load_pb(
            paths=[
                ("aimdk.protocol.ta.ta_whole_body_command_pb2",
                 "TaWholeBodyCommand"),
            ],
            label="TA 单帧 command",
        )
        self._TaJointLayout = self._try_load_pb(
            paths=[
                ("aimdk.protocol.ta.ta_whole_body_command_pb2",
                 "TaJointLayout"),
            ],
            label="TA joint layout enum",
        )

        # SONIC external-token 输入通道 (sonic 侧订阅 /sonic/token_input,
        # token_io.mode=external_token_{pre,post}_fsq 时消费)。64D token wire 类型,
        # 见 gear_sonic_deploy/.../sonic/sonic_token_channel.proto::SonicTokenChannel。
        self._SonicTokenChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.sonic.sonic_token_channel_pb2",
                 "SonicTokenChannel"),
            ],
            label="SONIC token 输入通道",
        )

        # PncArm 插值通道 — pnc_arm 模块的"在线插值模式"用的 channel,
        # 切到 PncArmControlMode_ONLINE_TRAJECTORY 后才生效。
        # flag: 0=joint 左臂, 1=joint 右臂, 2=joint 双臂,
        #       100=SE3 左臂, 101=SE3 右臂, 102=SE3 双臂
        # positions: 17D — 前 3 维是腰 (实测控制不了, 填 0 占位), 后 14 维是双臂
        self._PncArmInterpolateChannel = self._try_load_pb(
            paths=[
                ("aimdk.protocol.pnc_arm.pnc_arm_channel_pb2",
                 "PncArmInterpolateChannel"),
            ],
            label="PncArm 插值",
        )

        # 真正的订阅/发布
        self._loco_pub = None
        self._waist_pub = None
        if self._RosMsgWrapper is not None:
            if self._MotionControlLocomotionVelocityChannel is not None:
                self._loco_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/motion/control/locomotion_velocity/pb_3Aaimdk_2Eprotocol_2EMotionControlLocomotionVelocityChannel",
                    qos_profile=qos)

            if self._MotionControlMoveWaistChannel is not None:
                self._waist_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/motion/control/move_waist/pb_3Aaimdk_2Eprotocol_2EMotionControlMoveWaistChannel",
                    qos_profile=qos)

            if self._TaWholeBodyCommandChannel is not None:
                self.create_subscription(
                    self._RosMsgWrapper,
                    "/ta/whole_body_command/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel",
                    self._ta_whole_body_cb, qos_profile=qos)

            # pnc_arm 插值通道发布器 — 仅在 pnc_arm 切到
            # PncArmControlMode_ONLINE_TRAJECTORY 时才有效, 否则 mc 模块会丢弃。
            self._pnc_arm_interp_pub = None
            if self._PncArmInterpolateChannel is not None:
                self._pnc_arm_interp_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/pnc_arm/motion/interpolate/pb_3Aaimdk_2Eprotocol_2EPncArmInterpolateChannel",
                    qos_profile=qos)

            # GR00T reference_window publisher — wire topic name follows
            # aimrt-src ros2_plugin naming: base + "/pb_" + ros2_name_encode(...).
            # Encoded "pb:aimdk.protocol.TaWholeBodyReferenceWindow" →
            # "pb_3Aaimdk_2Eprotocol_2ETaWholeBodyReferenceWindow".
            # topic base 已从 /gr00t/reference_window 改为 /wbc/infer/reference_window
            # (WBC 全身链路)。QoS 说明同下 (best_effort/volatile/1)。
            self._gr00t_ref_window_pub = None
            if self._TaWholeBodyReferenceWindow is not None:
                self._gr00t_ref_window_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/wbc/infer/reference_window/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyReferenceWindow",
                    qos_profile=qos)

            # SONIC /sonic/token_input publisher (external-token replay 直发路径)。
            # wire topic 命名同 ros2_plugin 规则:base + "/pb_" + ros2_name_encode(
            # "pb:aimdk.protocol.SonicTokenChannel") → "pb_3Aaimdk_2Eprotocol_2ESonicTokenChannel"。
            # QoS 与 reference_window 一致(best_effort/volatile);sonic 订阅端在
            # a3_aimrt_config 里同样按 best_effort 配。
            self._sonic_token_pub = None
            if self._SonicTokenChannel is not None:
                self._sonic_token_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/sonic/token_input/pb_3Aaimdk_2Eprotocol_2ESonicTokenChannel",
                    qos_profile=qos)

            # /ta/whole_body_command single-frame publisher (MC-direct path).
            # Same wire topic the TA module publishes /ta/whole_body_command on
            # (and that a3_server subscribes to for /get_ta_whole_body_command).
            # When emit_mode=ta_cmd, a 60Hz thread in interp_pub snapshots the
            # wb chunk's played frame, this callback builds a single-frame
            # TaWholeBodyCommandChannel and publishes it here. MC consumes it
            # exactly like a TA frame — no sonic in between.
            self._ta_whole_body_pub = None
            if self._TaWholeBodyCommandChannel is not None:
                self._ta_whole_body_pub = self.create_publisher(
                    self._RosMsgWrapper,
                    "/ta/whole_body_command/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel",
                    qos_profile=qos)

        # ==================== 发布器（插值）====================
        self.interp_pub = InterpolationPublisher(
            self, target_fps=150.0, interp_steps=5, hand_kind=self.hand_kind)

        # 50Hz timer to publish TaWholeBodyReferenceWindow from the interp_pub's
        # currently-loaded whole-body chunk. Skips the tick if no chunk is
        # loaded (interp_pub.wb_snapshot_for_reference_window returns None).
        # Kept OUT of the 150Hz interp loop so that loop stays lean; ROS timer
        # runs on the executor and its jitter (~1-2ms) is fine at 50Hz.
        self.REF_WINDOW_LEN = 10
        self.REF_WINDOW_PUBLISH_HZ = 50.0
        # seq 计数器,供 publish_reference_window_raw(direct-50Hz replay 直发路径)用。
        self._raw_ref_seq = 0
        # seq 计数器,供 publish_token_raw(external-token replay 直发路径)用。
        self._raw_token_seq = 0
        # Tell interp_pub the reference-window publish rate + length so its
        # 30→50Hz snapshot interpolation uses the SAME cadence as this timer
        # (window frame spacing == 1/REF_WINDOW_PUBLISH_HZ). Single source of
        # truth — don't let the two drift.
        self.interp_pub.set_reference_window_params(
            hz=self.REF_WINDOW_PUBLISH_HZ, window_len=self.REF_WINDOW_LEN,
        )
        if (self._gr00t_ref_window_pub is not None
                and self._TaWholeBodyReferenceWindow is not None
                and self._TaWholeBodyCommand is not None
                and self._TaJointLayout is not None):
            self._ref_window_timer = self.create_timer(
                1.0 / self.REF_WINDOW_PUBLISH_HZ,
                self._publish_gr00t_reference_window,
            )
            self.get_logger().info(
                f"/wbc/infer/reference_window publisher up "
                f"(window_len={self.REF_WINDOW_LEN}, "
                f"pub_hz={self.REF_WINDOW_PUBLISH_HZ})")
        else:
            self._ref_window_timer = None
            if self._TaWholeBodyReferenceWindow is None:
                self.get_logger().warn(
                    "TaWholeBodyReferenceWindow pb2 missing → "
                    "/wbc/infer/reference_window publisher disabled")

        # /ta/whole_body_command per-frame emit (MC-direct path).
        # emit_mode selects which wire path a whole-body chunk takes on
        # /send_chunk:
        #   "reference_window" (legacy) → 50Hz 10-frame TaWholeBodyReferenceWindow
        #                                 on /wbc/infer/reference_window (sonic consumes)
        #   "ta_cmd"            (default) → 60Hz single-frame TaWholeBodyCommandChannel
        #                                     on /ta/whole_body_command (MC consumes)
        # ta_cmd_hz is the emit rate for the ta_cmd path; the chunk itself may
        # arrive at any chunk_fps (VLA 20/30Hz), this thread linearly
        # interpolates between waypoints up to ta_cmd_hz.
        # Both paths share the same wb chunk buffer + s_used_local atomic swap,
        # so actual_delay semantics are identical.
        self._wb_emit_mode = "ta_cmd"   # "ta_cmd" | "reference_window"
        self._ta_cmd_hz = 60.0
        if (self._ta_whole_body_pub is not None
                and self._TaWholeBodyCommandChannel is not None
                and self._TaJointLayout is not None):
            self.interp_pub.set_ta_cmd_emit(self._ta_cmd_hz,
                                            self._publish_ta_whole_body_command)
            self.get_logger().info(
                f"/ta/whole_body_command emit ready (ta_cmd_hz={self._ta_cmd_hz}, "
                f"interpolated from chunk_fps; default emit_mode=ta_cmd)")
        else:
            # Fall back to reference_window if the single-frame pb2 is missing.
            self._wb_emit_mode = "reference_window"
            self.get_logger().warn(
                "TaWholeBodyCommandChannel pb2 missing → /ta/whole_body_command "
                "emit disabled, falling back to emit_mode=reference_window")

        self.get_logger().info(
            f"A3ServerNode 初始化完成 (robot_ip={robot_ip}, "
            f"hand_kind={self.hand_kind.value}, hand_dim={hand_dim(self.hand_kind)}, "
            f"不订阅相机)")

    def _try_load_pb(self, paths, label):
        """按优先顺序尝试 import (mod, name)，返回 class 或 None。"""
        for mod_name, cls_name in paths:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                self.get_logger().info(f"加载 {label} pb2: {mod_name}.{cls_name}")
                return cls
            except (ImportError, AttributeError):
                continue
        self.get_logger().warn(
            f"无法加载 {label} pb2，相关功能将禁用 (尝试过: {paths})")
        return None

    # ==================== 回调 ====================

    def _camera_cb(self, name: str, msg):
        """sensor_msgs/Image -> BGR ndarray, 缓存最新一帧。

        cv_bridge 会按 msg.encoding 自动转换；这里强制目标 bgr8 与 cv2/HTTP
        约定保持一致；mono / rgb8 / bgr8 / 16UC1 都能转。
        """
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            # 极端情况下 (例如 16UC1 深度) bgr8 转换会报错, 退化为原始解析
            try:
                img = self.cv_bridge.imgmsg_to_cv2(msg)
            except Exception as e2:
                self.get_logger().warn(f"相机 {name} 解码失败: {e} / {e2}")
                return
        with self._cam_lock:
            self.latest_cameras[name] = img

    def _arm_cb(self, msg):
        self.latest_arm_joints = {
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
        }

    def _hand_cb(self, msg):
        self.latest_hand_joints = {
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
        }

    def _eef_pose_cb(self, msg):
        if len(msg.poses) < 2:
            return

        def p2d(p):
            return {
                "position": {"x": p.position.x, "y": p.position.y, "z": p.position.z},
                "orientation": {"x": p.orientation.x, "y": p.orientation.y,
                                "z": p.orientation.z, "w": p.orientation.w},
            }

        self.latest_eef_pose = {"left": p2d(msg.poses[0]), "right": p2d(msg.poses[1])}

    def _neck_cb(self, msg):
        self.latest_neck_joints = {
            "position": list(msg.position),
            "velocity": list(msg.velocity) if msg.velocity else [],
            "effort": list(msg.effort) if msg.effort else [],
        }

    def _imu_cb(self, which: str, msg):
        """sensor_msgs/Imu callback. ``which`` ∈ {"pelvis", "torso"}.

        gravity_dir is derived from the reported orientation quaternion by
        rotating world-frame gravity [0, 0, -1] into the IMU body frame:
            R^T @ [0, 0, -1]  where R is the rotation matrix from quat.
        This matches sonic_a3_full modality.json's ``pelvis_gravity`` semantics.
        """
        q = msg.orientation
        # scipy-native xyzw order
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        # Body-frame gravity direction from quat (world gravity = -z_world):
        # Formula derived from R^T @ [0, 0, -1]. Rows of R^T are columns of R;
        # column 2 of R for (qx,qy,qz,qw) is:
        #   [ 2*(qx*qz + qy*qw),  2*(qy*qz - qx*qw),  1 - 2*(qx² + qy²) ].
        # Negate for gravity (-z_world).
        gx = -2.0 * (qx * qz + qy * qw)
        gy = -2.0 * (qy * qz - qx * qw)
        gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        cache = {
            "orientation_xyzw": [qx, qy, qz, qw],
            "angular_velocity": [float(msg.angular_velocity.x),
                                 float(msg.angular_velocity.y),
                                 float(msg.angular_velocity.z)],
            "linear_acceleration": [float(msg.linear_acceleration.x),
                                    float(msg.linear_acceleration.y),
                                    float(msg.linear_acceleration.z)],
            "gravity_dir": [gx, gy, gz],
            "timestamp": time.time(),
        }
        with self._imu_lock:
            if which == "pelvis":
                self.latest_pelvis_imu = cache
            else:
                self.latest_torso_imu = cache

    @staticmethod
    def _wrapper_data_to_bytes(msg) -> bytes:
        """把 RosMsgWrapper.data 转成 bytes。

        rclpy 在不同 ROS2 版本下，可能是 bytes / array.array('B') / list[bytes]，
        需要兼容；最后一种直接 bytes(...) 会报
        "'bytes' object cannot be interpreted as an integer"。
        """
        d = msg.data
        if isinstance(d, (bytes, bytearray, memoryview)):
            return bytes(d)
        try:
            return bytes(d)
        except TypeError:
            return b"".join(d)

    def _waist_cb(self, msg):
        self.latest_waist_joints = {
            "position": list(msg.position),
            "velocity": list(msg.velocity) if msg.velocity else [],
            "effort": list(msg.effort) if msg.effort else [],
        }

    def _ta_whole_body_cb(self, msg):
        """TA 全身指令回调：把 wrapper.data 反序列化为 TaWholeBodyCommandChannel 后缓存。"""
        ser = (msg.serialization_type or "").strip().lower()
        if ser not in ("pb", "protobuf", ""):
            return
        try:
            raw = self._wrapper_data_to_bytes(msg)
            ch = self._TaWholeBodyCommandChannel()
            ch.ParseFromString(raw)
            decoded = _msg_to_dict(ch)
        except Exception as e:
            self.get_logger().warn(f"TA whole_body 解析失败: {e}")
            return
        with self._ta_lock:
            self.latest_ta_whole_body = decoded
            self.latest_ta_whole_body_ts = time.time()
            self.latest_ta_whole_body_raw_len = len(raw)

    def _leg_cb(self, msg):
        self.latest_leg_joints = {
            "position": list(msg.position),
            "velocity": list(msg.velocity) if msg.velocity else [],
            "effort": list(msg.effort) if msg.effort else [],
        }

    # ==================== WBC whole-body state (whole_body 链路) ====================

    @staticmethod
    def _joint_group_to_dict(group) -> dict:
        """aimdk TaJointGroupState (repeated JointState) → {position, velocity,
        effort} 列表, 保持 proto 内顺序 (与 leg/waist/head/arm 命令顺序一致)。"""
        pos, vel, eff = [], [], []
        for js in group.states:
            pos.append(float(js.position))
            vel.append(float(js.velocity))
            eff.append(float(js.effort))
        return {"position": pos, "velocity": vel, "effort": eff}

    @staticmethod
    def _ta_imu_to_dict(imu) -> dict:
        """aimdk TaImuState → 与 _imu_cb 同 shape 的缓存 dict。

        TaImuState.quat 是 wxyz (proto), 转成 scipy-native xyzw; gravity_dir
        沿用 _imu_cb 的公式 (world 重力 [0,0,-1] 反旋到 IMU body frame)。
        """
        q = list(imu.quat_wxyz) if len(imu.quat_wxyz) >= 4 else [1.0, 0.0, 0.0, 0.0]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        gx = -2.0 * (qx * qz + qy * qw)
        gy = -2.0 * (qy * qz - qx * qw)
        gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        av = list(imu.angular_velocity_xyz)
        la = list(imu.linear_acceleration_xyz)
        return {
            "orientation_xyzw": [qx, qy, qz, qw],
            "angular_velocity": [float(x) for x in (av[:3] or [0.0, 0.0, 0.0])],
            "linear_acceleration": [float(x) for x in (la[:3] or [0.0, 0.0, 0.0])],
            "gravity_dir": [gx, gy, gz],
            "timestamp": time.time(),
        }

    def _whole_body_state_cb(self, msg):
        """/wbc/whole_body_state 回调: RosMsgWrapper.data → TaWholeBodyStateChannel,
        填 latest_wb_* 缓存 (leg/waist/head→neck/arm + pelvis/torso IMU)。手不在
        此消息内 (whole_body 用手时从分散 latest_hand_joints 取)。"""
        ser = (msg.serialization_type or "").strip().lower()
        if ser not in ("pb", "protobuf", ""):
            return
        try:
            raw = self._wrapper_data_to_bytes(msg)
            ch = self._TaWholeBodyStateChannel()
            ch.ParseFromString(raw)
            data = ch.data
            leg = self._joint_group_to_dict(data.leg_state)
            waist = self._joint_group_to_dict(data.waist_state)
            neck = self._joint_group_to_dict(data.head_state)
            arm = self._joint_group_to_dict(data.arm_state)
            pelvis = self._ta_imu_to_dict(data.pelvis_imu) \
                if data.HasField("pelvis_imu") else None
            torso = self._ta_imu_to_dict(data.torso_imu) \
                if data.HasField("torso_imu") else None
        except Exception as e:
            self.get_logger().warn(f"whole_body_state 解析失败: {e}")
            return
        with self._wb_state_lock:
            self.latest_wb_leg_joints = leg
            self.latest_wb_waist_joints = waist
            self.latest_wb_neck_joints = neck
            self.latest_wb_arm_joints = arm
            if pelvis is not None:
                self.latest_wb_pelvis_imu = pelvis
            if torso is not None:
                self.latest_wb_torso_imu = torso

    def enable_wb_state_source(self) -> bool:
        """懒创建 /wbc/whole_body_state 订阅 (whole_body 模式)。幂等; 由
        /set_state_source 调用。需要 RosMsgWrapper + TaWholeBodyStateChannel pb2。

        upper_body 模式永不调用此方法, 故该订阅不会创建, 分散链路不受影响。
        """
        if self._RosMsgWrapper is None or self._TaWholeBodyStateChannel is None:
            self.get_logger().warn(
                "无法创建 whole_body_state 订阅: 缺 RosMsgWrapper 或 "
                "TaWholeBodyStateChannel pb2")
            return False
        with self._wb_sub_lock:
            if self._wb_state_sub is not None:
                return True
            self._wb_state_sub = self.create_subscription(
                self._RosMsgWrapper,
                "/wbc/whole_body_state/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyStateChannel",
                self._whole_body_state_cb, qos_profile=_create_qos())
        self.get_logger().info(
            "whole_body_state 订阅已创建 (source=whole_body): "
            "/wbc/whole_body_state/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyStateChannel")
        return True

    # ==================== /wbc/infer/reference_window publisher timer ====================

    def _publish_gr00t_reference_window(self):
        """50Hz timer callback: snapshot the whole-body chunk into a 10-frame
        window and publish as TaWholeBodyReferenceWindow on /wbc/infer/reference_window.

        Skips the tick if no whole-body chunk is loaded — sonic will just see
        no reference until a new chunk arrives (its subscriber holds latest).
        """
        snap = self.interp_pub.wb_snapshot_for_reference_window(
            window_len=self.REF_WINDOW_LEN,
        )
        if snap is None:
            return
        q31 = snap["q31"]              # (10, 31)
        dq31 = snap["dq31"]            # (10, 31)
        pelvis_wxyz = snap["pelvis_wxyz"]  # (10, 4)

        # Build TaWholeBodyReferenceWindow proto.
        win = self._TaWholeBodyReferenceWindow()
        # Header
        _fill_pb_header(win.header)
        win.header.seq = snap["seq"]
        # Frames
        BODY_31 = self._TaJointLayout.TaJointLayout_BODY_31
        for k in range(q31.shape[0]):
            cmd = win.frames.add()
            cmd.joint_layout = BODY_31
            cmd.pelvis_pose.quat_wxyz.extend(pelvis_wxyz[k].tolist())
            cmd.leg_command.angles_rad.extend(q31[k, 0:12].tolist())
            cmd.waist_command.angles_rad.extend(q31[k, 12:15].tolist())
            cmd.head_command.angles_rad.extend(q31[k, 15:17].tolist())
            cmd.arm_command.angles_rad.extend(q31[k, 17:31].tolist())
            cmd.joint_velocities.velocities_rad_s.extend(dq31[k].tolist())

        # Wrap into RosMsgWrapper (aimrt ros2_plugin wire format).
        wrap = self._RosMsgWrapper()
        wrap.serialization_type = "pb"
        wrap.data = win.SerializeToString()
        self._gr00t_ref_window_pub.publish(wrap)

    def publish_reference_window_raw(self, leg, waist, arm, pelvis_wxyz,
                                     dq31=None, head=None):
        """直发路径(sim direct-50Hz replay 专用)。

        客户端已把整条 episode 插值到 50Hz 并自己滑 10 帧窗口,这里把传入的
        窗口**原样**组成 TaWholeBodyReferenceWindow 直接 publish —— 不做
        wb_snapshot / 30→50Hz 插值 / RTC swap。与 _publish_gr00t_reference_window
        用同一份 proto 填充和同一个 publisher,故对 sonic 完全透明。

        入参(H 通常=10):leg(H,12) waist(H,3) arm(H,14) pelvis_wxyz(H,4);
        dq31(H,31) 可选(缺则置零);head(H,2) 可选(缺则置零 hold)。
        """
        if self._gr00t_ref_window_pub is None or self._TaWholeBodyReferenceWindow is None:
            return False
        leg = np.asarray(leg, dtype=np.float64)
        waist = np.asarray(waist, dtype=np.float64)
        arm = np.asarray(arm, dtype=np.float64)
        pelvis_wxyz = np.asarray(pelvis_wxyz, dtype=np.float64)
        dq31 = np.asarray(dq31, dtype=np.float64) if dq31 is not None else None
        head = np.asarray(head, dtype=np.float64) if head is not None else None
        H = leg.shape[0]

        win = self._TaWholeBodyReferenceWindow()
        _fill_pb_header(win.header)
        win.header.seq = self._raw_ref_seq & 0xFFFFFFFF
        self._raw_ref_seq += 1
        BODY_31 = self._TaJointLayout.TaJointLayout_BODY_31
        for k in range(H):
            cmd = win.frames.add()
            cmd.joint_layout = BODY_31
            cmd.pelvis_pose.quat_wxyz.extend(pelvis_wxyz[k].tolist())
            cmd.leg_command.angles_rad.extend(leg[k].tolist())
            cmd.waist_command.angles_rad.extend(waist[k].tolist())
            head_k = head[k].tolist() if head is not None else [0.0, 0.0]
            cmd.head_command.angles_rad.extend(head_k)
            cmd.arm_command.angles_rad.extend(arm[k].tolist())
            if dq31 is not None:
                cmd.joint_velocities.velocities_rad_s.extend(dq31[k].tolist())
            else:
                cmd.joint_velocities.velocities_rad_s.extend([0.0] * 31)

        wrap = self._RosMsgWrapper()
        wrap.serialization_type = "pb"
        wrap.data = win.SerializeToString()
        self._gr00t_ref_window_pub.publish(wrap)
        return True

    def publish_token_raw(self, token):
        """直发路径(sim external-token replay 专用)。

        把一个 64D token(pre_fsq 连续 latent 或 post_fsq 量化码,由 sonic
        token_io.mode 决定如何解释)原样组成 SonicTokenChannel 发到
        /sonic/token_input。发布节拍完全由调用方(driver)决定。

        入参: token(64,) float。返回 True/False。
        """
        if self._sonic_token_pub is None or self._SonicTokenChannel is None:
            return False
        token = np.asarray(token, dtype=np.float64).reshape(-1)
        if token.shape[0] != 64:
            return False
        ch = self._SonicTokenChannel()
        _fill_pb_header(ch.header)
        ch.header.seq = self._raw_token_seq & 0xFFFFFFFF
        self._raw_token_seq += 1
        data = ch.data
        # stamp_ns:与 header 一致的生产端时间戳(sonic 按 seq 判 fresh,stamp 仅记录)。
        data.stamp_ns = int(datetime.utcnow().timestamp() * 1e9)
        data.encoded_token.extend(token.tolist())

        wrap = self._RosMsgWrapper()
        wrap.serialization_type = "pb"
        wrap.data = ch.SerializeToString()
        self._sonic_token_pub.publish(wrap)
        return True



    def _publish_ta_whole_body_command(self, q31, dq31, pelvis_wxyz,
                                       seq, chunk_id):
        """60Hz callback: build a single-frame TaWholeBodyCommandChannel from
        one interpolated wb frame and publish it on /ta/whole_body_command.

        This is the MC-direct path (emit_mode=ta_cmd): MC consumes it exactly
        like a TA frame, no sonic in between. q31 is BODY_31
        (leg(12)+waist(3)+head(2)+arm(14)); head is the neck_hold snapshot
        already baked in by _build_q31_dq31. pelvis_position_xyz is left empty
        (VLA doesn't predict pelvis translation; TA's loco/walk handles that).
        """
        if self._ta_whole_body_pub is None or self._TaWholeBodyCommandChannel is None:
            return
        try:
            ch = self._TaWholeBodyCommandChannel()
            _fill_pb_header(ch.header)
            ch.header.seq = int(seq) & 0xFFFFFFFF
            cmd = ch.data
            cmd.joint_layout = self._TaJointLayout.TaJointLayout_BODY_31
            cmd.pelvis_pose.quat_wxyz.extend(np.asarray(pelvis_wxyz,
                                                        dtype=np.float64).tolist())
            cmd.leg_command.angles_rad.extend(
                np.asarray(q31[0:12], dtype=np.float64).tolist())
            cmd.waist_command.angles_rad.extend(
                np.asarray(q31[12:15], dtype=np.float64).tolist())
            cmd.head_command.angles_rad.extend(
                np.asarray(q31[15:17], dtype=np.float64).tolist())
            cmd.arm_command.angles_rad.extend(
                np.asarray(q31[17:31], dtype=np.float64).tolist())
            cmd.joint_velocities.velocities_rad_s.extend(
                np.asarray(dq31, dtype=np.float64).tolist())

            wrap = self._RosMsgWrapper()
            wrap.serialization_type = "pb"
            wrap.data = ch.SerializeToString()
            self._ta_whole_body_pub.publish(wrap)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"/ta/whole_body_command publish failed: {e}")

    # ==================== TF ====================

    def get_eef_state_tf(self):
        try:
            lt = self.tf_buffer.lookup_transform(
                "base_link", "left_arm_link07", rclpy.time.Time())
            rt = self.tf_buffer.lookup_transform(
                "base_link", "right_arm_link07", rclpy.time.Time())

            def t2d(tf):
                t = tf.transform
                return {
                    "position": {"x": t.translation.x, "y": t.translation.y, "z": t.translation.z},
                    "orientation": {"x": t.rotation.x, "y": t.rotation.y,
                                    "z": t.rotation.z, "w": t.rotation.w},
                }

            return {"left": t2d(lt), "right": t2d(rt)}
        except Exception:
            return None

    # ==================== 直接发布 ====================

    def publish_arm(self, values):
        self.interp_pub.publish_arm_raw(values)

    def publish_hand(self, values):
        self.interp_pub.publish_hand_raw(values)

    def publish_eef(self, values):
        self.interp_pub.publish_eef_raw(values)

    def publish_head(self, shake, nod):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "user_McScript"  # motion control 守护进程鉴权用, 必须给
        msg.name = ["head_yaw_joint", "head_pitch_joint"]
        msg.position = [float(shake), float(nod)]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        self._neck_pub.publish(msg)
        return True

    def publish_loco(self, forward, lateral, angular, mode=0):
        """通过 protobuf RosMsgWrapper 发布行走速度命令。

        mode: 0 = 默认 / 1 = 导航 (与 T_LocomotionVelocity.py 一致)。
        """
        if self._loco_pub is None or self._MotionControlLocomotionVelocityChannel is None:
            self.get_logger().warn("行走 Topic 发布器未初始化")
            return False
        try:
            ch = self._MotionControlLocomotionVelocityChannel()
            _fill_pb_header(ch.header)
            ch.data.mode = int(mode)
            ch.data.forward_velocity = float(forward)
            ch.data.lateral_velocity = float(lateral)
            ch.data.angular_velocity = float(angular)

            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._loco_pub.publish(wrapper)
            return True
        except Exception as e:
            self.get_logger().error(f"行走 Topic 发布失败: {e}")
            return False

    def publish_waist(self, pitch=0.0, roll=0.0, yaw=0.0, height=0.0):
        """通过 protobuf RosMsgWrapper 发布腰部控制命令 (与 T_WaistMove.py 同形)。

        参数范围:
          pitch:  [-0.5, 0.5]   rad
          roll:   [-0.3, 0.3]   rad
          yaw:    [-1.57, 1.57] rad
          height: [-0.4, 0.0]   m

        Proto (新 AIMA layout, motion_control/motion/mc_motion_channel.proto):
          message MotionControlMoveWaistChannel {
              Header header = 1;
              double waist_pitch = 2; double waist_roll = 3;
              double waist_yaw = 4;   double waist_height = 5;
          }
        字段是扁平的 (与老 mc/motion/.../McMoveWaistChannel 嵌套 data 不同),
        机器人上 mc 守护进程订阅的是这一套, 必须用这个 proto 才能反序列化。
        """
        if self._waist_pub is None or self._MotionControlMoveWaistChannel is None:
            self.get_logger().warn("腰部 Topic 发布器未初始化")
            return False
        try:
            ch = self._MotionControlMoveWaistChannel()
            _fill_pb_header(ch.header)
            ch.waist_pitch = float(pitch)
            ch.waist_roll = float(roll)
            ch.waist_yaw = float(yaw)
            ch.waist_height = float(height)

            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._waist_pub.publish(wrapper)
            return True
        except Exception as e:
            self.get_logger().error(f"腰部 Topic 发布失败: {e}")
            return False

    def publish_leg(self, values):
        """发布腿部关节命令 (sensor_msgs/JointState 到 /body_drive/leg_joint_command_ros2)。

        values: 长度 12 的列表/数组，顺序对应 LEG_JOINT_NAMES。
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "user_McScript"  # motion control 守护进程鉴权用
        msg.name = list(LEG_JOINT_NAMES)
        msg.position = [float(x) for x in values]
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)
        self._leg_pub.publish(msg)
        return True

    def publish_arm_interpolate(self, positions, flag=2,
                                  velocities=None, accelerations=None, effort=None):
        """通过 PncArmInterpolateChannel 发布双臂插值控制命令。

        前提: pnc_arm 模块已切到 PncArmControlMode_ONLINE_TRAJECTORY (用
        scripts/utils/change_pnc_arm_mode_a3.sh 或 ADU 上手动跑 S_SetControlMode.py)。
        否则 mc/pnc_arm 守护进程会丢弃命令。

        Args:
            positions: 14D (双臂关节角弧度) 或 14D (SE3 双臂位姿) 或 17D (含 3D 腰占位)。
              - 14D 时自动补 [0,0,0] 在前 (腰占位, 实测不可控)
              - 17D 时直接透传, 调用方自行处理腰前 3 维 (建议填 0)
            flag: 0=joint 左臂, 1=joint 右臂, 2=joint 双臂 (默认),
                  100=SE3 左臂, 101=SE3 右臂, 102=SE3 双臂
            velocities/accelerations/effort: 与 positions 等长, None 时填全 0
        """
        if self._pnc_arm_interp_pub is None or self._PncArmInterpolateChannel is None:
            self.get_logger().warn("PncArm 插值发布器未初始化")
            return False
        try:
            pos = [float(x) for x in positions]
            # 14D → 补腰前 3 维占位 (proto 设计前 3 维是腰; A3 上腰在此通道控制不了,
            # 只能用 mc 的 publish_waist, 这里填 0 占位即可)
            if len(pos) == 14:
                pos = [0.0, 0.0, 0.0] + pos
            n = len(pos)

            ch = self._PncArmInterpolateChannel()
            _fill_pb_header(ch.header)
            ch.flag = int(flag)
            ch.positions[:] = pos
            ch.velocities[:] = ([0.0] * n) if velocities is None \
                                else [float(x) for x in velocities]
            ch.accelerations[:] = ([0.0] * n) if accelerations is None \
                                  else [float(x) for x in accelerations]
            ch.effort[:] = ([0.0] * n) if effort is None \
                           else [float(x) for x in effort]

            wrapper = self._RosMsgWrapper()
            wrapper.serialization_type = "pb"
            wrapper.data = ch.SerializeToString()
            self._pnc_arm_interp_pub.publish(wrapper)
            return True
        except Exception as e:
            self.get_logger().error(f"PncArm 插值发布失败: {e}")
            return False


# ==================== Flask HTTP API ====================

app = Flask(__name__)
node: "A3ServerNode" = None


# --- 状态获取 ---
@app.route("/get_joint_states", methods=["GET"])
def get_joint_states():
    return jsonify({
        "arm":   node.latest_arm_joints or None,
        "hand":  node.latest_hand_joints or None,
        "neck":  node.latest_neck_joints or None,
        "waist": node.latest_waist_joints or None,
        "leg":   node.latest_leg_joints or None,
        "eef":   node.latest_eef_pose or node.get_eef_state_tf(),
    })


@app.route("/get_eef_state", methods=["GET"])
def get_eef_state():
    state = node.latest_eef_pose or node.get_eef_state_tf()
    if state is None:
        return jsonify({"error": "no eef state"}), 500
    return jsonify(state)


# --- 相机 ---
def _img_response(img):
    if img is None:
        return jsonify({"error": "No image"}), 404
    ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return jsonify({"error": "Encode failed"}), 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


def _encode_image_b64(img):
    """JPEG/base64 encode one cached camera frame for a JSON response."""
    if img is None:
        return None
    try:
        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        import base64
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


# cv2.imencode releases the GIL. Reusing one module-level pool keeps the 10-way
# A3 snapshot close to one JPEG encode in latency instead of serialising all
# requested cameras in the RTC hot path.
_ENCODE_POOL = ThreadPoolExecutor(max_workers=12, thread_name_prefix="jpegenc")


def _encode_images_b64_parallel(images):
    """Encode camera frames concurrently while preserving input order."""
    return list(_ENCODE_POOL.map(_encode_image_b64, images))


@app.route("/get_observation_with_progress", methods=["GET"])
def get_observation_with_progress():
    """Return A3 cameras, state and chunk progress in one RTC snapshot.

    Query parameters:
      cameras: comma-separated camera names (default: all cameras)
      include_imu: true/false (default: true)
      state_source: ``whole_body_state`` or empty/scattered state

    JPEG encoding is deliberately outside every ROS/interpolator lock.
    """
    cam_arg = request.args.get("cameras", "").strip()
    if cam_arg:
        cam_names = [name.strip() for name in cam_arg.split(",") if name.strip()]
        cam_names = [name for name in cam_names if name in CAMERA_TOPICS]
    else:
        cam_names = list(CAMERA_TOPICS)
    include_imu = request.args.get("include_imu", "true").lower() != "false"
    use_wb_state = request.args.get("state_source", "") == "whole_body_state"

    # The interpolator lock defines the snapshot instant for RTC. Cache locks
    # are only held long enough to copy references/dicts; callbacks replace
    # complete cache objects and never mutate an object after publishing it.
    with node.interp_pub._lock:
        chunk_progress = node.interp_pub._chunk_played_idx_nolock()

        with node._cam_lock:
            camera_images = [node.latest_cameras.get(name) for name in cam_names]

        if use_wb_state:
            with node._wb_state_lock:
                joints = {
                    "leg": node.latest_wb_leg_joints,
                    "waist": node.latest_wb_waist_joints,
                    "neck": node.latest_wb_neck_joints,
                    "arm": node.latest_wb_arm_joints,
                    "hand": node.latest_hand_joints,
                }
                if include_imu:
                    pelvis = (dict(node.latest_wb_pelvis_imu)
                              if node.latest_wb_pelvis_imu else None)
                    torso = (dict(node.latest_wb_torso_imu)
                             if node.latest_wb_torso_imu else None)
                    imu_out = {"pelvis": pelvis, "torso": torso}
                else:
                    imu_out = None
            eef_fallback = False
        else:
            joints = {
                "arm": node.latest_arm_joints,
                "hand": node.latest_hand_joints,
                "neck": node.latest_neck_joints,
                "waist": node.latest_waist_joints,
                "leg": node.latest_leg_joints,
                "eef": node.latest_eef_pose,
            }
            if include_imu:
                with node._imu_lock:
                    pelvis = (dict(node.latest_pelvis_imu)
                              if node.latest_pelvis_imu else None)
                    torso = (dict(node.latest_torso_imu)
                             if node.latest_torso_imu else None)
                imu_out = {"pelvis": pelvis, "torso": torso}
            else:
                imu_out = None
            eef_fallback = joints["eef"] is None

        snapshot_timestamp = time.time()

    # TF lookup may block and is not sampled from the same ROS message anyway;
    # never hold the 150 Hz interpolator lock while doing it.
    if eef_fallback:
        joints["eef"] = node.get_eef_state_tf()

    encoded = _encode_images_b64_parallel(camera_images)
    out = dict(zip(cam_names, encoded))
    out.update({
        "joints": joints,
        "imu": imu_out,
        "chunk_progress": chunk_progress,
        "state_source": "whole_body_state" if use_wb_state else None,
        "timestamp": snapshot_timestamp,
    })
    return jsonify(out)


@app.route("/list_cameras", methods=["GET"])
def list_cameras():
    """列出 server 端订阅的所有 A3 相机名称及对应 topic。"""
    return jsonify({
        "cameras": list(CAMERA_TOPICS.keys()),
        "topics": CAMERA_TOPICS,
    })


@app.route("/get_camera/<name>", methods=["GET"])
def get_camera(name):
    """取最近一帧 BGR 图像并以 JPEG 返回。

    name 必须是 CAMERA_TOPICS 的 key, 例如:
        head_stereo_left, head_stereo_right, head_left, head_right, head_rear,
        chest_front, waist_front, wrist_left, wrist_right, armpit_right
    """
    if name not in CAMERA_TOPICS:
        return jsonify({"error": f"unknown camera: {name}",
                        "available": list(CAMERA_TOPICS.keys())}), 404
    with node._cam_lock:
        img = node.latest_cameras.get(name)
    return _img_response(img)


@app.route("/get_ta_whole_body_command", methods=["GET"])
def get_ta_whole_body_command():
    """返回最近一帧 TA 全身指令（已 protobuf parse）的 dict 表示。

    返回:
        200: {"timestamp": <wall_t>, "data_len": <bytes>, "decoded": {...}}
        404: 还未收到任何 TA 全身指令
        503: ta_channel_pb2 未加载（缺 aima_protocol-main 生成的 pb2）
    """
    if node._TaWholeBodyCommandChannel is None:
        return jsonify({"error": "ta_channel_pb2 未加载"}), 503
    with node._ta_lock:
        decoded = node.latest_ta_whole_body
        ts = node.latest_ta_whole_body_ts
        ln = node.latest_ta_whole_body_raw_len
    if decoded is None:
        return jsonify({"error": "未收到任何 TA 全身指令"}), 404
    return jsonify({"timestamp": ts, "data_len": ln, "decoded": decoded})


# --- WBC whole-body state (whole_body 链路) ---
@app.route("/set_state_source", methods=["POST"])
def set_state_source():
    """选择本 session 的 state 链路 (客户端在 detect_embodiment_kind 后调一次)。

    POST JSON: {"source": "whole_body" | "upper_body"}
      - whole_body → 懒创建 /wbc/whole_body_state 订阅 (enable_wb_state_source)。
      - upper_body → 不创建 wbc 订阅 (分散链路照旧, 见 /get_joint_states)。

    幂等; 老客户端不调用则维持分散链路, 向后兼容。
    """
    d = request.get_json(silent=True) or {}
    source = str(d.get("source", "")).strip().lower()
    if source not in ("whole_body", "upper_body"):
        return jsonify({"ok": False,
                        "error": "source must be 'whole_body' or 'upper_body'"}), 400
    if source == "whole_body":
        ok = node.enable_wb_state_source()
        return (jsonify({"ok": bool(ok), "source": source,
                         "wb_state_sub": ok}),
                200 if ok else 503)
    # upper_body: 无需启用任何东西 (wbc 订阅不创建)。
    return jsonify({"ok": True, "source": source,
                    "wb_state_sub": node._wb_state_sub is not None})


@app.route("/get_whole_body_state", methods=["GET"])
def get_whole_body_state():
    """返回 whole_body 链路的 state: 身体 (leg/waist/neck/arm) + pelvis/torso IMU
    来自 /wbc/whole_body_state, 手 (hand) 来自分散的 /motion/control/hand_joint_state
    (TaWholeBodyState 不含手, whole_body 也要用手)。

    503: ta_whole_body_state_pb2 未加载。字段为 None 表示尚未收到对应数据 /
         (对 hand) 分散订阅尚无数据。
    """
    if node._TaWholeBodyStateChannel is None:
        return jsonify({"error": "ta_whole_body_state_pb2 未加载"}), 503
    with node._wb_state_lock:
        leg = node.latest_wb_leg_joints
        waist = node.latest_wb_waist_joints
        neck = node.latest_wb_neck_joints
        arm = node.latest_wb_arm_joints
        pelvis = dict(node.latest_wb_pelvis_imu) if node.latest_wb_pelvis_imu else None
        torso = dict(node.latest_wb_torso_imu) if node.latest_wb_torso_imu else None
    # 手走分散链路 (always-on hand sub)。
    hand = node.latest_hand_joints or None
    return jsonify({
        "joints": {"leg": leg, "waist": waist, "neck": neck,
                   "arm": arm, "hand": hand},
        "imu": {"pelvis": pelvis, "torso": torso},
        "timestamp": time.time(),
    })


# --- 命令 ---
def _parse_wait_args():
    wait = request.args.get("wait", "").lower() == "true"
    try:
        settle_ms = float(request.args.get("settle_ms", "0"))
    except Exception:
        settle_ms = 0.0
    return wait, max(0.0, settle_ms) / 1000.0


@app.route("/send_arm", methods=["POST"])
def send_arm():
    values = request.get_json()["values"]
    if request.args.get("raw", "").lower() == "true":
        node.interp_pub.publish_arm_raw(values)
    else:
        node.interp_pub.set_arm_target(values)
        wait, settle = _parse_wait_args()
        if wait:
            node.interp_pub.wait_arm_done(settle_sec=settle)
    return jsonify({"status": "ok"})


@app.route("/send_hand", methods=["POST"])
def send_hand():
    data = request.get_json()
    values = data["values"]
    effort = data.get("effort")
    if request.args.get("raw", "").lower() == "true":
        node.interp_pub.publish_hand_raw(values, effort=effort)
    else:
        node.interp_pub.set_hand_target(values, effort=effort)
        wait, settle = _parse_wait_args()
        if wait:
            node.interp_pub.wait_hand_done(settle_sec=settle)
    return jsonify({"status": "ok"})


@app.route("/send_eef", methods=["POST"])
def send_eef():
    values = request.get_json()["values"]
    if request.args.get("raw", "").lower() == "true":
        node.interp_pub.publish_eef_raw(values)
    else:
        node.interp_pub.set_eef_target(values)
        wait, settle = _parse_wait_args()
        if wait:
            node.interp_pub.wait_eef_done(settle_sec=settle)
    return jsonify({"status": "ok"})


@app.route("/send_head", methods=["POST"])
def send_head():
    d = request.get_json()
    node.publish_head(d["shake"], d["nod"])
    return jsonify({"status": "ok"})


@app.route("/send_loco", methods=["POST"])
def send_loco():
    d = request.get_json()
    ok = node.publish_loco(d["forward"], d["lateral"], d["angular"], d.get("mode", 0))
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/send_waist", methods=["POST"])
def send_waist():
    """POST JSON: {pitch, roll, yaw, height} 单位 rad/m。

    兼容旧调用：传 {"values": [yaw, roll, pitch]} 时按顺序 fallback。
    """
    d = request.get_json() or {}
    if "values" in d:
        v = d["values"]
        # 兼容旧 a2 风格 [yaw, roll, pitch]
        ok = node.publish_waist(yaw=float(v[0]), roll=float(v[1]),
                                 pitch=float(v[2]),
                                 height=float(v[3]) if len(v) > 3 else 0.0)
    else:
        ok = node.publish_waist(
            pitch=float(d.get("pitch", 0.0)),
            roll=float(d.get("roll", 0.0)),
            yaw=float(d.get("yaw", 0.0)),
            height=float(d.get("height", 0.0)),
        )
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/send_leg", methods=["POST"])
def send_leg():
    d = request.get_json()
    values = d["values"]
    ok = node.publish_leg(values)
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/send_arm_interp", methods=["POST"])
def send_arm_interp():
    """PncArm 在线插值接口 — 走 PncArmInterpolateChannel。

    需要 pnc_arm 切到 PncArmControlMode_ONLINE_TRAJECTORY (用 scripts/utils/change_pnc_arm_mode_a3.sh)。

    POST JSON:
        values:        14D 双臂关节角弧度 (joint 模式) 或 14D SE3 双臂位姿
                       (左 px,py,pz,qx,qy,qz,qw + 右同) (eef 模式) 或 17D 直接透传
        flag:          int, 默认 2 (joint 双臂)
                       0/1/2 = joint 左/右/双; 100/101/102 = SE3 左/右/双
        velocities/accelerations/effort: 可选, 与 positions 等长, 默认全 0
    """
    d = request.get_json() or {}
    values = d["values"]
    flag = int(d.get("flag", 2))
    ok = node.publish_arm_interpolate(
        positions=values, flag=flag,
        velocities=d.get("velocities"),
        accelerations=d.get("accelerations"),
        effort=d.get("effort"),
    )
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/send_eef_interp", methods=["POST"])
def send_eef_interp():
    """A3 EEF (SE3 位姿) 插值控制 — 复用 PncArmInterpolateChannel + flag=102 (SE3 双臂)。

    需要 pnc_arm 切到 PncArmControlMode_ONLINE_TRAJECTORY。

    POST JSON:
        values: 14D [left_px, left_py, left_pz, left_qx, left_qy, left_qz, left_qw,
                     right_px, right_py, right_pz, right_qx, right_qy, right_qz, right_qw]
        flag:   可选, 默认 102 (SE3 双臂); 100/101 = SE3 左/右臂
    """
    d = request.get_json() or {}
    values = d["values"]
    flag = int(d.get("flag", 102))
    ok = node.publish_arm_interpolate(positions=values, flag=flag)
    return jsonify({"status": "ok" if ok else "error"})


@app.route("/set_speed", methods=["POST"])
def set_speed():
    d = request.get_json()
    hz = float(d["hz"])
    node.interp_pub.set_send_fps(hz)
    return jsonify({"status": "ok", "hz": hz, "interp_steps": node.interp_pub.interp_steps})


@app.route("/wait_step_done", methods=["POST"])
def wait_step_done():
    d = request.get_json(silent=True) or {}
    settle_ms = float(d.get("settle_ms", 0.0))
    timeout_ms = float(d.get("timeout_ms", 2000.0))
    node.interp_pub.wait_all_done(settle_sec=settle_ms / 1000.0,
                                   timeout_sec=timeout_ms / 1000.0)
    return jsonify({"status": "ok"})


@app.route("/send_chunk", methods=["POST"])
def send_chunk():
    """两条路径合一:

    (A) legacy A3 chunk: 只有 arm/hand/eef, 走 InterpolationPublisher.set_*_chunk,
        通过 /motion/control/*_joint_command 直接驱动 A3 电机。返回 {"status": "ok"}。

    (B) sonic-a3 whole-body chunk: 收到 arm+leg+waist+pelvis_quat_wxyz (可选
        s_used_local, chunk_id, adaptive_transition), server 端 atomic swap +
        50Hz publish TaWholeBodyReferenceWindow 到 /wbc/infer/reference_window,
        sonic 侧订阅并驱动底层。返回 {"ok", "actual_delay", ...}。

    路由判据: 收到任一 (leg / waist / pelvis_quat_wxyz / s_used_local) 就走 (B)。
    whole-body hand 不进 reference_window，但会与 body 在同一个服务端锁内
    按同一 actual_delay 切片并装载；legacy hand 仍独立装载。
    """
    d = request.get_json() or {}
    chunk_fps = float(d.get("chunk_fps", 30.0))

    # ---- (C) UPPER-BODY server-atomic chunk (arm + hand + waist) ----
    # Added alongside the whole-body (B) and legacy (A) paths. Routed FIRST and
    # explicitly by mode=="upper_body" so it never collides with the existing
    # waist/s_used_local -> whole-body heuristic below (which is left untouched).
    if d.get("mode") == "upper_body":
        arm = d.get("arm")
        hand = d.get("hand")
        waist = d.get("waist")
        arm_arr = np.asarray(arm, dtype=np.float64) if arm is not None else None
        hand_arr = np.asarray(hand, dtype=np.float64) if hand is not None else None
        waist_arr = np.asarray(waist, dtype=np.float64) if waist is not None else None
        if arm_arr is not None and (arm_arr.ndim != 2 or arm_arr.shape[1] != 14):
            return jsonify({"ok": False,
                            "error": f"arm expected (H,14), got {arm_arr.shape}"}), 400
        if waist_arr is not None and (waist_arr.ndim != 2 or waist_arr.shape[1] != 4):
            return jsonify({"ok": False,
                            "error": f"waist expected (H,4), got {waist_arr.shape}"}), 400
        if hand_arr is not None:
            if str(d.get("hand_value", "raw")).lower() != "raw":
                return jsonify({
                    "ok": False,
                    "error": "upper-body hand must be actuator/raw at server boundary",
                }), 400
            expected_hand_dim = hand_dim(node.hand_kind)
            if hand_arr.ndim != 2 or hand_arr.shape[1] != expected_hand_dim:
                return jsonify({
                    "ok": False,
                    "error": (
                        f"hand expected (H,{expected_hand_dim}) actuator values, "
                        f"got {hand_arr.shape}"
                    ),
                }), 400
        horizons = {
            name: arr.shape[0]
            for name, arr in (
                ("arm", arm_arr),
                ("hand", hand_arr),
                ("waist", waist_arr),
            )
            if arr is not None
        }
        if horizons and (
            min(horizons.values()) <= 0
            or len(set(horizons.values())) != 1
        ):
            return jsonify({
                "ok": False,
                "error": f"upper-body chunk horizons must match and be non-empty: {horizons}",
            }), 400
        s_used_local = d.get("s_used_local", None)
        s_used_local = int(s_used_local) if s_used_local is not None else None
        hand_effort = d.get("hand_effort")
        resp = node.interp_pub.swap_upper_chunk_atomic(
            arm=arm_arr, hand=hand_arr, waist=waist_arr,
            chunk_fps=chunk_fps, s_used_local=s_used_local, hand_effort=hand_effort,
        )
        if request.args.get("wait", "").lower() == "true":
            try:
                settle_ms = float(request.args.get("settle_ms", "0"))
            except Exception:
                settle_ms = 0.0
            try:
                timeout_ms = float(request.args.get("timeout_ms", "30000"))
            except Exception:
                timeout_ms = 30000.0
            node.interp_pub.wait_chunk_done(settle_sec=max(0.0, settle_ms) / 1000.0,
                                             timeout_sec=max(0.0, timeout_ms) / 1000.0)
        return jsonify({"ok": True, **resp})

    # ---- 判断是否 whole-body chunk ----
    is_wb = any(k in d for k in ("leg", "waist", "pelvis_quat_wxyz")) \
        or ("s_used_local" in d) or (d.get("mode") == "whole_body")

    wb_response = None
    if is_wb:
        # Required for whole-body path.
        arm = d.get("arm")
        leg = d.get("leg")
        waist = d.get("waist")
        pelvis_quat_wxyz = d.get("pelvis_quat_wxyz")
        if arm is None or leg is None or waist is None or pelvis_quat_wxyz is None:
            return jsonify({
                "ok": False,
                "error": "whole-body chunk requires arm+leg+waist+pelvis_quat_wxyz",
            }), 400

        arm_arr    = np.asarray(arm, dtype=np.float64)
        leg_arr    = np.asarray(leg, dtype=np.float64)
        waist_arr  = np.asarray(waist, dtype=np.float64)
        pelvis_arr = np.asarray(pelvis_quat_wxyz, dtype=np.float64)
        for name, arr, expected in (("arm", arm_arr, 14), ("leg", leg_arr, 12),
                                    ("waist", waist_arr, 3),
                                    ("pelvis_quat_wxyz", pelvis_arr, 4)):
            if arr.ndim != 2 or arr.shape[1] != expected:
                return jsonify({
                    "ok": False,
                    "error": f"{name} expected (H, {expected}), got {arr.shape}",
                }), 400
        H = arm_arr.shape[0]
        for name, arr in (("leg", leg_arr), ("waist", waist_arr),
                          ("pelvis_quat_wxyz", pelvis_arr)):
            if arr.shape[0] != H:
                return jsonify({
                    "ok": False,
                    "error": f"{name}.shape[0]={arr.shape[0]} != arm.shape[0]={H}",
                }), 400

        hand_arr = None
        if d.get("hand") is not None:
            if str(d.get("hand_value", "raw")).lower() != "raw":
                return jsonify({
                    "ok": False,
                    "error": "whole-body hand must be actuator/raw at server boundary",
                }), 400
            hand_arr = np.asarray(d["hand"], dtype=np.float64)
            expected_hand_dim = hand_dim(node.hand_kind)
            if (hand_arr.ndim != 2
                    or hand_arr.shape != (H, expected_hand_dim)):
                return jsonify({
                    "ok": False,
                    "error": (f"hand expected ({H}, {expected_hand_dim}) actuator values, "
                              f"got {hand_arr.shape}"),
                }), 400

        # D2: neck comes from latest neck state at chunk-load time, held for
        # the full chunk. If we haven't received a neck message yet, fall back
        # to zeros (safe — head is a slow / mostly-static DOF).
        neck_joints = node.latest_neck_joints
        if neck_joints is not None and len(neck_joints.get("position", [])) >= 2:
            neck_hold = np.asarray(neck_joints["position"][:2], dtype=np.float64)
        else:
            neck_hold = np.zeros(2, dtype=np.float64)

        s_used_local = d.get("s_used_local", None)
        s_used_local = int(s_used_local) if s_used_local is not None else None
        # source_fps is the pre-upsampling policy rate. In the wholebody-human
        # 30Hz-send path the client sends the policy chunk verbatim, so
        # chunk_fps == source_fps == 30 and the stored chunk IS the policy
        # chunk. (The rescale below is kept for back-compat with an older path
        # where the client pre-upsampled to 50Hz before sending, making
        # chunk_fps=50 != source_fps=30.)
        try:
            source_fps = float(d.get("source_fps", chunk_fps))
        except (TypeError, ValueError):
            source_fps = chunk_fps
        if source_fps <= 0:
            source_fps = chunk_fps
        # s_used_local is the source-axis (30Hz) RTC value. If chunk_fps ==
        # source_fps (30Hz-send path) s_used_local_wire == s_used_local. The
        # rescale only bites on the legacy 50Hz-wire path; normal 30Hz RTC
        # callers don't pass s_used_local_wire and it collapses to s_used_local.
        s_used_local_wire = d.get("s_used_local_wire", None)
        if s_used_local_wire is not None:
            s_used_local_wire = int(s_used_local_wire)
        elif s_used_local is not None:
            s_used_local_wire = int(np.floor(s_used_local * chunk_fps / source_fps))
        chunk_id = int(d.get("chunk_id", -1) or -1)
        adaptive_transition = bool(d.get("adaptive_transition", False))

        # emit_mode selects the wire path for this chunk (overrides the
        # server default for this call). Both modes share the same wb chunk
        # buffer + atomic swap, so actual_delay is identical either way.
        #   "ta_cmd"           → 60Hz single-frame /ta/whole_body_command (MC)
        #   "reference_window" → 50Hz 10-frame /wbc/infer/reference_window (sonic)
        emit_mode = str(d.get("emit_mode") or node._wb_emit_mode or "ta_cmd")
        if emit_mode not in ("ta_cmd", "reference_window"):
            emit_mode = "ta_cmd"
        # ta_cmd_hz lets the client tune the /ta emit rate per-call (e.g. 60
        # for a 30Hz chunk, 50 to match sonic). Only applies in ta_cmd mode.
        ta_cmd_hz = d.get("ta_cmd_hz", None)
        if emit_mode == "ta_cmd":
            if ta_cmd_hz is not None:
                try:
                    node.interp_pub.set_ta_cmd_emit(
                        float(ta_cmd_hz), node._publish_ta_whole_body_command)
                    node._ta_cmd_hz = float(ta_cmd_hz)
                except Exception as e:  # noqa: BLE001
                    node.get_logger().warn(f"set_ta_cmd_emit({ta_cmd_hz}) failed: {e}")
            # stop the reference_window timer if it's running — only one
            # consumer path should be active to avoid double-driving.
            if node._ref_window_timer is not None:
                node._ref_window_timer.cancel()
                node._ref_window_timer = None
        else:  # reference_window
            # stop the ta_cmd emit thread
            node.interp_pub.set_ta_cmd_emit(0, None)
            if (node._ref_window_timer is None
                    and node._gr00t_ref_window_pub is not None):
                node._ref_window_timer = node.create_timer(
                    1.0 / node.REF_WINDOW_PUBLISH_HZ,
                    node._publish_gr00t_reference_window,
                )

        if s_used_local is not None:
            wb_response = node.interp_pub.swap_whole_body_chunk_atomic(
                leg=leg_arr, waist=waist_arr, arm=arm_arr,
                pelvis_quat=pelvis_arr, neck_hold=neck_hold,
                hand=hand_arr, hand_effort=d.get("hand_effort"),
                chunk_fps=chunk_fps, s_used_local=s_used_local_wire,
                chunk_id=chunk_id, adaptive_transition=adaptive_transition,
                source_fps=source_fps,
            )
        else:
            wb_response = node.interp_pub.set_whole_body_chunk(
                leg=leg_arr, waist=waist_arr, arm=arm_arr,
                pelvis_quat=pelvis_arr, neck_hold=neck_hold,
                hand=hand_arr, hand_effort=d.get("hand_effort"),
                chunk_fps=chunk_fps, chunk_id=chunk_id,
            )

    # Legacy hand chunks are installed independently. Whole-body hand is
    # installed and delay-sliced in the same lock as the body chunk above.
    if not is_wb and d.get("hand") is not None:
        node.interp_pub.set_hand_chunk(d["hand"], chunk_fps=chunk_fps,
                                        effort=d.get("hand_effort"))

    # legacy arm/eef 只在 non-wb 路径生效 (wb 已经吃了 arm)。
    if not is_wb:
        if d.get("arm") is not None:
            node.interp_pub.set_arm_chunk(d["arm"], chunk_fps=chunk_fps)
        if d.get("eef") is not None:
            node.interp_pub.set_eef_chunk(d["eef"], chunk_fps=chunk_fps)

    if request.args.get("wait", "").lower() == "true":
        try:
            settle_ms = float(request.args.get("settle_ms", "0"))
        except Exception:
            settle_ms = 0.0
        try:
            timeout_ms = float(request.args.get("timeout_ms", "30000"))
        except Exception:
            timeout_ms = 30000.0
        node.interp_pub.wait_chunk_done(settle_sec=max(0.0, settle_ms) / 1000.0,
                                         timeout_sec=max(0.0, timeout_ms) / 1000.0)

    if wb_response is not None:
        return jsonify({"ok": True, **wb_response})
    return jsonify({"status": "ok"})


@app.route("/push_reference_window", methods=["POST"])
def push_reference_window():
    """sim direct-50Hz replay 直发端点。

    客户端已把 episode 插值到 50Hz 且自己滑窗,这里把传入的一个(通常 10 帧)
    reference window **原样**发布到 /wbc/infer/reference_window,不做 snapshot /
    30→50Hz 插值 / RTC。收到后立即 publish,发布节拍完全由客户端决定。

    body: leg(H,12) waist(H,3) arm(H,14) pelvis_quat_wxyz(H,4)
          [dq31(H,31)] [head(H,2)]
    """
    d = request.get_json() or {}
    leg = d.get("leg"); waist = d.get("waist"); arm = d.get("arm")
    pelvis = d.get("pelvis_quat_wxyz")
    if leg is None or waist is None or arm is None or pelvis is None:
        return jsonify({"ok": False,
                        "error": "requires leg+waist+arm+pelvis_quat_wxyz"}), 400
    leg_a   = np.asarray(leg, dtype=np.float64)
    waist_a = np.asarray(waist, dtype=np.float64)
    arm_a   = np.asarray(arm, dtype=np.float64)
    pel_a   = np.asarray(pelvis, dtype=np.float64)
    for name, arr, exp in (("leg", leg_a, 12), ("waist", waist_a, 3),
                           ("arm", arm_a, 14), ("pelvis_quat_wxyz", pel_a, 4)):
        if arr.ndim != 2 or arr.shape[1] != exp:
            return jsonify({"ok": False,
                            "error": f"{name} expected (H,{exp}), got {arr.shape}"}), 400
    H = leg_a.shape[0]
    for name, arr in (("waist", waist_a), ("arm", arm_a), ("pelvis_quat_wxyz", pel_a)):
        if arr.shape[0] != H:
            return jsonify({"ok": False,
                            "error": f"{name}.shape[0]={arr.shape[0]} != leg {H}"}), 400
    dq = d.get("dq31"); head = d.get("head")
    dq_a   = np.asarray(dq, dtype=np.float64) if dq is not None else None
    head_a = np.asarray(head, dtype=np.float64) if head is not None else None

    # 关掉自主 ref_window timer:direct 模式下客户端独占该 topic,避免双写竞争。
    if node._ref_window_timer is not None:
        node._ref_window_timer.cancel()
        node._ref_window_timer = None

    ok = node.publish_reference_window_raw(leg_a, waist_a, arm_a, pel_a,
                                           dq31=dq_a, head=head_a)
    if not ok:
        return jsonify({"ok": False,
                        "error": "reference_window publisher unavailable"}), 503
    return jsonify({"ok": True, "frames": int(H)})


@app.route("/push_token", methods=["POST"])
def push_token():
    """sim external-token replay 直发端点。

    把一个 64D token 原样发布到 /sonic/token_input(SonicTokenChannel),
    sonic 按其 token_io.mode(external_token_pre_fsq / external_token_post_fsq)
    解释。发布节拍完全由客户端决定。

    body: token(64,)  — pre_fsq 连续 latent 或 post_fsq 量化码。
    """
    d = request.get_json() or {}
    token = d.get("token")
    if token is None:
        return jsonify({"ok": False, "error": "requires token"}), 400
    tok_a = np.asarray(token, dtype=np.float64).reshape(-1)
    if tok_a.shape[0] != 64:
        return jsonify({"ok": False,
                        "error": f"token expected (64,), got {tok_a.shape}"}), 400

    ok = node.publish_token_raw(tok_a)
    if not ok:
        return jsonify({"ok": False,
                        "error": "sonic token publisher unavailable"}), 503
    return jsonify({"ok": True})


@app.route("/get_imu", methods=["GET"])
def get_imu():
    """Return the latest pelvis + torso IMU samples (server subscribed to
    /body_drive/{pelvis,torso}_imu/data). See A3ServerNode._imu_cb for the
    per-sample dict layout (orientation_xyzw / gravity_dir / ...)."""
    with node._imu_lock:
        pelvis = dict(node.latest_pelvis_imu) if node.latest_pelvis_imu else None
        torso = dict(node.latest_torso_imu) if node.latest_torso_imu else None
    return jsonify({"pelvis": pelvis, "torso": torso})


@app.route("/get_chunk_progress", methods=["GET"])
def get_chunk_progress():
    """Return the current wall-clock played index into the whole-body chunk.

    - "arm" key returned for legacy client compat (A3RobotInterface.get_chunk_progress
      expects an "arm" or "eef" key). Upper-body RTC uses the arm chunk's own
      played index (from interp_pub.arm_chunk_played_idx); when no arm chunk is
      active it falls back to the whole-body played index.
    - "wb" key is the same value under an explicit whole-body name.
    """
    return jsonify(node.interp_pub.chunk_played_idx())


@app.route("/measure_rtt", methods=["GET", "POST"])
def measure_rtt():
    """Trivial endpoint for RTT probing from the client (matches a2_server)."""
    return jsonify({"ok": True, "server_ns": time.monotonic_ns()})


@app.route("/cancel_chunk", methods=["POST"])
def cancel_chunk():
    node.interp_pub.cancel_chunk()
    return jsonify({"status": "ok"})


@app.route("/wait_chunk_done", methods=["POST"])
def wait_chunk_done():
    d = request.get_json(silent=True) or {}
    settle_ms = float(d.get("settle_ms", 0.0))
    timeout_ms = float(d.get("timeout_ms", 30000.0))
    node.interp_pub.wait_chunk_done(settle_sec=settle_ms / 1000.0,
                                     timeout_sec=timeout_ms / 1000.0)
    return jsonify({"status": "ok",
                    "remaining": node.interp_pub.chunk_remaining_steps()})


# ==================== 入口 ====================

def main():
    global node
    # ros2 run 会把节点 args 带进来 (`--ros-args ...`); argparse 用
    # parse_known_args 取出我们关心的, 其它转给 rclpy.init。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-kind",
        choices=[k.value for k in HandKind],
        default=os.environ.get("A3_HAND_KIND", HandKind.HAND.value),
        help="末端类型: hand=O10 灵巧手 (默认), gripper=AgiClaw 双指夹爪",
    )
    args, ros_args = parser.parse_known_args(sys.argv[1:])

    rclpy.init(args=ros_args)
    node = A3ServerNode(hand_kind=HandKind(args.hand_kind))

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.0005)
        time.sleep(0.1)

    node.publish_head(0.0, 0.0)

    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5050, threaded=True, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    node.get_logger().info("HTTP API 启动在 0.0.0.0:5050")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.interp_pub.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
