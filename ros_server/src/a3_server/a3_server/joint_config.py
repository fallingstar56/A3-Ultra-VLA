"""A3 关节名称常量。

A3 与 A2 在控制 topic 上完全一致，关节命名按机上实测 (`ros2 topic echo
/motion/control/{arm,leg,waist,neck,hand}_joint_state`) 对齐。

末端执行器有两种型号：
  - O10 灵巧手 (`HandKind.HAND`)：20D，名称按 a3u_tool/T_MoveHandRos2.py
    的 `left/thumb_roration_pos_0` 这种 `left/...pos_X` 风格 (注意 `roration`
    是机上既有拼写)，frame_id="O10Hand"。
  - AgiClaw 双指夹爪 (`HandKind.GRIPPER`)：2D `left_claw / right_claw`，
    frame_id="AgiClaw" (参考 a3u_tool/T_MoveClawRos2.py)。

工程层在初始化 server 时通过 ``hand_kind`` 选择，二选一。
"""
from enum import Enum


class HandKind(str, Enum):
    HAND = "hand"
    GRIPPER = "gripper"


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# O10 灵巧手 20D，按 T_MoveHandRos2.py 实测可下发的 name list (`left/...pos_X` 风格)。
HAND_JOINT_NAMES = [
    "left/thumb_roration_pos_0", "left/thumb_wiggles_pos_1", "left/thumb_bent_pos_2",
    "left/index_wiggles_pos_0", "left/index_bent_pos_1", "left/middle_bent_pos",
    "left/ring_wiggles_pos_0", "left/ring_bent_pos_1",
    "left/pinky_wiggles_pos_0", "left/pinky_bent_pos_1",
    "right/thumb_roration_pos_0", "right/thumb_wiggles_pos_1", "right/thumb_bent_pos_2",
    "right/index_wiggles_pos_0", "right/index_bent_pos_1", "right/middle_bent_pos",
    "right/ring_wiggles_pos_0", "right/ring_bent_pos_1",
    "right/pinky_wiggles_pos_0", "right/pinky_bent_pos_1",
]
HAND_FRAME_ID = "O10Hand"

# AgiClaw 双指夹爪 2D。
GRIPPER_JOINT_NAMES = ["left_claw", "right_claw"]
GRIPPER_FRAME_ID = "AgiClaw"


def hand_name_list(kind: HandKind):
    return list(GRIPPER_JOINT_NAMES if kind == HandKind.GRIPPER else HAND_JOINT_NAMES)


def hand_frame_id(kind: HandKind):
    return GRIPPER_FRAME_ID if kind == HandKind.GRIPPER else HAND_FRAME_ID


def hand_dim(kind: HandKind) -> int:
    return len(GRIPPER_JOINT_NAMES) if kind == HandKind.GRIPPER else len(HAND_JOINT_NAMES)


LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
