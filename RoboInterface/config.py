"""
A2 机器人配置：关节名称、OmnihandCtrl 手部转换器、初始位姿常量。
"""
import numpy as np
from typing import List, Union
from enum import Enum


# ==================== 关节名称常量 ====================

ARM_JOINT_NAMES = [
    "idx13_left_arm_joint1", "idx14_left_arm_joint2", "idx15_left_arm_joint3",
    "idx16_left_arm_joint4", "idx17_left_arm_joint5", "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1", "idx21_right_arm_joint2", "idx22_right_arm_joint3",
    "idx23_right_arm_joint4", "idx24_right_arm_joint5", "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]

HAND_JOINT_NAMES = [
    "left_thumb_rotation_0", "left_thumb_wiggles_1", "left_thumb_bent_2",
    "left_index_wiggles_0", "left_index_bent_1", "left_middle_bent",
    "left_ring_wiggles_0", "left_ring_bent_1", "left_pinky_wiggles_0", "left_pinky_bent_1",
    "right_thumb_rotation_0", "right_thumb_wiggles_1", "right_thumb_bent_2",
    "right_index_wiggles_0", "right_index_bent_1", "right_middle_bent",
    "right_ring_wiggles_0", "right_ring_bent_1", "right_pinky_wiggles_0", "right_pinky_bent_1",
]

HEAD_JOINT_NAMES = ["idx27_head_joint1", "idx28_head_joint2"]  # shake, nod

WAIST_JOINT_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]

LEG_JOINT_NAMES = [
    "idx01_left_hip_roll", "idx02_left_hip_yaw", "idx03_left_hip_pitch",
    "idx04_left_tarsus", "idx05_01_left_toe_motorA", "idx06_01_left_toe_motorB",
    "idx07_right_hip_roll", "idx08_right_hip_yaw", "idx09_right_hip_pitch",
    "idx10_right_tarsus", "idx11_01_right_toe_motorA", "idx12_01_right_toe_motorB",
]

# pose 2: 取自数据集 /home/agiuser/project/data/stamp_new/dataset-conv2lerobot/task_9081
# episode_000000 第一帧 state  (arm = state[314:328], hand = state[0:20] actuator raw)
VLA_ARM_INIT_POS_2 = [
    0.00041264898027293384, 0.8792910575866699, -0.0007064184756018221,
    -1.4997557401657104, 0.8993767499923706, -0.000868504517711699,
    0.0020450137089937925,
    -0.00023881354718469083, -0.8800125122070312, 0.00013904986553825438,
    1.500093698501587, 0.9003448486328125, -0.0005550009082071483,
    -0.00022982143855188042,
]

VLA_HAND_INIT_POS_2 = [
    3203.0, 0.0, 3996.0, 4002.0, 4000.0, 4000.0, 4057.0, 4000.0, 27.0, 4000.0,
    3190.0, 4021.0, 3995.0, 18.0, 4000.0, 3999.0, 499.0, 4000.0, 4030.0, 4000.0,
]

# pose 3: 取自数据集 /home/agiuser/project/data/single_robot_stamp_demo530_trimmed
# episode_000000 第一帧 state (arm = state[314:328], hand = state[0:20] actuator raw)
VLA_ARM_INIT_POS_3 = [
    0.0007225770386867225, 1.099545955657959, -0.0002732808643486351,
    -1.5495100021362305, 1.2495102882385254, -0.0009015182731673121,
    0.003123128553852439,
    -0.0005001756944693625, -1.1004679203033447, 0.0004937812918797135,
    1.550187110900879, 1.2501380443572998, -0.0009018679847940803,
    -0.0014910914469510317,
]

VLA_HAND_INIT_POS_3 = [
    3194.0, 3161.0, 3979.0, 4002.0, 3999.0, 3997.0, 3916.0, 3999.0, 18.0, 3999.0,
    3190.0, 4036.0, 3996.0, 0.0, 4000.0, 3999.0, 524.0, 3997.0, 3993.0, 3999.0,
]

# ==================== 初始位姿常量 ====================

VLA_ARM_INIT_POS = [
    0.00019073486328125, 1.1999311447143555, -0.00019073486328125,
    -0.09975624084472656, 0.8993282318115234, -0.0022585329265663178,
    0.0007149516283152869,
    -0.00019073486328125, -1.200312614440918, 0.00019073486328125,
    0.10013771057128906, 0.9004726409912109, 0.0035243129965938547,
    0.00032227659434135464,
]

VLA_ARM_INIT_UP_POS = [0,1.1,0,-1.55,1.25,0,0,  0,-1.1,0,1.55,1.25,0,0]

# 数据驱动初始位姿 — 由 infer_debug_ui_generic.py::load_policy_from_config 在
# 加载模型时从 <ckpt>/assets/init.parquet 第 0 帧填充. None 表示当前 ckpt 没带
# init.parquet, reset 选 Pose Data 时会 fallback 到 VLA_HAND_INIT_POS.
VLA_HAND_INIT_POS_DATA: list[float] | None = None

# 数据集起始位姿 — 取自 /home/agiuser/project/data/530/530_39_fold_rag
# episode_000000 第 0 帧 observation.state[393:413] (hand_rad 20D), 经
# OMNIHAND_LEFT/RIGHT.radians_to_actuator 转 actuator.
# 含义: thumb_rotation_0 内旋至 ~2780 让食指 / 大拇指对掌; thumb_wiggles_1
# 处于数据集起始姿; middle/ring/pinky 三指 bent 默认合上 (idx 5/7/9 = 45 ≈
# 数据里恒为 1.48 rad). 这样训练→推理初始 state 完全一致, 模型 opening
# 信号驱动 thumb_bent / index_bent 时背景姿态与训练分布对齐.
VLA_HAND_INIT_POS = [
    2780.0, 1942.0, 3954.0, 3927.0, 3995.0, 45.0, 4031.0, 45.0, 17.0, 45.0,
    2784.0, 2044.0, 3933.0,    0.0, 3995.0, 45.0,  512.0, 45.0, 3909.0, 45.0,
]

# 握拳
VLA_HAND_FIST_POS = [
    3200.0, 0, 0, 4000.0, 0, 0, 4000.0, 0, 74.0, 0,
    3200.0, 3998.0, 0, 0, 0, 0, 500.0, 0, 4000.0, 0,
]

# 测试专用: 5 指全部伸直张开。VLA_HAND_INIT_POS 是数据对齐的"指向"姿态 —
# middle/ring/pinky_bent (idx 5/7/9) 默认就是 ~45 (合上), 拿它做硬件自检的
# OPEN 端跟 FIST 比, 中指/无名/小指基本不动, 只能看到 thumb+index 在合并。
# 这里把 5 指的 bent 都置到 3995 (与 thumb_bent/index_bent 在 INIT 的伸直
# 值一致, 在已验证的电机量程内), wiggle/rotation 保持 INIT 的值不动。
VLA_HAND_TEST_OPEN_POS = [
    2780.0, 1942.0, 3954.0, 3927.0, 3995.0, 3995.0, 4031.0, 3995.0,   17.0, 3995.0,
    2784.0, 2044.0, 3933.0,    0.0, 3995.0, 3995.0,  512.0, 3995.0, 3909.0, 3995.0,
]

# AgiClaw 双指夹爪 2D 张开 / 闭合 (actuator counts, 0-4096), 对应 server 端
# hand_kind=gripper 路径; T_MoveClawRos2.py 用同一 0..4096 区间。
VLA_GRIPPER_OPEN_POS = [4096, 4096]
VLA_GRIPPER_CLOSE_POS = [0, 0]

# 手臂放下 (对应 VLA_ARM_INIT_POS 的 EEF 位姿)
VLA_LEFT_EEF_DOWN_TARGET = {
    "position": {"x": -0.005560553468324528, "y": 0.36785788160183064, "z": -0.13375086912700485},
    "orientation": {"x": -0.17569977619915533, "y": -0.5970884471766944, "z": -0.4390894657563637, "w": 0.6479316445069508},
}

VLA_RIGHT_EEF_DOWN_TARGET = {
    "position": {"x": -0.0054587497429022525, "y": -0.36727996012800196, "z": -0.13393979894599328},
    "orientation": {"x": -0.5975647471161389, "y": -0.17675552537180772, "z": 0.6468604207026908, "w": -0.43959692148933327},
}

# 手臂抬起 (对应 VLA_ARM_INIT_UP_POS 的 EEF 位姿)
VLA_LEFT_EEF_UP_TARGET = {
    "position": {"x": 0.22476240737183173, "y": 0.29574998920910367, "z": 0.1155420728939558},
    "orientation": {"x": 0.004362302149557777, "y": 0.6522669536980829, "z": 0.757897367623012, "w": -0.01097139842901468},
}

VLA_RIGHT_EEF_UP_TARGET = {
    "position": {"x": 0.2246740273287433, "y": -0.29548461679206545, "z": 0.11552760529648654},
    "orientation": {"x": 0.6521230800291534, "y": 0.004818096792193646, "z": -0.010604881505369175, "w": 0.7580236216140289},
}

# 向后兼容的别名 (旧代码默认指向 down 姿势)
VLA_LEFT_EEF_TARGET = VLA_LEFT_EEF_DOWN_TARGET
VLA_RIGHT_EEF_TARGET = VLA_RIGHT_EEF_DOWN_TARGET


# ==================== RPC URL 配置 ====================
# 注意：手臂和手部的 RPC 已废弃，只保留头部/移动/腰部的 RPC

DEFAULT_ROBOT_IP = "192.168.100.100"

RPC_URLS = {
    "head":  "http://{ip}:56322/rpc/aimdk.protocol.McMotionService/SetNeckCommand",
    "loco":  "http://{ip}:56322/channel/%2Fmotion%2Fcontrol%2Flocomotion_velocity/pb%3Aaimdk.protocol.McLocomotionVelocityChannel",
    "waist": "http://{ip}:56322/channel/%2Fmotion%2Fcontrol%2Fmove_waist/pb%3Aaimdk.protocol.McMoveWaistChannel",
    "get_head": "http://{ip}:56322/rpc/aimdk.protocol.McDataService/GetNeckState",
}


def get_rpc_urls(robot_ip: str = DEFAULT_ROBOT_IP) -> dict:
    return {k: v.format(ip=robot_ip) for k, v in RPC_URLS.items()}


# ==================== 四元数球面线性插值 ====================

def slerp(q0, q1, t):
    """四元数球面线性插值 [x, y, z, w]"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return (1 - t) * q0 + t * q1
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta


# ==================== OmnihandCtrl ====================

class OmnihandCtrl:
    """Omnihand 控制器：主动关节弧度 <-> 执行器输入 (0-4096) 的转换。"""

    def __init__(self, hand_type: bool = False):
        """
        :param hand_type: True = 左手, False = 右手
        """
        self.hand_type = hand_type
        self.active_joint_count = 10

        # 右手基准参数
        self.active_joint_max = [1.12, 0.05, 0.8416, 0, 1.48, 1.48, 0.17, 1.48, 0.19, 1.48]
        self.motor_max = [1.12, 0.05, 1.33, 0, 1.43, 1.43, 0.17, 1.43, 0.19, 1.43]
        self.active_joint_min = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motor_min = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_pos_direction = [-1, -1, -1, -1, 1, 1, -1, 1, -1, 1]

        if not hand_type:  # 右手
            self.actuator_max = [0, 0, 0, 0, 0, 0, 4096, 0, 0, 0]
            self.actuator_min = [4096, 4096, 4096, 4096, 4096, 4096, 0, 4096, 4096, 4096]
        else:  # 左手
            left_active_joint_max = self.active_joint_max.copy()
            left_active_joint_min = self.active_joint_min.copy()
            left_motor_max = self.motor_max.copy()
            left_motor_min = self.motor_min.copy()
            for i in range(10):
                if self.left_pos_direction[i] == -1:
                    left_active_joint_max[i] = -self.active_joint_min[i]
                    left_active_joint_min[i] = -self.active_joint_max[i]
                    left_motor_max[i] = -self.motor_min[i]
                    left_motor_min[i] = -self.motor_max[i]
            self.active_joint_min = left_active_joint_min
            self.active_joint_max = left_active_joint_max
            self.motor_max = left_motor_max
            self.motor_min = left_motor_min
            self.actuator_max = [4095, 0, 4095, 0, 0, 0, 4095, 0, 0, 0]
            self.actuator_min = [0, 4095, 0, 4095, 4095, 4095, 0, 4095, 4095, 4095]

        # 多项式系数
        self.finger_mcp2motor_poly = [
            0.00944480234881967, 0.455882677008572, 0.683758090072141,
            -0.916673507519311, 0.459387725400186,
        ]
        self.finger_motor2mcp_poly = [
            -0.000257594494466942, 1.57144033291557, 0.217395210463076,
            -0.768328304426314, 0.248168989312469,
        ]
        self.right_thumb_mcp2motor_poly = [
            0.00126371020922368, 0.919140692758276, 0.550958572722048,
            -0.785384985903032, 1.25635285116862,
        ]
        self.left_thumb_mcp2motor_poly = [
            -0.00126371020922368, 0.919140692758276, -0.550958572722048,
            -0.785384985903032, -1.25635285116862,
        ]
        self.right_thumb_motor2mcp_poly = [
            -0.000677604838762652, 1.05175893483608, -0.280133575638901,
            -0.115384415912668, 0.0676128925382166,
        ]
        self.left_thumb_motor2mcp_poly = [
            0.000677604838762652, 1.05175893483608, 0.280133575638901,
            -0.115384415912668, -0.0676128925382166,
        ]

    @staticmethod
    def _polyval(x: float, coeffs: List[float]) -> float:
        result = 0.0
        power = 1.0
        for c in coeffs:
            result += c * power
            power *= x
        return result

    def _clamp(self, pos: List[float]) -> List[float]:
        for i in range(self.active_joint_count):
            pos[i] = max(self.active_joint_min[i], min(self.active_joint_max[i], pos[i]))
        return pos

    def radians_to_actuator(self, active_joint_pos: Union[List[float], np.ndarray]) -> List[int]:
        """将主动关节弧度转换为执行器输入 (0-4096)"""
        assert len(active_joint_pos) == self.active_joint_count
        hand_pos = list(active_joint_pos)
        self._clamp(hand_pos)

        if not self.hand_type:
            hand_pos[2] = self._polyval(hand_pos[2], self.right_thumb_mcp2motor_poly)
        else:
            hand_pos[2] = self._polyval(hand_pos[2], self.left_thumb_mcp2motor_poly)

        for idx in [4, 5, 7, 9]:
            hand_pos[idx] = self._polyval(hand_pos[idx], self.finger_mcp2motor_poly)

        actuator_input = []
        for i in range(self.active_joint_count):
            denom = self.motor_max[i] - self.motor_min[i]
            if abs(denom) < 1e-9:
                mapped = self.actuator_min[i]
            else:
                mapped = ((hand_pos[i] - self.motor_min[i])
                          * (self.actuator_max[i] - self.actuator_min[i]) / denom
                          + self.actuator_min[i])
            actuator_input.append(int(mapped))
        return actuator_input

    def actuator_to_radians(self, actuator_input: Union[List[int], np.ndarray]) -> List[float]:
        """将执行器输入 (0-4096) 转换为主动关节弧度"""
        assert len(actuator_input) == self.active_joint_count
        hand_input = list(actuator_input)

        motor_pos = []
        for i in range(self.active_joint_count):
            denom = self.actuator_max[i] - self.actuator_min[i]
            if abs(denom) < 1e-9:
                pos = self.motor_min[i]
            else:
                pos = ((hand_input[i] - self.actuator_min[i])
                       * (self.motor_max[i] - self.motor_min[i]) / denom
                       + self.motor_min[i])
            motor_pos.append(pos)

        if not self.hand_type:
            motor_pos[2] = self._polyval(motor_pos[2], self.right_thumb_motor2mcp_poly)
        else:
            motor_pos[2] = self._polyval(motor_pos[2], self.left_thumb_motor2mcp_poly)

        for idx in [4, 5, 7, 9]:
            motor_pos[idx] = self._polyval(motor_pos[idx], self.finger_motor2mcp_poly)

        self._clamp(motor_pos)
        return motor_pos


# 预创建的左右手控制器实例
OMNIHAND_LEFT = OmnihandCtrl(hand_type=True)
OMNIHAND_RIGHT = OmnihandCtrl(hand_type=False)
