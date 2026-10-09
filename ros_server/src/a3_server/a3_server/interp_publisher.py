"""
InterpolationPublisher: 150Hz 插值发布器。

接收 30Hz 目标命令，内部线性插值到 150Hz 后发布到 ROS Topic。
支持 arm (14D)、hand (20D)、eef (14D) 三种发布通道。
提供 raw 接口绕过插值直接发布（向后兼容）。

性能优化:
  - 预分配 ROS 消息对象，每 tick 只修改 position 字段
  - 插值计算在 lock 内，publish 在 lock 外
  - 使用 time.monotonic 和 collections.deque 减少开销
  - PI 增益加大 (Kp=2.0, Ki=0.5) 快速收敛到目标频率
"""

import time
import threading
import numpy as np
from collections import deque
from typing import Optional


def _slerp_quat_wxyz_batch(quat_wxyz_chunk: np.ndarray, t_samples: np.ndarray) -> np.ndarray:
    """球面线性插值(SLERP)一个 wxyz 四元数 chunk 到给定采样点。

    用途: 把 30Hz policy waypoint chunk 插值到 50Hz 参考窗时间轴。pelvis 姿态是
    SO(3) 元素,线性插值会偏离球面并产生非单位四元数 → 重锚/发送时 pelvis 跳变。
    SLERP 保证插值结果始终在单位球面上。numpy 手写,不引入 scipy 依赖
    (a3_server 的 package.xml 只声明 rclpy/sensor_msgs 等系统依赖)。

    Args:
        quat_wxyz_chunk: (H, 4) float64/float32,Hamilton wxyz,每行单位四元数。
        t_samples: (N,) float,采样位置(以 chunk 内 waypoint 索引为单位,0=chunk[0],
            可以是小数,允许超出 [0, H-1] —— 超出端点 hold 首尾)。
    Returns:
        (N, 4) float64 wxyz 单位四元数。

    实现要点:
        - 双覆盖处理: 相邻 chunk quat 的内积 < 0 时翻转 q1 符号,保证走最短弧。
          注意是"逐段翻转",即对每对 (q0=chunk[k], q1=chunk[k+1]) 独立判 dot,
          不是对整条 chunk 做半球对齐 —— 否则跨过 dot<0 的边界会绕长弧。
        - 端点 hold: t<=0 返回 chunk[0],t>=(H-1) 返回 chunk[-1]。
        - 归一化: 浮点误差会让结果略偏离 1,末尾归一化兜底。
    """
    q = np.asarray(quat_wxyz_chunk, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_wxyz_chunk must be (H, 4), got {q.shape}")
    t = np.asarray(t_samples, dtype=np.float64).reshape(-1)
    H = q.shape[0]
    if H == 0:
        raise ValueError("empty quat chunk")
    out = np.zeros((t.shape[0], 4), dtype=np.float64)

    # 端点 hold (t<=0 → chunk[0], t>=H-1 → chunk[-1])。
    lo = t <= 0
    hi = t >= (H - 1)
    mid = ~(lo | hi)
    out[lo] = q[0]
    if H >= 2:
        out[hi] = q[H - 1]
    else:
        out[hi] = q[0]
    if not np.any(mid):
        # 全部落在端点,直接返回(hold 首尾)。
        return out

    # 中间段: floor(t) → k, t-k → alpha,在 (q[k], q[k+1]) 间 SLERP。
    t_mid = t[mid]
    k = np.clip(np.floor(t_mid).astype(np.int64), 0, H - 2)
    alpha = t_mid - k
    q0 = q[k]                # (M, 4)
    q1 = q[k + 1]
    # 最短弧: 若 dot(q0,q1) < 0 翻转 q1。
    dot = np.sum(q0 * q1, axis=-1)               # (M,)
    flip = dot < 0.0
    q1 = np.where(flip[:, None], -q1, q1)
    dot = np.where(flip, -dot, dot)
    # theta = arccos(clamp(dot, -1, 1));若 sin(theta) ~ 0(两 quat 接近)→ 线性插值即可。
    dot_c = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot_c)                     # (M,)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-9
    # 一般段: SLERP
    #   s0 = sin((1-alpha)*theta)/sin(theta)
    #   s1 = sin(alpha*theta)/sin(theta)
    a = alpha
    s0 = np.sin((1.0 - a) * theta) / np.where(near, 1.0, sin_theta)
    s1 = np.sin(a * theta) / np.where(near, 1.0, sin_theta)
    # 接近段退化为归一化线性插值(nlerp): (1-alpha)*q0 + alpha*q1 再归一。
    s0 = np.where(near, 1.0 - a, s0)
    s1 = np.where(near, a, s1)
    res = s0[:, None] * q0 + s1[:, None] * q1    # (M, 4)
    # 归一化兜底。
    n = np.linalg.norm(res, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    res = res / n
    out[np.where(mid)[0]] = res
    return out


from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
    QoSDurabilityPolicy, QoSLivelinessPolicy,
)
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose
from joint_msgs.msg import JointCommand, Command as JointCommandEntry

from a3_server.joint_config import (
    ARM_JOINT_NAMES, HandKind, hand_name_list, hand_frame_id, hand_dim,
)


def _create_qos(depth=1):
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        liveliness=QoSLivelinessPolicy.AUTOMATIC,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# 预计算的常量
_ARM_NAMES = list(ARM_JOINT_NAMES)
_ARM_COMMAND_TOPIC = "/motion/control/pnc/arm_joint_command"
# A3 in-place upper-body PNC gains, matched to the active robot config at
# /opt/agibot/config/teleop_avatar/em_aimrt_teleop_avatar.yaml.
# Order is left arm 7 + right arm 7 and matches ARM_JOINT_NAMES.
_ARM_STIFFNESS = (
    260.0, 260.0, 117.0, 143.0, 117.0, 52.0, 52.0,
    260.0, 260.0, 117.0, 143.0, 117.0, 52.0, 52.0,
)
_ARM_DAMPING = (
    7.0, 7.0, 5.0, 5.0, 3.5, 2.0, 2.0,
    7.0, 7.0, 5.0, 5.0, 3.5, 2.0, 2.0,
)
if not (len(_ARM_NAMES) == len(_ARM_STIFFNESS) == len(_ARM_DAMPING)):
    raise RuntimeError("A3 arm names/KP/KD length mismatch")


class InterpolationPublisher:
    """150Hz 插值发布器，包装 ROS arm/hand/eef 发布。

    hand_kind 决定末端 (O10Hand / AgiClaw) 的 name list 和 frame_id，
    dim 也跟着变 (20 或 2)。
    """

    # 默认 effort：O10 手 0–255 量纲，AgiClaw 同步用 100 (T_MoveClawRos2 默认)。
    DEFAULT_HAND_EFFORT = 100.0

    def __init__(self, node: Node, target_fps: float = 150.0, interp_steps: int = 5,
                 hand_kind: HandKind = HandKind.HAND):
        self.node = node
        self.target_fps = target_fps
        self.interp_steps = interp_steps
        self.hand_kind = HandKind(hand_kind)
        self.hand_dim = hand_dim(self.hand_kind)
        self._hand_names = hand_name_list(self.hand_kind)
        self._hand_frame_id = hand_frame_id(self.hand_kind)

        qos = _create_qos()

        # ROS 发布器
        self.pub_arm = node.create_publisher(
            JointCommand, _ARM_COMMAND_TOPIC, qos_profile=qos)
        self.pub_hand = node.create_publisher(
            JointState, "/motion/control/hand_joint_command", qos_profile=qos)
        self.pub_eef = node.create_publisher(
            PoseArray, "/pnc_arm/move_eef_pose", 10)

        # 预分配 ROS 消息 (避免每 tick 创建新对象)
        # arm: /motion/control/pnc/arm_joint_command 是 joint_msgs/JointCommand,
        # 不是旧通道的 sensor_msgs/JointState。每轴携带同一帧 sequence,
        # position + zero velocity/effort, 以及机上 upper-body teleop 当前的 KP/KD。
        self._arm_msg = JointCommand()
        self._arm_sequence = 0
        self._arm_entries = []
        for name, stiffness, damping in zip(
                _ARM_NAMES, _ARM_STIFFNESS, _ARM_DAMPING):
            entry = JointCommandEntry()
            entry.name = name
            entry.velocity = 0.0
            entry.effort = 0.0
            entry.stiffness = stiffness
            entry.damping = damping
            self._arm_entries.append(entry)
        self._arm_msg.joints = self._arm_entries

        self._hand_msg = JointState()
        self._hand_msg.header.frame_id = self._hand_frame_id
        self._hand_msg.name = list(self._hand_names)

        self._eef_msg = PoseArray()
        self._eef_msg.header.frame_id = "base_link"
        self._eef_msg.poses = [Pose(), Pose()]

        # 插值状态 (受 _lock 保护)
        self._lock = threading.Lock()

        # arm (float64 numpy arrays)
        self._arm_start = None
        self._arm_target = None
        self._arm_current = None
        self._arm_step = 0
        self._arm_total = 0
        # chunk 模式: 整 chunk 排队, 每跑完一帧自动推进到下一帧
        self._arm_chunk = None        # np.ndarray (H, 14) 或 None
        self._arm_chunk_idx = 0       # 下一个要装载的 waypoint 索引
        self._arm_chunk_steps = 0     # 每个 waypoint 用多少 150Hz tick (= round(150/chunk_fps))

        # hand
        self._hand_start = None
        self._hand_target = None
        self._hand_current = None
        self._hand_step = 0
        self._hand_total = 0
        self._hand_effort = None  # dim D effort (0-255), None means default (100)
        self._hand_chunk = None
        self._hand_chunk_idx = 0
        self._hand_chunk_steps = 0
        self._hand_chunk_effort = None   # 整 chunk 共用一个 effort

        # eef
        self._eef_start = None
        self._eef_target = None
        self._eef_current = None
        self._eef_step = 0
        self._eef_total = 0
        self._eef_chunk = None
        self._eef_chunk_idx = 0
        self._eef_chunk_steps = 0

        # waist (UPPER-BODY chunk path). Parallel to arm/hand/eef, same
        # step-counter interpolation model. When an upper-body chunk installs a
        # waist trajectory, the 150Hz loop interpolates _waist_current toward
        # each waypoint and publishes MotionControlMoveWaistChannel per frame
        # (decimated to ~50Hz) via node.publish_waist. 4D [yaw, roll, pitch,
        # height]. Legacy /send_waist (one-shot) does NOT touch this state, and
        # the whole-body path never sets it, so both remain unaffected.
        self._waist_start = None
        self._waist_target = None
        self._waist_current = None
        self._waist_step = 0
        self._waist_total = 0
        self._waist_chunk = None       # np.ndarray (H, 4) or None
        self._waist_chunk_idx = 0
        self._waist_chunk_steps = 0

        # ---- Whole-body chunk buffer (for /wbc/infer/reference_window) ----
        # This is a PARALLEL state to _arm_chunk / _hand_chunk / _eef_chunk.
        # When a sonic client sends a whole-body chunk via /send_chunk (with
        # pelvis_quat_wxyz / s_used_local), we store the assembled 31D q + dq
        # matrices HERE and start publishing TaWholeBodyReferenceWindow at
        # ~50Hz to /wbc/infer/reference_window. sonic subscribes to that topic on
        # the A3 side — nothing else on the a3_server needs to actually drive
        # the motors; sonic handles all downstream control.
        #
        # BODY_31 layout:  leg(0:12) + waist(12:15) + head(15:17) + arm(17:31)
        # head slot is filled from ``neck_hold`` (snapshot of latest_neck_joints
        # at chunk-load time) so the model doesn't have to predict neck — see
        # gr00t sonic_a3_full modality (drops head idx 15/16 from body).
        self._wb_q31_chunk = None       # (H, 31) float64, absolute joint positions
        self._wb_dq31_chunk = None      # (H, 31) float64, forward-diff velocities (rad/s)
        self._wb_pelvis_chunk = None    # (H, 4)  float64, quat_wxyz per frame
        self._wb_chunk_fps = 0.0        # spacing (Hz) between adjacent waypoints WITHIN the chunk
        self._wb_chunk_start_t = 0      # wall-clock ns at chunk load — same math as _arm_chunk_start_t
        self._wb_chunk_transition_ns = 0  # first-segment transition (for adaptive_transition)
        self._wb_chunk_id = -1
        # Monotonically-increasing per-message seq for TaWholeBodyReferenceWindow.header.
        self._wb_ref_win_seq = 0
        # /wbc/infer/reference_window 发布频率(50Hz)与窗长(10)。wb chunk 以 chunk_fps
        # (30Hz,来自 client policy 输出)存为 waypoint;50Hz 发布时需要把 30Hz
        # waypoint 插值到 50Hz 时间轴再取 10 帧。chunk_fps <= ref_window_hz 时
        # 也走插值路径(等价于最近邻),统一一条代码。
        self._wb_ref_window_hz = 50.0
        self._wb_ref_window_len = 10

        # ---- /ta/whole_body_command per-frame emit thread ----
        # Same wb chunk buffer, but instead of (or in addition to) the 50Hz
        # 10-frame reference_window, a 60Hz thread snapshots the single frame
        # at the current played position and hands it to the server node to
        # build+publish a TaWholeBodyCommandChannel on /ta/whole_body_command.
        # The chunk arrives at chunk_fps (VLA default 20Hz); this thread
        # linearly interpolates between adjacent waypoints to 60Hz, matching
        # the TA native whole-body cadence (proto comment: 60Hz). dq is
        # forward-diff at the chunk's native spacing (NOT the 60Hz spacing —
        # the reference_window path does the same; MC/sonic consume it as a
        # per-waypoint velocity feedforward).
        self._ta_cmd_hz = 0.0           # 0 = thread off; >0 = run at this Hz
        self._ta_cmd_cb = None          # server-node callback: (q31(31,), dq31(31,), pelvis(4,), seq, chunk_id) -> None
        self._ta_cmd_seq = 0
        self._ta_cmd_thread = None

        # Paused clock for chunk timing (matches _arm_chunk_start_t semantics).
        # Kept as a placeholder here even though a3_server's pause path is
        # currently minimal — the atomic-swap logic (ported from a2_server)
        # respects it and having it here keeps math consistent.
        self._paused = False
        self._pause_at_ns = 0

        # PI 频率控制 — 增大增益，快速收敛
        self._fps_kp = 2.0
        self._fps_ki = 0.5
        self._fps_window = 30
        self._frame_ts = deque(maxlen=self._fps_window + 1)  # 存纳秒时间戳

        # 命令接收频率统计 (仅统计 set_*_target 调用)
        self._cmd_ts = deque(maxlen=61)
        self._cmd_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._interp_loop, daemon=True)
        self._thread.start()
        node.get_logger().info(
            f"InterpolationPublisher 启动: target_fps={target_fps}, interp_steps={interp_steps}")

    # ==================== 插值完成状态查询 ====================

    def arm_remaining_steps(self) -> int:
        """返回 arm 剩余未发布的插值步数 (0 表示已发完)。"""
        with self._lock:
            if self._arm_target is None:
                return 0
            return max(0, self._arm_total - self._arm_step)

    def hand_remaining_steps(self) -> int:
        with self._lock:
            if self._hand_target is None:
                return 0
            return max(0, self._hand_total - self._hand_step)

    def eef_remaining_steps(self) -> int:
        with self._lock:
            if self._eef_target is None:
                return 0
            return max(0, self._eef_total - self._eef_step)

    def wait_arm_done(self, settle_sec: float = 0.0, timeout_sec: float = 2.0):
        """阻塞直到 arm 插值发布完成 (可选再 sleep settle_sec 等电机跟上)。"""
        deadline = time.monotonic() + timeout_sec
        while self.arm_remaining_steps() > 0:
            if time.monotonic() > deadline:
                break
            time.sleep(1.0 / self.target_fps)
        if settle_sec > 0:
            time.sleep(settle_sec)

    def wait_hand_done(self, settle_sec: float = 0.0, timeout_sec: float = 2.0):
        deadline = time.monotonic() + timeout_sec
        while self.hand_remaining_steps() > 0:
            if time.monotonic() > deadline:
                break
            time.sleep(1.0 / self.target_fps)
        if settle_sec > 0:
            time.sleep(settle_sec)

    def wait_eef_done(self, settle_sec: float = 0.0, timeout_sec: float = 2.0):
        deadline = time.monotonic() + timeout_sec
        while self.eef_remaining_steps() > 0:
            if time.monotonic() > deadline:
                break
            time.sleep(1.0 / self.target_fps)
        if settle_sec > 0:
            time.sleep(settle_sec)

    def wait_all_done(self, settle_sec: float = 0.0, timeout_sec: float = 2.0):
        """阻塞直到 arm/hand/eef 三个通道的插值都发布完成。"""
        deadline = time.monotonic() + timeout_sec
        while True:
            if (self.arm_remaining_steps() == 0
                    and self.hand_remaining_steps() == 0
                    and self.eef_remaining_steps() == 0):
                break
            if time.monotonic() > deadline:
                break
            time.sleep(1.0 / self.target_fps)
        if settle_sec > 0:
            time.sleep(settle_sec)

    # ==================== 动态调整发送频率 ====================

    def set_send_fps(self, send_fps: float):
        """根据实际发送频率动态调整插值步数。

        例: send_fps=10 → interp_steps = round(150/10) = 15
            send_fps=30 → interp_steps = round(150/30) = 5
        """
        new_steps = max(1, int(round(self.target_fps / send_fps)))
        old_steps = self.interp_steps
        self.interp_steps = new_steps
        self.node.get_logger().info(
            f"[InterpolationPublisher] send_fps={send_fps:.1f} → interp_steps: {old_steps} → {new_steps}")

    # ==================== 设置目标 (30Hz 调用) ====================

    def _log_cmd_rate(self):
        """收到本地命令时, 每 60 帧打印一次 ROS topic 实际发布频率。仅由 set_arm_target 调用。"""
        self._cmd_ts.append(time.monotonic_ns())
        self._cmd_count += 1
        if self._cmd_count % 60 == 0:
            ts = self._frame_ts
            n = len(ts)
            if n >= 2:
                dt_ns = ts[-1] - ts[0]
                if dt_ns > 0:
                    pub_fps = (n - 1) * 1e9 / dt_ns
                    cmd_ts = self._cmd_ts
                    cmd_dt = cmd_ts[-1] - cmd_ts[0]
                    cmd_fps = (len(cmd_ts) - 1) * 1e9 / cmd_dt if cmd_dt > 0 else 0
                    self.node.get_logger().info(
                        f"[InterpolationPublisher] publish_fps={pub_fps:.1f}, command_fps={cmd_fps:.1f}")

    def set_arm_target(self, values):
        self._log_cmd_rate()
        target = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._arm_start = self._arm_current.copy() if self._arm_current is not None else target.copy()
            self._arm_target = target
            self._arm_step = 0
            self._arm_total = self.interp_steps

    def _hand_start_for_resume_locked(self, target):
        """Use live hand feedback when publishing resumes after cancellation."""
        if self._hand_target is None:
            feedback = getattr(self.node, "latest_hand_joints", None)
            positions = feedback.get("position") if isinstance(feedback, dict) else None
            if positions is not None:
                measured = np.asarray(positions, dtype=np.float64)
                if measured.shape == target.shape:
                    return measured.copy()
        if self._hand_current is not None:
            return self._hand_current.copy()
        return target.copy()

    def set_hand_target(self, values, effort=None):
        target = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._hand_start = self._hand_start_for_resume_locked(target)
            self._hand_target = target
            self._hand_step = 0
            self._hand_total = self.interp_steps
            if effort is not None:
                self._hand_effort = [float(x) for x in effort]

    def set_eef_target(self, values):
        target = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._eef_start = self._eef_current.copy() if self._eef_current is not None else target.copy()
            self._eef_target = target
            self._eef_step = 0
            self._eef_total = self.interp_steps

    # ==================== 设置 chunk (整包动作, 内部按 chunk_fps 节奏自动推进) ====================

    def _chunk_steps_from_fps(self, chunk_fps: float) -> int:
        """根据 chunk fps 计算每个 waypoint 用多少个 150Hz tick 插值完成。"""
        if chunk_fps and chunk_fps > 0:
            return max(1, int(round(self.target_fps / float(chunk_fps))))
        return self.interp_steps

    def set_arm_chunk(self, values_2d, chunk_fps: float = 30.0):
        """装载一整 chunk arm 动作 (shape [H, 14]), 内部按 chunk_fps 节奏自动推进。"""
        chunk = np.asarray(values_2d, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"arm chunk must be 2D, got shape {chunk.shape}")
        steps = self._chunk_steps_from_fps(chunk_fps)
        with self._lock:
            self._arm_chunk = chunk
            self._arm_chunk_idx = 0
            self._arm_chunk_steps = steps
            # 立刻装载第一帧, 下一个 tick 起就开始插值过去
            self._arm_start = (self._arm_current.copy()
                               if self._arm_current is not None else chunk[0].copy())
            self._arm_target = chunk[0].copy()
            self._arm_step = 0
            self._arm_total = steps
            self._arm_chunk_idx = 1

    # ==================== Whole-body chunk (for /wbc/infer/reference_window) ====================
    #
    # Parallel to the arm/hand/eef chunk paths above. sonic-a3 subscribes to
    # /wbc/infer/reference_window (TaWholeBodyReferenceWindow, 10 future frames)
    # and drives its own low-level control — a3_server just needs to:
    #   1. accept a whole-body chunk (leg / waist / arm / pelvis_quat + neck-hold)
    #   2. optionally do server-atomic swap using s_used_local (ported from
    #      a2_server.swap_arm_chunk_atomic — no HTTP RTT gap)
    #   3. snapshot 10 frames on demand for the ROS publisher
    # The 150Hz interp loop below is UNTOUCHED — sonic-mode doesn't drive the
    # legacy /motion/control/*_command publishers.
    #
    # BODY_31 layout:  leg(0:12) + waist(12:15) + head(15:17) + arm(17:31).
    # head slot holds ``neck_hold`` — the model didn't predict neck (see gr00t
    # sonic_a3_full modality.json), so it's held constant across the chunk →
    # head_velocity ≡ 0 after forward-diff.

    @staticmethod
    def _build_q31_dq31(leg, waist, arm, neck_hold, chunk_fps):
        H = int(leg.shape[0])
        q31 = np.zeros((H, 31), dtype=np.float64)
        q31[:, 0:12]  = leg
        q31[:, 12:15] = waist
        q31[:, 15:17] = neck_hold[None, :]
        q31[:, 17:31] = arm
        dq31 = np.zeros_like(q31)
        if H >= 2 and chunk_fps > 0:
            dt_s = 1.0 / chunk_fps
            dq31[:-1] = (q31[1:] - q31[:-1]) / dt_s
            dq31[-1] = dq31[-2]
        return q31, dq31

    def _wb_swap_locked(self, q31, dq31, pelvis_quat, chunk_fps,
                       s_used_local, transition_ns):
        """Install (q31, dq31, pelvis) and return the front-slice delay in the
        stored chunk's own frames.

        ``actual_delay = max(0, ceil(played_now_of_OLD_chunk) - s_used_local)``.

        NOTE (wholebody-human / 30Hz-send): the stored chunk is now the policy
        chunk itself (chunk_fps == source_fps == 30Hz), so this "wire" delay is
        already on the 30Hz policy axis — server_node's public ``actual_delay``
        equals it (the ceil(delay * source_fps/chunk_fps) rescale is identity).
        The 30→50Hz interpolation happens ONLY at reference-window snapshot time
        (wb_snapshot_for_reference_window); it does NOT change the stored chunk
        or this delay. ``played_now`` is computed from wall-clock
        (now - _wb_chunk_start_t) * _wb_chunk_fps — the SAME formula the 50Hz
        snapshot and wb_played_idx use, so client-reported s_used_local and this
        server-side played_now share one timeline.
        """
        actual_delay_wire = 0
        if (s_used_local is not None
                and self._wb_q31_chunk is not None
                and self._wb_chunk_fps > 0):
            now_ns = self._pause_at_ns if self._paused else time.monotonic_ns()
            t_ns = now_ns - self._wb_chunk_start_t
            old_max = float(self._wb_q31_chunk.shape[0] - 1)
            wp_ns = int(1e9 / self._wb_chunk_fps)
            trans_ns = self._wb_chunk_transition_ns
            if trans_ns > 0 and trans_ns != wp_ns:
                if t_ns < trans_ns:
                    played_now = -1.0 + t_ns / trans_ns
                else:
                    played_now = min((t_ns - trans_ns) * self._wb_chunk_fps / 1e9, old_max)
            else:
                played_now = min(-1.0 + t_ns * self._wb_chunk_fps / 1e9, old_max)
            actual_delay_wire = max(
                0, int(np.ceil(played_now)) - int(s_used_local)
            )
            actual_delay_wire = min(
                actual_delay_wire, max(0, q31.shape[0] - 1)
            )
            if actual_delay_wire > 0:
                q31 = q31[actual_delay_wire:]
                dq31 = dq31[actual_delay_wire:]
                pelvis_quat = pelvis_quat[actual_delay_wire:]

        self._wb_q31_chunk = q31
        self._wb_dq31_chunk = dq31
        self._wb_pelvis_chunk = pelvis_quat
        self._wb_chunk_fps = float(chunk_fps)
        self._wb_chunk_start_t = (self._pause_at_ns if self._paused
                                  else time.monotonic_ns())
        self._wb_chunk_transition_ns = int(transition_ns)
        return actual_delay_wire

    def set_whole_body_chunk(self, leg, waist, arm, pelvis_quat, neck_hold,
                             chunk_fps=30.0, chunk_id=-1,
                             hand=None, hand_effort=None):
        """Cold-start / non-RTC path — install without slicing."""
        chunk_fps = float(chunk_fps)
        wp_ns = int(1e9 / chunk_fps) if chunk_fps > 0 else int(1e9 / self.target_fps)
        q31, dq31 = self._build_q31_dq31(leg, waist, arm, neck_hold, chunk_fps)
        hand_a = np.asarray(hand, dtype=np.float64) if hand is not None else None
        steps = self._chunk_steps_from_fps(chunk_fps)
        with self._lock:
            self._wb_swap_locked(q31, dq31, pelvis_quat, chunk_fps,
                                 s_used_local=None, transition_ns=wp_ns)
            if hand_a is not None:
                self._install_hand_chunk_locked(hand_a, steps, hand_effort)
            self._wb_chunk_id = int(chunk_id)
        return {"actual_delay": 0, "actual_delay_wire": 0, "pos_skip": 0,
                "max_wps_pre_skip": None, "backstep_gated": False,
                "transition_ms": None, "jump": None, "max_wps": None}

    def swap_whole_body_chunk_atomic(self, leg, waist, arm, pelvis_quat, neck_hold,
                                     chunk_fps=30.0, s_used_local=None,
                                     chunk_id=-1, adaptive_transition=False,
                                     source_fps=None, hand=None,
                                     hand_effort=None):
        """Whole-body server-atomic swap. See a2_server.swap_arm_chunk_atomic
        for the reference-time semantics we're preserving.

        adaptive_transition defaults False on the wb path — sonic's downstream
        low-level control handles smoothing between chunks. Left as an opt-in
        for diagnostic runs; heuristic mirrors a2 (base_ms scaled by peak-wps).
        """
        chunk_fps = float(chunk_fps)
        source_fps = float(source_fps) if source_fps is not None else chunk_fps
        if source_fps <= 0:
            source_fps = chunk_fps
        wp_ns = int(1e9 / chunk_fps) if chunk_fps > 0 else int(1e9 / self.target_fps)
        q31_new, dq31_new = self._build_q31_dq31(leg, waist, arm, neck_hold, chunk_fps)
        hand_new = np.asarray(hand, dtype=np.float64) if hand is not None else None
        steps = self._chunk_steps_from_fps(chunk_fps)

        transition_ns = wp_ns
        jump_val = None
        wps_max_val = None
        JUMP_EPS = 0.05
        with self._lock:
            if (adaptive_transition and self._wb_q31_chunk is not None
                    and q31_new.shape[0] >= 2):
                base_ms = 1000.0 / chunk_fps if chunk_fps > 0 else 50.0
                # Compare new chunk[0] to the OLD chunk's current played frame.
                played_idx_old = max(0, int(np.floor(
                    (time.monotonic_ns() - self._wb_chunk_start_t)
                    * self._wb_chunk_fps / 1e9)))
                played_idx_old = min(played_idx_old,
                                     self._wb_q31_chunk.shape[0] - 1)
                cur = self._wb_q31_chunk[played_idx_old]
                step_diffs = np.abs(np.diff(q31_new, axis=0))
                speed_per_dim = step_diffs.max(axis=0)
                jumps_per_dim = np.abs(q31_new[0] - cur)
                jump_val = float(jumps_per_dim.max())
                wps_max = 1.0
                for j in range(q31_new.shape[1]):
                    if jumps_per_dim[j] < JUMP_EPS:
                        continue
                    sp = float(speed_per_dim[j])
                    if sp <= 1e-6:
                        sp = 0.087  # ~5°/wp conservative floor
                    wps = jumps_per_dim[j] / sp
                    if wps > wps_max:
                        wps_max = wps
                wps_max_val = float(wps_max)
                t_ms = max(base_ms, min(2000.0, wps_max * base_ms))
                transition_ns = int(t_ms * 1e6)

            actual_delay_wire = self._wb_swap_locked(
                q31_new, dq31_new, pelvis_quat, chunk_fps,
                s_used_local=s_used_local, transition_ns=transition_ns,
            )
            if hand_new is not None:
                hand_sliced = hand_new[actual_delay_wire:]
                self._install_hand_chunk_locked(hand_sliced, steps, hand_effort)
            self._wb_chunk_id = int(chunk_id)

        # actual_delay is on the stored chunk's axis. In the wholebody-human
        # 30Hz-send path chunk_fps == source_fps == 30, so this rescale is the
        # identity and actual_delay == actual_delay_wire (both 30Hz policy
        # frames). The rescale is kept for back-compat with an older 50Hz-wire
        # send path where the stored chunk was pre-upsampled (chunk_fps=50,
        # source_fps=30). The 30→50Hz reference-window interpolation is done
        # later at snapshot time and does NOT affect this delay.
        actual_delay = int(np.ceil(actual_delay_wire * source_fps / chunk_fps))
        return {"actual_delay": actual_delay,
                "actual_delay_wire": int(actual_delay_wire), "pos_skip": 0,
                "max_wps_pre_skip": None, "backstep_gated": False,
                "transition_ms": (None if transition_ns == wp_ns
                                  else transition_ns / 1e6),
                "jump": jump_val, "max_wps": wps_max_val}

    def wb_played_idx(self):
        """Wall-clock played index into the current wb chunk. None if no chunk."""
        with self._lock:
            return self._wb_played_idx_nolock()

    def _wb_played_idx_nolock(self):
        """``wb_played_idx`` without locking; caller must hold ``self._lock``."""
        if self._wb_q31_chunk is None or self._wb_chunk_fps <= 0:
            return None
        now = self._pause_at_ns if self._paused else time.monotonic_ns()
        t_ns = now - self._wb_chunk_start_t
        fps = self._wb_chunk_fps
        max_played = float(self._wb_q31_chunk.shape[0] - 1)
        wp_ns = int(1e9 / fps)
        trans_ns = self._wb_chunk_transition_ns
        if trans_ns > 0 and trans_ns != wp_ns:
            if t_ns < trans_ns:
                return -1.0 + t_ns / trans_ns
            return min((t_ns - trans_ns) * fps / 1e9, max_played)
        return min(-1.0 + t_ns * fps / 1e9, max_played)

    def set_reference_window_params(self, hz: float, window_len: int = 10) -> None:
        """Set the /wbc/infer/reference_window publish rate + window length used by
        wb_snapshot_for_reference_window's 30→50Hz interpolation. Called once by
        the server node so the snapshot cadence matches the ROS timer that
        publishes it (single source of truth). hz<=0 falls back to 50."""
        self._wb_ref_window_hz = float(hz) if hz and hz > 0 else 50.0
        self._wb_ref_window_len = int(window_len) if window_len and window_len > 0 else 10

    def wb_snapshot_for_reference_window(self, window_len=10):
        """把当前 chunk 插值到 ref_window_hz(50Hz)时间轴并取 window_len(10)帧。

        与旧"截 waypoint"实现的关键区别: client 现在按 policy 输出频率(30Hz)
        发整 chunk,server 把这 30Hz waypoint 在锁内插值到 50Hz 时间轴再取 10 帧,
        而不是直接取最近的 30Hz waypoint(hold-last)。这样 sonic 侧收到的就是
        真正 50Hz×10 帧的 reference_window,窗内相邻帧间隔 = 1/50s = 20ms。

        插值:
          - 关节(leg/waist/arm/neck 31D q): np.interp 线性。
          - pelvis quat(4D wxyz): SLERP(球面),_slerp_quat_wxyz_batch。
            pelvis 是 SO(3) 元素,线性插值会偏离球面 → 边界跳变;必须球面。
          - dq(31D 速度): 对插好的 50Hz q_win 做窗内前向差分(dt=1/ref_hz),
            末帧 hold-last。这样 dq 与实际发出的 q_win 逐帧自洽,匹配原始
            bridge / ta_channel.proto 的"窗内前向差分,dq_9=dq_8 hold-last"语义。
            对比另外两种(都不采用):
              (a) floor 取 30Hz dq —— 段边界阶跃,且与 q_win 不自洽;
              (b) 对 30Hz dq 做 np.interp —— 是"点上瞬时值插值",不是"区间平均",
                  同样与 q_win 不逐帧自洽。
            注意:线性插值下,采样区间完全落在某个 30Hz 段内时,50Hz 前向差分
            严格等于该段的 30Hz 原生差分(直线斜率与采样密度无关);只有横跨
            30Hz 段边界的帧才得到相邻两段速度按时间占比的加权,这正是 q_win
            在该 20ms 区间的真实平均斜率。

        采样点: played 是 chunk 内的浮点 waypoint 索引(30Hz);50Hz 窗的 k 帧
        采样点 = played + k * (chunk_fps / ref_window_hz),即每帧前进
        30/50 = 0.6 个 waypoint。窗口溢出 chunk 末尾 → pelvis/joint 都 hold-last
        (_slerp/interp 的端点 clamp 处理)。

        Returns None when no chunk is loaded. transition 段(played<0)clamp 到 chunk[0]
        语义保留: 采样点 < 0 时 interp/slerp 返回 chunk[0]。
        """
        with self._lock:
            if (self._wb_q31_chunk is None or self._wb_dq31_chunk is None
                    or self._wb_pelvis_chunk is None
                    or self._wb_chunk_fps <= 0):
                return None
            now = self._pause_at_ns if self._paused else time.monotonic_ns()
            t_ns = now - self._wb_chunk_start_t
            fps = self._wb_chunk_fps
            H = self._wb_q31_chunk.shape[0]
            max_played = float(H - 1)
            wp_ns = int(1e9 / fps)
            trans_ns = self._wb_chunk_transition_ns
            if trans_ns > 0 and trans_ns != wp_ns:
                if t_ns < trans_ns:
                    played = -1.0 + t_ns / trans_ns
                else:
                    played = min((t_ns - trans_ns) * fps / 1e9, max_played)
            else:
                played = min(-1.0 + t_ns * fps / 1e9, max_played)

            ref_hz = self._wb_ref_window_hz if self._wb_ref_window_hz > 0 else 50.0
            # 50Hz 窗内第 k 帧在 chunk(30Hz)waypoint 索引轴上的采样位置。
            step = fps / ref_hz
            t_samples = played + np.arange(window_len, dtype=np.float64) * step
            # 关节线性插值: 逐列 np.interp (waypoint 索引 → value)。
            idx_src = np.arange(H, dtype=np.float64)
            q31 = self._wb_q31_chunk
            q_win = np.stack(
                [np.interp(t_samples, idx_src, q31[:, d]) for d in range(q31.shape[1])],
                axis=1,
            ).astype(np.float64)
            # dq: 对插好的 50Hz q_win 做窗内前向差分(dt = 1/ref_hz),最后一帧
            # hold-last。这样 dq 与实际发出的 q_win 自洽,且匹配原始 bridge /
            # ta_channel.proto 的语义("窗内前向差分,dt_step=1/publish_rate_hz,
            # dq_9=dq_8 hold-last")。注意:线性插值下,段内部的 50Hz 差分严格
            # 等于 30Hz 原生差分(直线斜率与采样密度无关);只有跨 30Hz 段边界的
            # 帧会得到相邻两段速度的线性混合,比取 floor 的阶跃更平滑,也才和
            # q_win 逐帧一致。
            dt_ref = 1.0 / ref_hz
            dq_win = np.zeros_like(q_win)
            if window_len >= 2:
                dq_win[:-1] = (q_win[1:] - q_win[:-1]) / dt_ref
                dq_win[-1] = dq_win[-2]
            # pelvis: SLERP(球面)。
            pelvis_win = _slerp_quat_wxyz_batch(
                self._wb_pelvis_chunk, t_samples
            ).astype(np.float64)

            self._wb_ref_win_seq = (self._wb_ref_win_seq + 1) & 0xFFFFFFFF
            seq = self._wb_ref_win_seq
            chunk_id = self._wb_chunk_id

        return {"seq": int(seq), "chunk_id": int(chunk_id),
                "played": float(played), "q31": q_win, "dq31": dq_win,
                "pelvis_wxyz": pelvis_win}

    def cancel_whole_body_chunk(self):
        with self._lock:
            self._wb_q31_chunk = None
            self._wb_dq31_chunk = None
            self._wb_pelvis_chunk = None
            self._wb_chunk_fps = 0.0
            self._wb_chunk_start_t = 0
            self._wb_chunk_transition_ns = 0
            self._wb_chunk_id = -1

    # ==================== /ta/whole_body_command per-frame emit ====================

    def set_ta_cmd_emit(self, hz: float, cb) -> None:
        """Enable/disable the 60Hz single-frame /ta/whole_body_command emit.

        hz == 0 (or <=0) stops the thread; hz > 0 (re)starts it at that rate.
        ``cb`` is invoked once per tick with
        ``(q31 (31,) float64, dq31 (31,) float64, pelvis_wxyz (4,) float64,
          seq int, chunk_id int)`` and must build+publish the
        TaWholeBodyCommandChannel off the publisher thread (keep it lean —
        proto build + 1 publish).

        Safe to call repeatedly; idempotent if already running at the same hz.
        """
        hz = float(hz)
        same = (abs(hz - self._ta_cmd_hz) < 1e-3)
        if hz <= 0:
            self._ta_cmd_hz = 0.0
            # thread reads _ta_cmd_hz under _lock each tick and exits when 0
            return
        self._ta_cmd_cb = cb
        if self._ta_cmd_thread is not None and self._ta_cmd_thread.is_alive() and same:
            self._ta_cmd_hz = hz
            return
        self._ta_cmd_hz = hz
        self._ta_cmd_thread = threading.Thread(
            target=self._ta_cmd_loop, name="wb_ta_cmd_emit", daemon=True)
        self._ta_cmd_thread.start()

    def _ta_cmd_loop(self) -> None:
        """60Hz loop: snapshot the wb chunk's played frame (interpolated) and
        hand it to the server node's publish callback.

        Interpolation: the chunk is chunk_fps-spaced (VLA default 20Hz); we
        sample at the wall-clock played index (float) and linearly blend
        q31 between floor/ceil waypoints. pelvis_quat is slerp-free nearest-
        floor (quaternion double-cover makes naive lerp sign-ambiguous; the
        pelvis target barely moves within one 50ms waypoint so nearest is
        fine and matches what the reference_window hold-last semantics do).
        dq is taken from the chunk's precomputed forward-diff (native spacing).
        """
        mono_ns = time.monotonic_ns
        sleep = time.sleep
        # busy-wait threshold: sleep jitter ~1ms, busy-wait below this
        BUSYWAIT_NS = 1_500_000
        while True:
            hz = self._ta_cmd_hz
            if hz <= 0:
                return
            interval_ns = int(1e9 / hz)
            next_wake = mono_ns()
            while True:
                if self._ta_cmd_hz <= 0:
                    return
                cb = self._ta_cmd_cb
                snap = self._wb_ta_cmd_snapshot() if cb is not None else None
                if snap is not None and cb is not None:
                    try:
                        cb(snap["q31"], snap["dq31"], snap["pelvis_wxyz"],
                           snap["seq"], snap["chunk_id"])
                    except Exception as e:  # noqa: BLE001 — never let pub kill the loop
                        self.node.get_logger().warn(
                            f"[ta_cmd_emit] publish callback error: {e}")
                next_wake += interval_ns
                now = mono_ns()
                if now > next_wake + interval_ns:
                    next_wake = now + interval_ns
                remaining = next_wake - mono_ns()
                if remaining > BUSYWAIT_NS:
                    sleep((remaining - BUSYWAIT_NS) / 1e9)
                while mono_ns() < next_wake:
                    pass

    def _wb_ta_cmd_snapshot(self) -> Optional[dict]:
        """Single interpolated frame at the current played index.

        Returns None when no chunk is loaded (caller skips the tick). During
        the first-segment transition (played < 0) clamps to chunk[0] so MC
        sees chunk[0] until the chunk actually starts.
        """
        with self._lock:
            if (self._wb_q31_chunk is None or self._wb_dq31_chunk is None
                    or self._wb_pelvis_chunk is None
                    or self._wb_chunk_fps <= 0
                    or self._ta_cmd_cb is None):
                return None
            now = self._pause_at_ns if self._paused else time.monotonic_ns()
            t_ns = now - self._wb_chunk_start_t
            fps = self._wb_chunk_fps
            H = self._wb_q31_chunk.shape[0]
            max_played = float(H - 1)
            wp_ns = int(1e9 / fps)
            trans_ns = self._wb_chunk_transition_ns
            if trans_ns > 0 and trans_ns != wp_ns:
                if t_ns < trans_ns:
                    played = -1.0 + t_ns / trans_ns
                else:
                    played = min((t_ns - trans_ns) * fps / 1e9, max_played)
            else:
                played = min(-1.0 + t_ns * fps / 1e9, max_played)

            if played < 0.0:
                # transition ramp before chunk[0]: hold chunk[0]
                q31 = self._wb_q31_chunk[0].copy()
                dq31 = self._wb_dq31_chunk[0].copy()
                pelvis = self._wb_pelvis_chunk[0].copy()
            else:
                i0 = int(np.floor(played))
                if i0 >= H - 1:
                    q31 = self._wb_q31_chunk[H - 1].copy()
                    dq31 = self._wb_dq31_chunk[H - 1].copy()
                    pelvis = self._wb_pelvis_chunk[H - 1].copy()
                else:
                    alpha = float(played - i0)
                    q31 = ((1.0 - alpha) * self._wb_q31_chunk[i0]
                           + alpha * self._wb_q31_chunk[i0 + 1]).copy()
                    # dq: native-spacing forward-diff at floor index (matches
                    # reference_window semantics; not re-diffed at 60Hz).
                    dq31 = self._wb_dq31_chunk[i0].copy()
                    pelvis = self._wb_pelvis_chunk[i0].copy()
            self._ta_cmd_seq = (self._ta_cmd_seq + 1) & 0xFFFFFFFF
            seq = self._ta_cmd_seq
            chunk_id = self._wb_chunk_id
        return {"seq": int(seq), "chunk_id": int(chunk_id),
                "q31": q31, "dq31": dq31, "pelvis_wxyz": pelvis}

    def set_hand_chunk(self, values_2d, chunk_fps: float = 30.0, effort=None):
        chunk = np.asarray(values_2d, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"hand chunk must be 2D, got shape {chunk.shape}")
        steps = self._chunk_steps_from_fps(chunk_fps)
        with self._lock:
            self._hand_chunk = chunk
            self._hand_chunk_idx = 0
            self._hand_chunk_steps = steps
            self._hand_chunk_effort = (
                [float(x) for x in effort] if effort is not None else None
            )
            self._hand_start = self._hand_start_for_resume_locked(chunk[0])
            self._hand_target = chunk[0].copy()
            self._hand_step = 0
            self._hand_total = steps
            self._hand_chunk_idx = 1
            if self._hand_chunk_effort is not None:
                self._hand_effort = self._hand_chunk_effort

    def set_eef_chunk(self, values_2d, chunk_fps: float = 30.0):
        chunk = np.asarray(values_2d, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"eef chunk must be 2D, got shape {chunk.shape}")
        steps = self._chunk_steps_from_fps(chunk_fps)
        with self._lock:
            self._eef_chunk = chunk
            self._eef_chunk_idx = 0
            self._eef_chunk_steps = steps
            self._eef_start = (self._eef_current.copy()
                               if self._eef_current is not None else chunk[0].copy())
            self._eef_target = chunk[0].copy()
            self._eef_step = 0
            self._eef_total = steps
            self._eef_chunk_idx = 1

    # ==================== UPPER-BODY chunk (arm + hand + waist) ====================
    #
    # Server-atomic swap for the upper-body mc topics, ADDED alongside (not
    # replacing) the whole-body path. Semantics mirror a2_server.swap_arm_chunk_
    # atomic: read the OLD arm chunk's played index, compute
    #     actual_delay = max(0, ceil(played_now) - s_used_local)
    # slice arm/hand/waist by the SAME actual_delay, install all three. arm/hand
    # reuse the existing 150Hz step-counter playback; waist adds a parallel
    # step-counter path (published via node.publish_waist, decimated in the loop).

    def _install_arm_chunk_locked(self, chunk, steps):
        self._arm_chunk = chunk
        self._arm_chunk_steps = steps
        self._arm_start = (self._arm_current.copy()
                           if self._arm_current is not None else chunk[0].copy())
        self._arm_target = chunk[0].copy()
        self._arm_step = 0
        self._arm_total = steps
        self._arm_chunk_idx = 1

    def _install_hand_chunk_locked(self, chunk, steps, effort=None):
        self._hand_chunk = chunk
        self._hand_chunk_steps = steps
        self._hand_chunk_effort = (
            [float(x) for x in effort] if effort is not None else None
        )
        self._hand_start = self._hand_start_for_resume_locked(chunk[0])
        self._hand_target = chunk[0].copy()
        self._hand_step = 0
        self._hand_total = steps
        self._hand_chunk_idx = 1
        if self._hand_chunk_effort is not None:
            self._hand_effort = self._hand_chunk_effort

    def _install_waist_chunk_locked(self, chunk, steps):
        self._waist_chunk = chunk
        self._waist_chunk_steps = steps
        self._waist_start = (self._waist_current.copy()
                             if self._waist_current is not None else chunk[0].copy())
        self._waist_target = chunk[0].copy()
        self._waist_step = 0
        self._waist_total = steps
        self._waist_chunk_idx = 1

    def _arm_played_now_locked(self):
        """Fractional played index into the CURRENT arm chunk (a2 semantics:
        -1 == at start pose, 0 == chunk[0], k == chunk[k]). Derived from the
        step-counter state:  played = (idx - 2) + step/total.
        Returns None when no arm chunk is active (cold-start / finished) —
        the caller then skips slicing (actual_delay = 0)."""
        if (self._arm_chunk is None or self._arm_target is None
                or self._arm_total <= 0):
            return None
        played = (self._arm_chunk_idx - 2) + (self._arm_step / self._arm_total)
        return max(-1.0, played)

    def arm_chunk_played_idx(self):
        """Public played index into the current arm chunk (float) or None.
        Used by /get_chunk_progress for the upper-body RTC client."""
        with self._lock:
            return self._arm_played_now_locked()

    def chunk_played_idx(self):
        """Return the A3 RTC progress snapshot under one interpolator lock."""
        with self._lock:
            return self._chunk_played_idx_nolock()

    def _chunk_played_idx_nolock(self):
        """``chunk_played_idx`` without locking; caller must hold ``self._lock``.

        The observation endpoint already holds this non-reentrant lock so it can
        sample robot state and progress together. Calling either public progress
        method there would acquire the same ``threading.Lock`` twice and deadlock
        the Flask worker together with the 150 Hz interpolation thread.
        """
        played_wb = self._wb_played_idx_nolock()
        played_arm = self._arm_played_now_locked()
        arm_val = played_arm if played_arm is not None else played_wb
        return {"arm": arm_val, "wb": played_wb, "eef": None, "hand": None}

    def set_waist_chunk(self, values_2d, chunk_fps: float = 30.0):
        """Install a waist trajectory (H, 4) [yaw, roll, pitch, height] for
        server-side per-frame pacing (no atomic slice)."""
        chunk = np.asarray(values_2d, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 4:
            raise ValueError(f"waist chunk must be (H, 4), got shape {chunk.shape}")
        steps = self._chunk_steps_from_fps(chunk_fps)
        with self._lock:
            self._install_waist_chunk_locked(chunk, steps)

    def swap_upper_chunk_atomic(self, arm=None, hand=None, waist=None,
                                chunk_fps: float = 30.0,
                                s_used_local=None, hand_effort=None) -> dict:
        """Atomic upper-body swap. In one lock critical section: read the OLD
        arm chunk's played index, compute actual_delay against ``s_used_local``,
        slice arm/hand/waist by that same delay, and install all three. No HTTP
        RTT gap between reading progress and installing → the sliced chunk[0]
        lines up with the robot's true position.

        Returns {"actual_delay": int}. actual_delay == 0 when s_used_local is
        None (cold-start) or no arm chunk is currently playing.
        """
        arm_a = np.asarray(arm, dtype=np.float64) if arm is not None else None
        hand_a = np.asarray(hand, dtype=np.float64) if hand is not None else None
        waist_a = np.asarray(waist, dtype=np.float64) if waist is not None else None
        steps = self._chunk_steps_from_fps(chunk_fps)
        actual_delay = 0

        with self._lock:
            # ---- Phase 1: actual_delay from OLD arm chunk played ----
            played = self._arm_played_now_locked()
            if (s_used_local is not None and played is not None
                    and arm_a is not None and arm_a.shape[0] > 0):
                ad = max(0, int(np.ceil(played)) - int(s_used_local))
                ad = min(ad, max(0, arm_a.shape[0] - 1))
                actual_delay = ad

            # ---- Phase 2: slice all three by the same actual_delay ----
            if actual_delay > 0:
                if arm_a is not None:
                    arm_a = arm_a[actual_delay:]
                if hand_a is not None:
                    hand_a = hand_a[actual_delay:]
                if waist_a is not None:
                    waist_a = waist_a[actual_delay:]

            # ---- Phase 3: install (arm/hand reuse existing playback) ----
            if arm_a is not None and arm_a.ndim == 2 and arm_a.shape[0] > 0:
                self._install_arm_chunk_locked(arm_a, steps)
            if hand_a is not None and hand_a.ndim == 2 and hand_a.shape[0] > 0:
                self._install_hand_chunk_locked(hand_a, steps, hand_effort)
            if waist_a is not None and waist_a.ndim == 2 and waist_a.shape[0] > 0:
                self._install_waist_chunk_locked(waist_a, steps)

        return {"actual_delay": int(actual_delay)}

    def cancel_chunk(self):
        """立即取消所有正在执行的 chunk and release the hand publisher.

        Also clears the whole-body chunk buffer so /wbc/infer/reference_window
        stops publishing. The hand/gripper target must become ``None`` rather
        than ``current``: whole-body hand commands use the independent MC hand
        topic, so keeping a hold target would continue publishing at 150 Hz and
        conflict with teleoperation after HIL stop.
        """
        with self._lock:
            self._arm_chunk = None
            self._arm_chunk_idx = 0
            if self._arm_current is not None:
                self._arm_start = self._arm_current.copy()
                self._arm_target = self._arm_current.copy()
            self._arm_step = 0
            self._arm_total = 0

            self._hand_chunk = None
            self._hand_chunk_idx = 0
            self._hand_chunk_effort = None
            if self._hand_current is not None:
                self._hand_start = self._hand_current.copy()
            self._hand_target = None
            self._hand_step = 0
            self._hand_total = 0

            self._eef_chunk = None
            self._eef_chunk_idx = 0
            if self._eef_current is not None:
                self._eef_start = self._eef_current.copy()
                self._eef_target = self._eef_current.copy()
            self._eef_step = 0
            self._eef_total = 0

            # Upper-body waist chunk (hold at current position).
            self._waist_chunk = None
            self._waist_chunk_idx = 0
            if self._waist_current is not None:
                self._waist_start = self._waist_current.copy()
                self._waist_target = self._waist_current.copy()
            self._waist_step = 0
            self._waist_total = 0

            # Also clear the whole-body chunk (used by /wbc/infer/reference_window).
            self._wb_q31_chunk = None
            self._wb_dq31_chunk = None
            self._wb_pelvis_chunk = None
            self._wb_chunk_fps = 0.0
            self._wb_chunk_start_t = 0
            self._wb_chunk_transition_ns = 0
            self._wb_chunk_id = -1

    def chunk_remaining_steps(self) -> int:
        """三通道剩余 (chunk + 当前段插值) 总 tick 数, 0 表示全部跑完。"""
        with self._lock:
            def _ch_left(chunk, idx, steps, total, step):
                left = max(0, total - step)
                if chunk is not None:
                    left += max(0, len(chunk) - idx) * steps
                return left
            return (_ch_left(self._arm_chunk, self._arm_chunk_idx,
                             self._arm_chunk_steps, self._arm_total, self._arm_step)
                    + _ch_left(self._hand_chunk, self._hand_chunk_idx,
                               self._hand_chunk_steps, self._hand_total, self._hand_step)
                    + _ch_left(self._eef_chunk, self._eef_chunk_idx,
                               self._eef_chunk_steps, self._eef_total, self._eef_step)
                    + _ch_left(self._waist_chunk, self._waist_chunk_idx,
                               self._waist_chunk_steps, self._waist_total, self._waist_step))

    def wait_chunk_done(self, settle_sec: float = 0.0, timeout_sec: float = 30.0):
        """阻塞直到所有 chunk 跑完。"""
        deadline = time.monotonic() + timeout_sec
        while self.chunk_remaining_steps() > 0:
            if time.monotonic() > deadline:
                break
            time.sleep(1.0 / self.target_fps)
        if settle_sec > 0:
            time.sleep(settle_sec)

    # ==================== 直接发布 (绕过插值) ====================

    def publish_arm_raw(self, values):
        self._publish_arm(values)
        arr = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._arm_current = arr
            self._arm_target = arr.copy()
            self._arm_start = arr.copy()
            self._arm_step = 0
            self._arm_total = 0

    def publish_hand_raw(self, values, effort=None):
        self._publish_hand(values, effort=effort)
        arr = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._hand_current = arr
            self._hand_target = arr.copy()
            self._hand_start = arr.copy()
            self._hand_step = 0
            self._hand_total = 0
            if effort is not None:
                self._hand_effort = [float(x) for x in effort]

    def publish_eef_raw(self, values):
        self._publish_eef(values)
        arr = np.asarray(values, dtype=np.float64)
        with self._lock:
            self._eef_current = arr
            self._eef_target = arr.copy()
            self._eef_start = arr.copy()
            self._eef_step = 0
            self._eef_total = 0

    # ==================== 内部 ROS 发布 (使用预分配消息) ====================

    def _publish_arm(self, values):
        self._arm_msg.header.stamp = self.node.get_clock().now().to_msg()
        positions = (values.tolist() if isinstance(values, np.ndarray)
                     else [float(x) for x in values])
        if len(positions) != len(self._arm_entries):
            raise ValueError(
                f"arm command must be {len(self._arm_entries)}D, got {len(positions)}D")
        sequence = self._arm_sequence
        for entry, position in zip(self._arm_entries, positions):
            entry.sequence = sequence
            entry.position = position
        self._arm_sequence = (sequence + 1) & 0xFFFFFFFF
        self.pub_arm.publish(self._arm_msg)

    def _publish_hand(self, values, effort=None):
        self._hand_msg.header.stamp = self.node.get_clock().now().to_msg()
        if isinstance(values, np.ndarray):
            self._hand_msg.position = np.rint(values).astype(np.float64).tolist()
        else:
            self._hand_msg.position = [float(x) for x in values]
        # effort 必须给：T_MoveClawRos2 / T_MoveHandRos2 都默认 100，长度=name 长度
        if effort is None:
            effort = [self.DEFAULT_HAND_EFFORT] * len(self._hand_msg.position)
        self._hand_msg.effort = [float(x) for x in effort]
        self.pub_hand.publish(self._hand_msg)

    def _publish_eef(self, v):
        left = self._eef_msg.poses[0]
        right = self._eef_msg.poses[1]
        left.position.x, left.position.y, left.position.z = float(v[0]), float(v[1]), float(v[2])
        right.position.x, right.position.y, right.position.z = float(v[3]), float(v[4]), float(v[5])
        # 线性插值后四元数 norm 会偏离 1，发布前归一化
        lq = np.asarray(v[6:10], dtype=np.float64)
        rq = np.asarray(v[10:14], dtype=np.float64)
        ln = float(np.linalg.norm(lq))
        rn = float(np.linalg.norm(rq))
        if ln > 1e-9:
            lq = lq / ln
        if rn > 1e-9:
            rq = rq / rn
        left.orientation.x, left.orientation.y, left.orientation.z, left.orientation.w = (
            float(lq[0]), float(lq[1]), float(lq[2]), float(lq[3]))
        right.orientation.x, right.orientation.y, right.orientation.z, right.orientation.w = (
            float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3]))
        self._eef_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.pub_eef.publish(self._eef_msg)

    # ==================== 150Hz 插值线程 ====================

    def _interp_loop(self):
        mono_ns = time.monotonic_ns
        sleep = time.sleep
        target_fps = self.target_fps
        interval_ns = int(1e9 / target_fps)

        # busy-wait 阈值: sleep 精度约 1ms, 小于此值用 busy-wait
        BUSYWAIT_NS = 1_500_000  # 1.5ms

        # 使用绝对时间基准，避免累积漂移
        next_wake = mono_ns()
        # waist publish decimation: interpolate every tick, publish ~50Hz
        # (every 3rd 150Hz tick) to avoid flooding the MC waist channel.
        waist_pub_tick = 0

        while self._running:
            # ---- 插值计算 (lock 内) ----
            pub_arm = None
            pub_hand = None
            pub_hand_effort = None
            pub_eef = None
            pub_waist = None

            with self._lock:
                # arm: 正在插值的段先推进; 段完且 chunk 还有剩, 自动装下一帧
                if self._arm_target is not None:
                    if self._arm_step < self._arm_total:
                        self._arm_step += 1
                        alpha = self._arm_step / self._arm_total
                        self._arm_current = (1.0 - alpha) * self._arm_start + alpha * self._arm_target
                    elif self._arm_chunk is not None and self._arm_chunk_idx < len(self._arm_chunk):
                        # 上一段插完, 把下一帧 waypoint 装载为新 target
                        self._arm_start = self._arm_current.copy()
                        self._arm_target = self._arm_chunk[self._arm_chunk_idx].copy()
                        self._arm_step = 1
                        self._arm_total = self._arm_chunk_steps
                        alpha = 1.0 / self._arm_total
                        self._arm_current = (1.0 - alpha) * self._arm_start + alpha * self._arm_target
                        self._arm_chunk_idx += 1
                    elif self._arm_chunk is not None and self._arm_chunk_idx >= len(self._arm_chunk):
                        # 整 chunk 跑完, 清空 chunk (target 保持在最后一帧)
                        self._arm_chunk = None
                        self._arm_chunk_idx = 0
                    pub_arm = self._arm_current

                if self._hand_target is not None:
                    if self._hand_step < self._hand_total:
                        self._hand_step += 1
                        alpha = self._hand_step / self._hand_total
                        self._hand_current = (1.0 - alpha) * self._hand_start + alpha * self._hand_target
                    elif self._hand_chunk is not None and self._hand_chunk_idx < len(self._hand_chunk):
                        self._hand_start = self._hand_current.copy()
                        self._hand_target = self._hand_chunk[self._hand_chunk_idx].copy()
                        self._hand_step = 1
                        self._hand_total = self._hand_chunk_steps
                        alpha = 1.0 / self._hand_total
                        self._hand_current = (1.0 - alpha) * self._hand_start + alpha * self._hand_target
                        self._hand_chunk_idx += 1
                    elif self._hand_chunk is not None and self._hand_chunk_idx >= len(self._hand_chunk):
                        self._hand_chunk = None
                        self._hand_chunk_idx = 0
                    pub_hand = self._hand_current
                    pub_hand_effort = self._hand_effort

                if self._eef_target is not None:
                    if self._eef_step < self._eef_total:
                        self._eef_step += 1
                        alpha = self._eef_step / self._eef_total
                        self._eef_current = (1.0 - alpha) * self._eef_start + alpha * self._eef_target
                    elif self._eef_chunk is not None and self._eef_chunk_idx < len(self._eef_chunk):
                        self._eef_start = self._eef_current.copy()
                        self._eef_target = self._eef_chunk[self._eef_chunk_idx].copy()
                        self._eef_step = 1
                        self._eef_total = self._eef_chunk_steps
                        alpha = 1.0 / self._eef_total
                        self._eef_current = (1.0 - alpha) * self._eef_start + alpha * self._eef_target
                        self._eef_chunk_idx += 1
                    elif self._eef_chunk is not None and self._eef_chunk_idx >= len(self._eef_chunk):
                        self._eef_chunk = None
                        self._eef_chunk_idx = 0
                    pub_eef = self._eef_current

                # waist (UPPER-BODY chunk path): same step-counter model as arm.
                if self._waist_target is not None:
                    if self._waist_step < self._waist_total:
                        self._waist_step += 1
                        alpha = self._waist_step / self._waist_total
                        self._waist_current = (1.0 - alpha) * self._waist_start + alpha * self._waist_target
                    elif self._waist_chunk is not None and self._waist_chunk_idx < len(self._waist_chunk):
                        self._waist_start = self._waist_current.copy()
                        self._waist_target = self._waist_chunk[self._waist_chunk_idx].copy()
                        self._waist_step = 1
                        self._waist_total = self._waist_chunk_steps
                        alpha = 1.0 / self._waist_total
                        self._waist_current = (1.0 - alpha) * self._waist_start + alpha * self._waist_target
                        self._waist_chunk_idx += 1
                    elif self._waist_chunk is not None and self._waist_chunk_idx >= len(self._waist_chunk):
                        self._waist_chunk = None
                        self._waist_chunk_idx = 0
                    pub_waist = self._waist_current

            # ---- publish (lock 外) ----
            if pub_arm is not None:
                self._publish_arm(pub_arm)
            if pub_hand is not None:
                self._publish_hand(pub_hand, effort=pub_hand_effort)
            if pub_eef is not None:
                self._publish_eef(pub_eef)
            # waist: 4D [yaw, roll, pitch, height] → node.publish_waist, ~50Hz.
            if pub_waist is not None:
                waist_pub_tick += 1
                if waist_pub_tick % 3 == 0:
                    try:
                        cur = pub_waist
                        self.node.publish_waist(
                            pitch=float(cur[2]), roll=float(cur[1]),
                            yaw=float(cur[0]), height=float(cur[3]),
                        )
                    except Exception:
                        pass

            # ---- 记录 publish 后的时间戳 (与 ros2 topic hz 对齐) ----
            self._frame_ts.append(mono_ns())

            # ---- 绝对时间定时: 基于固定节拍，不累积漂移 ----
            next_wake += interval_ns
            now = mono_ns()
            # 如果已经超过了下一个唤醒时间 (missed deadline)，重置基准
            if now > next_wake + interval_ns:
                next_wake = now + interval_ns

            remaining = next_wake - mono_ns()
            if remaining > BUSYWAIT_NS:
                sleep((remaining - BUSYWAIT_NS) / 1e9)
            while mono_ns() < next_wake:
                pass

            # ---- PI 自适应频率调节 ----
            ts = self._frame_ts
            n = len(ts)
            if n >= 2:
                dt_ns = ts[-1] - ts[0]
                if dt_ns > 0:
                    actual_fps = (n - 1) * 1e9 / dt_ns
                    fps_error = target_fps - actual_fps
                    # 直接微调 interval_ns 补偿
                    interval_ns = int(1e9 / target_fps - fps_error * 500)

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
