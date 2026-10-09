"""
A2 机器人高层客户端接口 - 纯 HTTP，运行在本地主机上。

通过 HTTP 调用机器人上的 ros_server (port 5050) 完成控制和观测。
不依赖 ROS2，任何 Python 环境都可使用。

插值说明:
  - send_arm / send_hand 默认从当前位姿插值到目标 (steps=200, fps=150)，
    设置 steps=1 可跳过插值直接发送单帧。
  - step() 用于 VLA 推理循环，以模型输出频率 (如 30Hz) 调用，
    内部自动插值到 target_fps (默认 150Hz)。
"""

import time
import numpy as np
import requests
import cv2
from typing import Optional, Callable, Any
from enum import Enum

from config import (
    OMNIHAND_LEFT, OMNIHAND_RIGHT, slerp,
    VLA_ARM_INIT_POS, VLA_HAND_INIT_POS, VLA_HAND_FIST_POS, VLA_ARM_INIT_UP_POS,
    VLA_LEFT_EEF_DOWN_TARGET, VLA_RIGHT_EEF_DOWN_TARGET,
    VLA_ARM_INIT_POS_2, VLA_HAND_INIT_POS_2,
    VLA_ARM_INIT_POS_3, VLA_HAND_INIT_POS_3,
    VLA_LEFT_EEF_UP_TARGET, VLA_RIGHT_EEF_UP_TARGET,
)

# Token-chunk streaming (VLA sends 20/30Hz chunks, robot interpolates to 60Hz,
# optional residual head compensation). See token_chunk_stream.py.
from token_chunk_stream import (
    DEFAULT_VLA_TOKEN_CHUNK_TOPIC,
    TokenChunkEnvelope,
    TokenChunkReceiver,
    TokenChunkStreamer,
    ResidualHeadRunner,
)


class ControlMode(Enum):
    JOINT = "joint"
    EEF = "eef"


class A2RobotInterface:
    """A2 机器人高层 HTTP 客户端。

    用法 (VLA 推理):
        robot = A2RobotInterface("192.168.2.50")
        obs = robot.get_observation()
        action = model.predict(obs)
        robot.step(action)   # 内部插值到 150Hz

    用法 (直接控制, 自动插值):
        robot.send_arm(target_14d)              # 从当前位姿平滑移动到目标
        robot.send_arm(target_14d, steps=1)     # 不插值, 直接发送
    """

    def __init__(self, robot_ip: str = "192.168.2.50", port: int = 5050,
                 timeout: float = 2.0, target_fps: float = 150.0):
        self.base_url = f"http://{robot_ip}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        # requests.Session 默认 HTTPAdapter 只有 10 个连接, 并发拉图时会排队。
        # 调大到 32, 让 get_observation 里 5 路图像 + joint_states 能真正并发。
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.target_fps = target_fps

        # 并发拉观测的线程池 (5 路图像 + 1 路关节 = 6)
        from concurrent.futures import ThreadPoolExecutor
        self._obs_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="obs")

        # step() 默认非阻塞, 发完 HTTP 就返回;
        # 需要读取执行后的最终观测时, 先调用 wait_for_done() 同步一次。
        self._step_wait = False
        self._step_settle_ms = 0.0  # wait_for_done 默认 settle 毫秒

        # 缓存上一次发送的各部位状态（用于插值起点）
        self._last_arm = None
        self._last_hand = None

        # 底层 Action 控制模式; set_action_mode() 成功时更新。
        # 所有上层 API (step / reset / send_arm) 均按此模式路由到对应 ROS topic。
        self._action_mode = "joint"

    def set_step_wait(self, wait: bool, settle_ms: float = None):
        """设置 step() 是否阻塞等待 server 端插值发布完成。

        Args:
            wait:       True = step() 同步等待, False = 发完 HTTP 就返回 (默认)
            settle_ms:  wait_for_done / 阻塞 step 末尾额外 sleep 的毫秒 (默认 30ms)
        """
        self._step_wait = bool(wait)
        if settle_ms is not None:
            self._step_settle_ms = max(0.0, float(settle_ms))

    def wait_for_done(self, settle_ms: float = None, timeout_ms: float = 2000.0) -> bool:
        """阻塞直到 server 端 arm/hand/eef 三通道的插值都发布完成。

        在 step() 非阻塞调用之后、读取 get_observation() 之前调用一次,
        即可确保观测反映的是执行后的最终位姿。

        Args:
            settle_ms:  插值发完后再 sleep 多少毫秒让电机跟上;
                        None = 使用 set_step_wait() 设的默认值 (30ms)
            timeout_ms: server 端最大等待毫秒 (防止误差累积卡死)

        Returns: True = 成功等到, False = HTTP 或超时失败
        """
        eff_settle = self._step_settle_ms if settle_ms is None else max(0.0, float(settle_ms))
        try:
            resp = self.session.post(
                f"{self.base_url}/wait_step_done",
                json={"settle_ms": eff_settle, "timeout_ms": float(timeout_ms)},
                timeout=(timeout_ms + eff_settle + 1000.0) / 1000.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _get(self, path: str):
        return self.session.get(f"{self.base_url}{path}", timeout=self.timeout)

    def _post(self, path: str, json_data: dict):
        return self.session.post(f"{self.base_url}{path}", json=json_data, timeout=self.timeout)

    def _get_image(self, path: str) -> Optional[np.ndarray]:
        """获取 JPEG 图像并解码为 numpy BGR。"""
        try:
            resp = self._get(path)
            if resp.status_code != 200:
                return None
            img_arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _ensure_last_state(self):
        """确保缓存了当前关节状态（用于插值起点）。"""
        if self._last_arm is None or self._last_hand is None:
            js = self.get_joint_states()
            if js:
                if self._last_arm is None and js.get("arm"):
                    self._last_arm = list(js["arm"]["position"])
                if self._last_hand is None and js.get("hand"):
                    self._last_hand = list(js["hand"]["position"])

    # ==================== 内部发送 ====================

    def set_speed(self, hz: float) -> bool:
        """设置发送频率，服务端据此调整插值步数。

        例: set_speed(hz=10)  → 服务端从 10Hz 插值到 150Hz (interp_steps=15)
            set_speed(hz=30)  → 服务端从 30Hz 插值到 150Hz (interp_steps=5)
        """
        try:
            resp = self._post("/set_speed", {"hz": float(hz)})
            return resp.status_code == 200
        except Exception:
            return False

    def set_action_mode(self, mode: str, timeout: float = 180.0) -> bool:
        """设置机器人底层 Action 模式 (joint / eef)。幂等: 已是目标则不切换。

        调用成功后 `step()` / `reset()` / `send_arm()` 会自动按此模式路由到
        对应 ROS topic, 不需要在调用处再传 mode / control_mode。

        首次使用 EEF 控制前应先调用本方法。内部通过本地的
        scripts/utils/change_action_mode_a2.sh 脚本 ssh 到 Orin/x86 切换底层
        Action, 耗时约 10-30 秒。本地需安装 sshpass: sudo apt install sshpass

        Args:
            mode:    "joint" 或 "eef"
            timeout: 子进程最长等待秒数 (默认 180s)

        Returns: True = 切换成功或已处于目标模式; False = 失败 (内部状态不更新)
        """
        mode = str(mode).lower()
        if mode not in ("joint", "eef"):
            raise ValueError(f"mode must be 'joint' or 'eef', got {mode!r}")

        import os
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "scripts", "utils", "change_action_mode_a2.sh")
        if not os.path.isfile(script):
            print(f"[set_action_mode] 找不到脚本: {script}")
            return False
        try:
            result = subprocess.run(["bash", script, mode], timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[set_action_mode] 超时 ({timeout}s), 切换未完成")
            return False
        if result.returncode != 0:
            return False
        self._action_mode = mode
        return True

    def _send_arm(self, arm_values, wait: bool = False, settle_ms: float = 0.0) -> bool:
        """发送 arm 命令, 服务端会插值到 150Hz。用于 step()。

        Args:
            wait:      True = 服务端等插值发完再返回 HTTP
            settle_ms: 发完后再让服务端 sleep 多少毫秒 (电机跟随延迟)
        """
        try:
            path = "/send_arm"
            if wait:
                path += f"?wait=true&settle_ms={settle_ms}"
            # 阻塞模式下 HTTP 超时需覆盖 interp_steps/150 + settle
            to = self.timeout if not wait else max(self.timeout, 5.0)
            resp = self.session.post(f"{self.base_url}{path}",
                                     json={"values": [float(x) for x in arm_values]},
                                     timeout=to)
            return resp.status_code == 200
        except Exception:
            return False

    def _send_hand(self, hand_values, effort=None, wait: bool = False, settle_ms: float = 0.0) -> bool:
        """发送 hand 命令, 服务端会插值到 150Hz。用于 step()。"""
        try:
            payload = {"values": [float(x) for x in hand_values]}
            if effort is not None:
                payload["effort"] = [float(x) for x in effort]
            path = "/send_hand"
            if wait:
                path += f"?wait=true&settle_ms={settle_ms}"
            to = self.timeout if not wait else max(self.timeout, 5.0)
            resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=to)
            return resp.status_code == 200
        except Exception:
            return False

    def _send_eef(self, eef_values, wait: bool = False, settle_ms: float = 0.0) -> bool:
        """发送 eef 命令, 服务端会插值到 150Hz。用于 step()。"""
        try:
            path = "/send_eef"
            if wait:
                path += f"?wait=true&settle_ms={settle_ms}"
            to = self.timeout if not wait else max(self.timeout, 5.0)
            resp = self.session.post(f"{self.base_url}{path}",
                                     json={"values": [float(x) for x in eef_values]},
                                     timeout=to)
            return resp.status_code == 200
        except Exception:
            return False

    def _send_arm_direct(self, arm_values) -> bool:
        """发送 arm 命令, 绕过服务端插值直接发布。用于 send_arm() 本地插值循环。"""
        try:
            resp = self._post("/send_arm?raw=true", {"values": [float(x) for x in arm_values]})
            return resp.status_code == 200
        except Exception:
            return False

    def _send_hand_direct(self, hand_values, effort=None) -> bool:
        """发送 hand 命令, 绕过服务端插值直接发布。用于 send_hand() 本地插值循环。"""
        try:
            payload = {"values": [float(x) for x in hand_values]}
            if effort is not None:
                payload["effort"] = [float(x) for x in effort]
            resp = self._post("/send_hand?raw=true", payload)
            return resp.status_code == 200
        except Exception:
            return False

    def _send_eef_direct(self, eef_values) -> bool:
        """发送 eef 命令, 绕过服务端插值直接发布。"""
        try:
            resp = self._post("/send_eef?raw=true", {"values": [float(x) for x in eef_values]})
            return resp.status_code == 200
        except Exception:
            return False

    # ==================== VLA 推理接口 ====================

    def step(self, action: dict, wait: bool = None, settle_ms: float = None):
        """执行一步动作，直接发送到机器人。

        以模型输出频率调用 (如 30Hz)，由调用者控制帧率。
        每次调用只发送一帧命令 (每个部位一次 HTTP 请求)。
        机器人端 server_node.py 内部会将 30Hz 命令插值到 150Hz 发布到 ROS Topic。

        默认 step() 会阻塞到 server 端插值发布完成后再返回 (wait=True),
        这样紧接着调用 get_observation() 得到的就是执行后的最终位姿。
        如果不希望阻塞 (如 RTC 流水线), 传 wait=False。

        控制模式 (joint / eef) 由 `set_action_mode()` 设置, action 中不再接受 "mode" 键。

        Args:
            wait:       True = 同步等待插值发布完成 (arm/hand/eef);
                        None (默认) = 使用 set_step_wait() 设置的默认值
            settle_ms:  插值发完后再等多少毫秒让电机跟上; None = 使用默认值

        action dict 的键 (全部可选，没有的键不控制):
            "arm":          (14,) 关节角弧度 (joint 模式) 或 EEF 位姿 (eef 模式)
            "hand":         (20,) actuator 原始值 0-4096 或弧度 (取决于 hand_value)
            "hand_value":   "raw" (默认, hand 为 0-4096) 或 "rad" (hand 为弧度, 自动转换)
            "hand_effort":  (20,) 手部力矩 0-255, 不传默认 100
            "head":         (2,) [shake, nod] 弧度
            "waist":        (6,) [x,y,z,roll,pitch,yaw] 或 (3,) [z,pitch,yaw]
            "loco":         (3,) [forward, lateral, angular]
        """
        mode = self._action_mode
        hand_value = action.get("hand_value", "raw")
        eff_wait = self._step_wait if wait is None else bool(wait)
        eff_settle = self._step_settle_ms if settle_ms is None else max(0.0, float(settle_ms))

        # 腰部和移动延迟较高 (HTTP RPC), 优先发送
        if "waist" in action:
            w = action["waist"]
            if len(w) == 3:
                self.send_waist(z=float(w[0]), pitch=float(w[1]), yaw=float(w[2]))
            else:
                self.send_waist(x=float(w[0]), y=float(w[1]), z=float(w[2]),
                                roll=float(w[3]), pitch=float(w[4]), yaw=float(w[5]))

        if "loco" in action:
            lo = action["loco"]
            self.send_loco(float(lo[0]), float(lo[1]), float(lo[2]))

        # arm/hand/eef: 先把所有命令发出去 (不阻塞), 最后统一等,
        # 这样多个部位的插值是并行发布的, 等同一个 settle 即可。
        has_arm = "arm" in action
        has_hand = "hand" in action
        has_eef = has_arm and mode == "eef"

        if has_arm:
            vals = [float(x) for x in action["arm"]]
            if mode == "eef":
                self._send_eef(vals, wait=False)
            else:
                self._send_arm(vals, wait=False)

        if has_hand:
            hand_effort = action.get("hand_effort")
            if hand_value == "rad":
                radians = list(action["hand"])
                left_raw = OMNIHAND_LEFT.radians_to_actuator(radians[:10])
                right_raw = OMNIHAND_RIGHT.radians_to_actuator(radians[10:])
                self._send_hand([float(x) for x in left_raw + right_raw],
                                effort=hand_effort, wait=False)
            else:
                self._send_hand([float(x) for x in action["hand"]],
                                effort=hand_effort, wait=False)

        if "head" in action:
            h = action["head"]
            self.send_head(float(h[0]), float(h[1]))

        # 阻塞等待: 调用 server 的 /wait_step_done, 让 server 根据 arm/hand/eef
        # 的剩余插值步数精确等待到所有通道发布完成, 再额外 sleep settle_ms 让
        # 电机跟上。这样调用完 step() 后立即 get_observation() 即可拿到执行后的
        # 最终状态。
        if eff_wait and (has_arm or has_hand or has_eef):
            try:
                timeout_ms = max(2000.0, self.target_fps * 50 + eff_settle + 1000.0)
                self.session.post(
                    f"{self.base_url}/wait_step_done",
                    json={"settle_ms": eff_settle, "timeout_ms": timeout_ms},
                    timeout=(timeout_ms + 1000.0) / 1000.0,
                )
            except Exception:
                # fallback: 本地保守 sleep
                time.sleep(5.0 / self.target_fps)
                if eff_settle > 0:
                    time.sleep(eff_settle / 1000.0)

        # 更新缓存
        if has_arm:
            self._last_arm = [float(x) for x in action["arm"]]
        if has_hand:
            self._last_hand = [float(x) for x in action["hand"]]

    # ==================== 整 chunk 发送 (一次发一整包动作) ====================

    def step_chunk(self, chunk: dict, chunk_fps: float = 30.0,
                   wait: bool = True, settle_ms: float = 0.0,
                   timeout_ms: float = 30000.0,
                   chunk_id: int = -1,
                   adaptive_transition: bool = True,
                   s_used_local: Optional[int] = None,
                   backstep_check: bool = False,
                   arm_dim_slice: Optional[tuple] = None,
                   backstep_tol: float = 0.005,
                   backstep_max_skip: int = 5,
                   backstep_max_wps_gate: float = 4.0):
        """一次性把整 chunk (H 帧动作) 发给 server, server 内部按 chunk_fps 自动推进。

        与 step() 的区别: step() 每次只发一帧, 由调用方控制循环节奏;
        step_chunk() 一次发 H 帧, 服务端自动按 chunk_fps 节奏切帧, 客户端只需等。

        控制模式 (joint / eef) 由 `set_action_mode()` 设置。chunk 中:
            "arm":         (H, 14)  joint 模式: 关节角弧度;  eef 模式: EEF 位姿
            "hand":        (H, 20)  actuator 0-4096 或 弧度 (取决于 hand_value)
            "hand_value":  "raw" (默认) 或 "rad"
            "hand_effort": (20,)    整 chunk 共用一个 effort, 不传则默认 100
            "head":        (2,)     [shake, nod] (只发一次, chunk 不分帧)
            "waist":       (3,)/(6,)
            "loco":        (3,)

        Args:
            chunk_fps:    chunk 内每帧之间的频率 (Hz), server 据此算 interp_steps
            wait:         True = 阻塞到整 chunk 跑完再返回 (默认)
            settle_ms:    跑完后再 sleep 毫秒
            timeout_ms:   server 端最长等待毫秒
            adaptive_transition: True (默认) = server 端启用 jump 自适应 transition_ms
                          (大 jump 时 transition 段拉长, 视觉平滑); False = server 不
                          补偿, 第一段恒为 1/chunk_fps. 关闭可暴露 RTC 算法本身的不
                          连续, 用于诊断.
            s_used_local: train-time RTC 专用. None (默认) = 老协议, server 直接
                          装载整 chunk; 整数 = 新协议, 把 client 推理 obs 时刻在
                          server-local 老 chunk 中的帧索引发给 server, server 在
                          swap 瞬间自己算 actual_delay 切片. 这样 RTT 期间物理推
                          进的帧不再被客户端漏算.
            backstep_check: True = server 在 actual_delay 切片之后, 再做 position-
                          based 回退检测, 跳过模型把 chunk[0] 画在 robot 物理位置
                          后方的开头几帧 (pos_skip). 默认 False (热插拔). 仅在
                          s_used_local 提供时生效 (新协议路径).
            arm_dim_slice: (start, stop) — chunk 维度上 arm 关节范围. backstep 投影
                          只在 arm 维做, 排除 hand/gripper/waist (这些维度突变会
                          污染方向估计). None → 即使 backstep_check=True 也跳过.
            backstep_tol: 投影回退判定阈值 (rad). 默认 0.005 (~0.3°).
            backstep_max_skip: pos_skip 上限 (防止误判把 chunk 跳空). 默认 5.
            backstep_max_wps_gate:
                          max_wps 闸门: 若 actual_delay 切片后 chunk[0] →
                          robot 当前位置的 max_wps 大于此阈值 (按 chunk 内
                          峰值速度反推), server 视为模型主动模式切换 → 不做
                          backstep, 让 adaptive_transition 处理大 jump. 默认 4.0.

        Returns:
            老协议 (s_used_local=None): bool — True 成功 / False 失败.
            新协议 (s_used_local 提供):  dict {"ok": bool, "actual_delay": int,
                                                "pos_skip": int,
                                                "max_wps_pre_skip": float | None,
                                                "backstep_gated": bool}.
        """
        mode = self._action_mode
        hand_value = chunk.get("hand_value", "raw")

        # 头/腰/移动: 头和移动仍只发一次 (语义上是 chunk 级 one-shot 命令);
        # 腰部支持 1D (整 chunk 共用) 或 2D (N x 3/6, 按 chunk_fps 节奏每帧发)。
        if "waist" in chunk:
            w = chunk["waist"]
            w_arr = np.asarray(w, dtype=float)
            if w_arr.ndim == 1:
                # 老路径: 整 chunk 一次性发
                if w_arr.shape[0] == 3:
                    self.send_waist(z=float(w_arr[0]), pitch=float(w_arr[1]), yaw=float(w_arr[2]))
                else:
                    self.send_waist(x=float(w_arr[0]), y=float(w_arr[1]), z=float(w_arr[2]),
                                    roll=float(w_arr[3]), pitch=float(w_arr[4]), yaw=float(w_arr[5]))
            else:
                # 2D: 起后台线程按 chunk_fps 节奏每帧发, 与 arm/hand chunk 时序对齐
                import threading as _th
                def _waist_pace(rows, fps):
                    interval = 1.0 / max(fps, 1.0)
                    t0 = time.time()
                    for i, row in enumerate(rows):
                        target = t0 + i * interval
                        sleep_s = target - time.time()
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        if len(row) == 3:
                            self.send_waist(z=float(row[0]), pitch=float(row[1]), yaw=float(row[2]))
                        else:
                            self.send_waist(x=float(row[0]), y=float(row[1]), z=float(row[2]),
                                            roll=float(row[3]), pitch=float(row[4]), yaw=float(row[5]))
                _th.Thread(target=_waist_pace, args=(w_arr.tolist(), chunk_fps), daemon=True).start()
        if "loco" in chunk:
            lo = chunk["loco"]
            self.send_loco(float(lo[0]), float(lo[1]), float(lo[2]))
        if "head" in chunk:
            h = chunk["head"]
            self.send_head(float(h[0]), float(h[1]))

        payload = {"chunk_fps": float(chunk_fps), "chunk_id": int(chunk_id),
                   "adaptive_transition": bool(adaptive_transition)}
        if s_used_local is not None:
            payload["s_used_local"] = int(s_used_local)
        # 仅在 s_used_local 提供 (新协议) 且开启 backstep 时附带 backstep 参数;
        # cold-start / 老协议路径 server 不读这些字段.
        if s_used_local is not None and backstep_check and arm_dim_slice is not None:
            payload["backstep_check"] = True
            payload["arm_dim_slice"] = [int(arm_dim_slice[0]),
                                         int(arm_dim_slice[1])]
            payload["backstep_tol"] = float(backstep_tol)
            payload["backstep_max_skip"] = int(backstep_max_skip)
            payload["backstep_max_wps_gate"] = float(backstep_max_wps_gate)

        if "arm" in chunk and chunk["arm"] is not None:
            arm_2d = np.asarray(chunk["arm"], dtype=float)
            if arm_2d.ndim != 2:
                raise ValueError(f"arm chunk must be 2D, got shape {arm_2d.shape}")
            if mode == "eef":
                payload["eef"] = arm_2d.tolist()
            else:
                payload["arm"] = arm_2d.tolist()

        if "hand" in chunk and chunk["hand"] is not None:
            hand_2d = np.asarray(chunk["hand"], dtype=float)
            if hand_2d.ndim != 2:
                raise ValueError(f"hand chunk must be 2D, got shape {hand_2d.shape}")
            if hand_value == "rad":
                converted = []
                for row in hand_2d:
                    left_raw = OMNIHAND_LEFT.radians_to_actuator(list(row[:10]))
                    right_raw = OMNIHAND_RIGHT.radians_to_actuator(list(row[10:]))
                    converted.append([float(x) for x in (left_raw + right_raw)])
                payload["hand"] = converted
            else:
                payload["hand"] = hand_2d.tolist()
            if "hand_effort" in chunk and chunk["hand_effort"] is not None:
                payload["hand_effort"] = [float(x) for x in chunk["hand_effort"]]

        # 没东西可发就直接返回
        if "arm" not in payload and "hand" not in payload and "eef" not in payload:
            return ({"ok": True, "actual_delay": 0, "pos_skip": 0,
                     "max_wps_pre_skip": None, "backstep_gated": False}
                    if s_used_local is not None else True)

        try:
            url = f"{self.base_url}/send_chunk"
            params = []
            if wait:
                params.append("wait=true")
                params.append(f"settle_ms={float(settle_ms)}")
                params.append(f"timeout_ms={float(timeout_ms)}")
            if params:
                url += "?" + "&".join(params)
            # HTTP 超时: 至少覆盖 chunk 时长 + settle + 缓冲
            horizon = max(
                len(payload.get("arm", []) or []),
                len(payload.get("hand", []) or []),
                len(payload.get("eef", []) or []),
            )
            chunk_dur_s = horizon / max(chunk_fps, 1.0)
            http_to = max(self.timeout, chunk_dur_s + settle_ms / 1000.0 + 5.0)
            resp = self.session.post(url, json=payload, timeout=http_to)
            ok = resp.status_code == 200
            actual_delay = 0
            pos_skip = 0
            max_wps_pre_skip = None
            backstep_gated = False
            if s_used_local is not None and ok:
                try:
                    rj = resp.json()
                    actual_delay = int(rj.get("actual_delay", 0) or 0)
                    pos_skip = int(rj.get("pos_skip", 0) or 0)
                    mws = rj.get("max_wps_pre_skip", None)
                    max_wps_pre_skip = (float(mws) if mws is not None else None)
                    backstep_gated = bool(rj.get("backstep_gated", False))
                except Exception:
                    actual_delay = 0
                    pos_skip = 0
                    max_wps_pre_skip = None
                    backstep_gated = False
        except Exception as e:
            print(f"[step_chunk] HTTP 失败: {e}")
            return ({"ok": False, "actual_delay": 0, "pos_skip": 0,
                     "max_wps_pre_skip": None, "backstep_gated": False}
                    if s_used_local is not None else False)

        # 更新缓存 (last_arm / last_hand 取最后一帧)
        if "arm" in payload and payload["arm"]:
            self._last_arm = [float(x) for x in payload["arm"][-1]]
        if "hand" in payload and payload["hand"]:
            self._last_hand = [float(x) for x in payload["hand"][-1]]

        if s_used_local is not None:
            return {"ok": ok, "actual_delay": actual_delay, "pos_skip": pos_skip,
                    "max_wps_pre_skip": max_wps_pre_skip,
                    "backstep_gated": backstep_gated}
        return ok

    def cancel_chunk(self) -> bool:
        """立即取消 server 端正在执行的 chunk, 电机停在当前位置。"""
        try:
            resp = self.session.post(f"{self.base_url}/cancel_chunk",
                                     json={}, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ==================== VLA token-chunk 流 (60Hz 插值 + residual 补偿) ====================
    #
    # 与 step_chunk 的区别:
    #   step_chunk       接收一整包 arm/hand/eef 命令 → server 端插到 150Hz 发布。
    #   token-chunk 流    上游 VLA 把 (H, action_dim) normalized token 通过一个
    #                    专用 topic (占位: /vla/token_chunk, 后面换真实名) 直接推给
    #                    RobotInterface, RobotInterface 本地按 source_hz (20/30Hz)
    #                    取 token, 60Hz 插值成密集 waypoint, 每步再叠加一次可选的
    #                    residual head δ (bounded ±0.3), 最后走 decode_fn 反归一化
    #                    成 send_arm/send_hand 命令交给 server 端做最后一段 60→150Hz
    #                    平滑。整条链路里 residual 在 normalized 空间加, 保证 head
    #                    看到的 base_action 与训练时一致.
    #
    # 用法:
    #   robot = A2RobotInterface(...)
    #   robot.start_token_chunk_stream(
    #       decode_fn=my_decode_normalized_to_action,
    #       residual_head_ckpt="/path/to/residual_head.pt",     # 或 None 关闭补偿
    #       residual_head_config={"action_dim":32,"output_horizon":40,...},
    #   )
    #   # 上游 VLA 侧通过 ROS ByteMultiArray publish msgpack 后 receiver 收到
    #   # 就自动流起来. 想暂停补偿:
    #   robot.set_residual_compensation(False)
    #
    def start_token_chunk_stream(
        self,
        decode_fn: Callable[[np.ndarray], dict],
        *,
        topic: str = DEFAULT_VLA_TOKEN_CHUNK_TOPIC,
        target_interp_hz: float = 60.0,
        default_source_hz: float = 20.0,
        residual_head_ckpt: Optional[str] = None,
        residual_head_config: Optional[dict] = None,
        residual_head_device: str = "cpu",
        residual_gain: float = 1.0,
        receiver: Optional[TokenChunkReceiver] = None,
        ros_node: Optional[Any] = None,
    ) -> "TokenChunkStreamer":
        """启动 VLA token chunk 流 → 60Hz 插值 → 可选 residual → send_arm/hand.

        Args:
            decode_fn:  把 (D,) normalized token 反归一化成 step() 认识的
                        action dict (至少含 "arm"/"hand"/... 之一).
            topic:      VLA→机器人 token chunk 的 topic 路径.
                        默认 /vla/token_chunk (占位), 上游 GR00T 端应 set 到同一路径.
            target_interp_hz: 本地插值输出频率 (60Hz).
            default_source_hz: 上游 VLA 若没在 envelope 里写 source_hz 时用的默认取
                        token 节奏 (20 或 30).
            residual_head_ckpt:  offline 训好的 .pt 路径, None → 不启用 residual.
            residual_head_config: 匹配 ckpt 的 head 结构参数
                        (action_dim / output_horizon / hidden_dim / cond_dim /
                         state_feature_dim / vla_feature_dim / history_len /
                         delta_bound / step_embed_dim / num_hidden_layers).
                        通常直接读 residual_head_config.json.
            residual_gain: 附加在 δ 上的标量, 1.0 = 训练时相同幅度. 现场想弱化 residual
                        可临时调小.
            receiver: 传入自己的 receiver (测试用). 未传时:
                      * ros_node 存在   → 自动建 RosTokenChunkReceiver
                      * ros_node 缺省   → 建通用 receiver, 上游用 submit() 直接投喂
            ros_node:  rclpy.node.Node — 只有走 ROS 通路时需要传, 会 create_subscription
                       到 `topic` 上.

        Returns:
            TokenChunkStreamer — 已 start()、后台线程运行中. 调用 stop() 关掉.
        """
        # 1. Receiver
        if receiver is None:
            if ros_node is not None:
                from token_chunk_stream import RosTokenChunkReceiver
                receiver = RosTokenChunkReceiver(ros_node, topic=topic)
            else:
                receiver = TokenChunkReceiver(topic=topic)

        # 2. Residual head runner (optional)
        residual_runner: Optional[ResidualHeadRunner] = None
        if residual_head_ckpt is not None and residual_head_config is not None:
            residual_runner = ResidualHeadRunner(
                head_state_dict_path=residual_head_ckpt,
                head_config=residual_head_config,
                device=residual_head_device,
                enabled=True,
            )

        # 3. Sink — decode_fn 拿到 (D,) normalized token, 生成 action dict, 走 step().
        def _sink(token: np.ndarray, meta: dict) -> None:
            action = decode_fn(token)
            if not isinstance(action, dict) or not action:
                return
            # 60Hz 流不适合走阻塞 step (wait_step_done 会毁频率), 强制 non-blocking.
            self.step(action, wait=False, settle_ms=0.0)

        # 4. Streamer
        streamer = TokenChunkStreamer(
            receiver=receiver,
            send_token_fn=_sink,
            target_interp_hz=target_interp_hz,
            default_source_hz=default_source_hz,
            residual_runner=residual_runner,
            residual_gain=residual_gain,
        )
        streamer.start()

        # 缓存以便外部访问 (方便暂停/关闭/切换 residual)
        self._token_stream = streamer
        self._token_receiver = receiver
        self._token_residual = residual_runner
        # 60Hz 流走 direct chunk-per-frame 语义, 把 server 端 30→60 插值步数
        # 拉到 60 fps 的 5 步 = 2.5 步, 但我们本地已经是 60Hz 密集了, 让 server
        # 保守用 1 步就够 (即透传). 用户想让 server 再插一层可以调 set_speed.
        try:
            self.set_speed(hz=float(target_interp_hz))
        except Exception:
            pass
        return streamer

    def stop_token_chunk_stream(self) -> None:
        """关掉 start_token_chunk_stream 起的后台流."""
        s = getattr(self, "_token_stream", None)
        if s is not None:
            s.stop()
            self._token_stream = None

    def set_residual_compensation(self, enabled: bool) -> None:
        """运行时切换 residual head 补偿是否叠加. 关掉即变纯插值。"""
        r = getattr(self, "_token_residual", None)
        if r is not None:
            r.set_enabled(bool(enabled))

    def get_token_stream_status(self) -> dict:
        """轻量诊断: 目前 receiver 队列长度 / residual 是否启用 / 最近 chunk_id."""
        s = getattr(self, "_token_stream", None)
        r = getattr(self, "_token_receiver", None)
        rh = getattr(self, "_token_residual", None)
        return {
            "running": bool(s is not None and s._running),
            "residual_enabled": bool(rh.enabled) if rh is not None else False,
            "receiver_queued": int(r.peek_available()) if r is not None else 0,
            "last_chunk_id": (s._last_chunk_id if s is not None else None),
        }

    def wait_chunk_done(self, settle_ms: float = 0.0,
                        timeout_ms: float = 30000.0) -> bool:
        """阻塞直到 server 端 chunk 队列全部跑完 (一般 step_chunk(wait=True) 已经等过了)。"""
        try:
            resp = self.session.post(
                f"{self.base_url}/wait_chunk_done",
                json={"settle_ms": float(settle_ms), "timeout_ms": float(timeout_ms)},
                timeout=(timeout_ms + settle_ms + 1000.0) / 1000.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_chunk_progress(self, timeout: float = 0.2) -> Optional[dict]:
        """查询服务端各通道当前已播放的 waypoint 索引 (浮点)。

        Returns:
            dict like {"arm": float|None, "hand": float|None, "eef": float|None} or
            None on HTTP failure. 浮点值表示已经完整播放完的 waypoint 数;
            None 表示该通道当前没有 chunk 在执行。
        """
        try:
            resp = self.session.get(f"{self.base_url}/chunk_progress", timeout=timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return {k: data.get(k) for k in ("arm", "hand", "eef")}
        except Exception:
            return None

    def get_observation_with_progress(self, hand_rad: bool = False,
                                       timeout: float = 3.0) -> Optional[dict]:
        """原子快照: 5 路相机 + 关节 + chunk 进度, 一次 HTTP 拿回。

        与分开调用 ``get_observation()`` + ``get_chunk_progress()`` 相比:
        - 一次 HTTP 往返 (而非 6 并发 + 1 串行 = 2 次 RTT 窗口);
        - server 端在 interp_pub._lock 内一次性采 chunk_played_idx, 与观测同一时刻,
          消除"采观测 → 查进度"之间的 RTT 错位 (原错位会让 RTC 换算的 s_used_local
          偏大亚帧~1 帧; 更重要的是原 get_observation 内部 6 路并发各通道也存在时刻
          错位, 会直接进模型)。

        返回 dict (字段与 ``get_observation`` 一致, 多一个 "chunk_progress"):
            "head_rgb", "fish_left", "fish_right",
            "hand_left_rgb", "hand_right_rgb": numpy BGR 或 None  (base64 已还原)
            "joints": {...}                                       (与 get_joint_states 同结构)
            "chunk_progress": {"arm": float|None, "hand": ..., "eef": ...}
            "timestamp": float

        Args:
            hand_rad: True 时把 hand position 从 actuator(0-4096) 转弧度
            timeout:  HTTP 超时 (秒); base64 5 图打包后体积较大, 默认 3s 比单图宽松

        server 不支持该 endpoint (老版本 a2_server) 时回退到旧路径:
            get_observation() + get_chunk_progress(), 调用方无感。
        """
        import base64
        try:
            resp = self.session.get(
                f"{self.base_url}/get_observation_with_progress", timeout=timeout)
            if resp.status_code != 200:
                return self._fallback_obs_with_progress(hand_rad)
            data = resp.json()
        except Exception:
            return self._fallback_obs_with_progress(hand_rad)

        def _b64_to_img(b64):
            if b64 is None:
                return None
            try:
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                return None

        joints = data.get("joints") or {}
        if hand_rad and joints and joints.get("hand"):
            raw = joints["hand"].get("position")
            if raw and len(raw) == 20:
                left_rad = OMNIHAND_LEFT.actuator_to_radians(raw[:10])
                right_rad = OMNIHAND_RIGHT.actuator_to_radians(raw[10:])
                joints = dict(joints)
                joints["hand"] = dict(joints["hand"])
                joints["hand"]["position"] = left_rad + right_rad

        return {
            "head_rgb": _b64_to_img(data.get("head_rgb")),
            "fish_left": _b64_to_img(data.get("fish_left")),
            "fish_right": _b64_to_img(data.get("fish_right")),
            "hand_left_rgb": _b64_to_img(data.get("hand_left_rgb")),
            "hand_right_rgb": _b64_to_img(data.get("hand_right_rgb")),
            "joints": joints,
            "chunk_progress": data.get("chunk_progress"),
            "timestamp": data.get("timestamp", time.time()),
        }

    def _fallback_obs_with_progress(self, hand_rad: bool) -> Optional[dict]:
        """老版本 a2_server 无 /get_observation_with_progress 时的回退路径。

        退化为 get_observation() + get_chunk_progress() 两次采样 (有错位, 但可用)。
        """
        obs = self.get_observation(hand_rad=hand_rad)
        if obs is None:
            return None
        prog = self.get_chunk_progress()
        obs["chunk_progress"] = prog
        return obs


    def measure_rtt(self, num_samples: int = 30, warmup: int = 5,
                    timeout: float = 1.0) -> Optional[float]:
        """启动期探测 client→server→client HTTP 往返延迟 (秒).

        用 server 端 /ping (lock-free 轻量) 打多次取 median.
        前 warmup 次丢弃 (TCP keepalive / connection pool 冷启动 jitter).

        Returns:
            median round-trip time in seconds, or None if all attempts failed.
            返回的是 RTT (round-trip), 上游用 RTT/2 估单向 HTTP 延迟即可.
        """
        import time as _time
        rtts = []
        n = max(1, int(num_samples) + max(0, int(warmup)))
        for i in range(n):
            t0 = _time.monotonic()
            try:
                resp = self.session.get(f"{self.base_url}/ping", timeout=timeout)
                t1 = _time.monotonic()
                if resp.status_code != 200:
                    continue
            except Exception:
                continue
            if i < warmup:
                continue
            rtts.append(t1 - t0)
        if not rtts:
            return None
        rtts.sort()
        return rtts[len(rtts) // 2]

    def pause_chunk(self, timeout: float = 0.2) -> bool:
        """冻结 server 端 chunk 时钟. 电机停在当前位置, chunk_progress 也冻结."""
        try:
            resp = self.session.post(f"{self.base_url}/pause_chunk", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def resume_chunk(self, timeout: float = 0.2) -> bool:
        try:
            resp = self.session.post(f"{self.base_url}/resume_chunk", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def start_video_record(self, fps: float = 20.0):
        try:
            self.session.post(f"{self.base_url}/start_video_record",
                              json={"fps": fps}, timeout=self.timeout)
        except Exception:
            pass

    def stop_video_record(self) -> dict:
        try:
            resp = self.session.post(f"{self.base_url}/stop_video_record",
                                     json={}, timeout=10.0)
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    def _pick_reset_hand_pose(self, hand_pose: str, hand_open):
        """挑 reset() 时要发送给 hand 的 20D actuator 值, 子类可 override
        来支持别的末端 (如 A3 的 AgiClaw 夹爪 2D)。

        返回 (values, label_for_print)。
        """
        if hand_pose == "fist":
            return VLA_HAND_FIST_POS, "握拳"
        return hand_open, "张开手"

    def reset(self, hand_pose: str = "open", arm_pose: str = "up",
              steps: int = 100, fps: float = 150, init_pose: int = 1):
        """复位手臂和手部到初始位姿。

        控制模式 (joint / eef) 由 `set_action_mode()` 设置; 调用前应先设置好。

        Args:
            hand_pose:    "open" (张开手) 或 "fist" (握拳)
            arm_pose:     "up" (默认, 手臂抬起, VLA_ARM_INIT_UP_POS) 或
                          "down" (手臂放下, VLA_ARM_INIT_POS)
            steps:        插值步数
            fps:          控制频率 (Hz)
            init_pose:    0 = ckpt 数据驱动 (config.VLA_HAND_INIT_POS_DATA, 由
                          UI load_policy_from_config 从 <ckpt>/assets/init.parquet
                          frame 0 填充; None 则 fallback 到 pose 1),
                          1 = 默认初始位姿, 2 = task_9081 数据集初始位姿,
                          3 = single_robot_stamp_demo530_trimmed 数据集初始位姿
        """
        from tqdm import tqdm

        control_mode = self._action_mode

        # 等待关节状态可用
        for _ in range(50):
            js = self.get_joint_states()
            if js and js.get("arm"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("未获取到关节状态")

        # 选择目标位姿
        if init_pose == 0:
            # 引用一次性读, 别用 module-level cache (load_policy_from_config 后才填)
            import config as _cfg
            arm_init = VLA_ARM_INIT_POS  # 手臂沿用默认 (data init.parquet 不一定有可信 arm)
            hand_open = _cfg.VLA_HAND_INIT_POS_DATA or VLA_HAND_INIT_POS
            if _cfg.VLA_HAND_INIT_POS_DATA is None:
                print("  [reset] WARN: ckpt 没有 init.parquet, Pose Data fallback 到 Pose 1")
        elif init_pose == 2:
            arm_init = VLA_ARM_INIT_POS_2
            hand_open = VLA_HAND_INIT_POS_2
        elif init_pose == 3:
            arm_init = VLA_ARM_INIT_POS_3
            hand_open = VLA_HAND_INIT_POS_3
        else:
            arm_init = VLA_ARM_INIT_POS
            hand_open = VLA_HAND_INIT_POS

        # 手部复位 (子类可 override _pick_reset_hand_pose 替换 hand 维度，
        # 例如 A3 gripper 用 VLA_GRIPPER_OPEN_POS/CLOSE_POS)
        hand_init, hand_label = self._pick_reset_hand_pose(hand_pose, hand_open)
        print(f"发送手部初始位 ({hand_label}, pose={init_pose})...")
        self.send_hand(hand_init)
        time.sleep(0.1)

        steps = max(1, steps)

        # 选择手臂目标姿态: 默认 up, 显式 "down" 才用 VLA_ARM_INIT_POS
        if arm_pose == "down":
            arm_target_pos = VLA_ARM_INIT_POS
            arm_pose_label = "放下"
        else:
            arm_target_pos = VLA_ARM_INIT_UP_POS
            arm_pose_label = "抬起"

        if control_mode == "joint":
            js = self.get_joint_states()
            arm_current = np.asarray(js["arm"]["position"], dtype=float)
            arm_target = np.asarray(arm_target_pos, dtype=float)
            print(f"关节角插值复位 ({arm_pose_label}): {steps} 步, {fps} Hz")
            t0 = time.time()
            for i in tqdm(range(1, steps + 1), desc="手臂复位中..."):
                alpha = i / steps
                self.send_arm(((1 - alpha) * arm_current + alpha * arm_target).tolist(), steps=1)
                time.sleep(1.0 / fps)
            print(f"手臂复位完成, 耗时: {time.time() - t0:.2f}s")

        elif control_mode == "eef":
            eef = self.get_eef_state()
            if eef is None:
                raise RuntimeError("未获取到 EEF 状态")

            def extract(d):
                p, o = d["position"], d["orientation"]
                return np.array([p["x"], p["y"], p["z"]]), np.array([o["x"], o["y"], o["z"], o["w"]])

            # 按 arm_pose 选择 EEF 目标 (与 joint 路径一致: up 默认, down 显式)
            if arm_pose == "down":
                left_eef_target = VLA_LEFT_EEF_DOWN_TARGET
                right_eef_target = VLA_RIGHT_EEF_DOWN_TARGET
            else:
                left_eef_target = VLA_LEFT_EEF_UP_TARGET
                right_eef_target = VLA_RIGHT_EEF_UP_TARGET

            lpc, lqc = extract(eef["left"])
            rpc, rqc = extract(eef["right"])
            lpt, lqt = extract(left_eef_target)
            rpt, rqt = extract(right_eef_target)

            print(f"EEF 插值复位 ({arm_pose_label}): {steps} 步, {fps} Hz")
            t0 = time.time()
            for i in tqdm(range(1, steps + 1), desc="手臂复位中..."):
                alpha = i / steps
                lp = (1 - alpha) * lpc + alpha * lpt
                rp = (1 - alpha) * rpc + alpha * rpt
                lq = slerp(lqc, lqt, alpha)
                rq = slerp(rqc, rqt, alpha)
                eef_14d = list(lp) + list(rp) + list(lq) + list(rq)
                self.send_eef(eef_14d)
                time.sleep(1.0 / fps)
            print(f"手臂复位完成, 耗时: {time.time() - t0:.2f}s")

        print("复位完成")

    def get_observation(self, hand_rad: bool = False) -> dict:
        """获取全部观测数据: 五路相机 + 关节状态。并发发送 6 路 HTTP 请求。

        Args:
            hand_rad: 若为 True, 将 hand position 从 actuator 值 (0-4096) 转为弧度

        返回 dict:
            "head_rgb", "fish_left", "fish_right",
            "hand_left_rgb", "hand_right_rgb":  numpy BGR 或 None
            "joints":  完整关节状态 dict, 格式如下:
                {
                    "arm":   {"position": [...], "velocity": [...], "effort": [...]},
                    "hand":  {"position": [...], "velocity": [...], "effort": [...]},
                    "neck":  {"position": [...], "velocity": [...], "effort": [...]},
                    "waist": {"position": [...]},
                    "leg":   {"position": [...], "velocity": [...], "effort": [...]},
                    "eef":   {"left": {...}, "right": {...}},
                }
            "timestamp":  float
        """
        # 并发发 6 路请求 (5 图 + 1 关节), 避免串行累加 HTTP 延迟
        futures = {
            "head_rgb": self._obs_executor.submit(self._get_image, "/get_head_rgb"),
            "fish_left": self._obs_executor.submit(self._get_image, "/get_fish_left"),
            "fish_right": self._obs_executor.submit(self._get_image, "/get_fish_right"),
            "hand_left_rgb": self._obs_executor.submit(self._get_image, "/get_hand_left_rgb"),
            "hand_right_rgb": self._obs_executor.submit(self._get_image, "/get_hand_right_rgb"),
            "joints": self._obs_executor.submit(self.get_joint_states),
        }
        results = {k: f.result() for k, f in futures.items()}
        joints = results["joints"]

        if hand_rad and joints and joints.get("hand"):
            raw = joints["hand"].get("position")
            if raw and len(raw) == 20:
                left_rad = OMNIHAND_LEFT.actuator_to_radians(raw[:10])
                right_rad = OMNIHAND_RIGHT.actuator_to_radians(raw[10:])
                joints["hand"]["position"] = left_rad + right_rad

        return {
            "head_rgb": results["head_rgb"],
            "fish_left": results["fish_left"],
            "fish_right": results["fish_right"],
            "hand_left_rgb": results["hand_left_rgb"],
            "hand_right_rgb": results["hand_right_rgb"],
            "joints": joints or {},
            "timestamp": time.time(),
        }

    # ==================== 各部位控制 (默认带插值) ====================

    def send_arm(self, arm_values, steps: int = 100, fps: float = 150.0) -> bool:
        """发送手臂命令，按 `set_action_mode()` 设置的模式路由。

        - JOINT 模式: 14D 关节角弧度, 默认从当前位姿线性插值到目标
        - EEF 模式:   14D 末端位姿 (pos6 + quat8), 直接发送 (不做本地插值;
                       四元数需 SLERP, 如需平滑请用 `reset()` 或自行插值)

        Args:
            arm_values: 14D 值 (含义随模式)
            steps:      插值步数 (仅 joint 模式有效), 1 = 不插值直接发送
            fps:        插值控制频率 (Hz, 仅 joint 模式有效)
        """
        if self._action_mode == "eef":
            return self._send_eef_direct(arm_values)

        if steps > 1:
            if self._last_arm is None:
                js = self.get_joint_states()
                if js and js.get("arm"):
                    self._last_arm = list(js["arm"]["position"])

            if self._last_arm is not None:
                current = np.array(self._last_arm, dtype=float)
                target = np.array([float(x) for x in arm_values], dtype=float)
                interval = 1.0 / fps
                for i in range(1, steps + 1):
                    alpha = i / steps
                    vals = ((1 - alpha) * current + alpha * target).tolist()
                    self._send_arm_direct(vals)
                    time.sleep(interval)
                self._last_arm = [float(x) for x in arm_values]
                return True

        ok = self._send_arm_direct(arm_values)
        if ok:
            self._last_arm = [float(x) for x in arm_values]
        return ok

    def send_hand(self, hand_values, steps: int = 100, fps: float = 150.0, effort=None) -> bool:
        """发送手部命令 (20D, 原始值 0-4096)，默认从当前位姿插值到目标。

        Args:
            hand_values: 20D actuator 值 (0-4096)
            steps: 插值步数, 1 = 不插值直接发送
            fps: 插值控制频率 (Hz)
            effort: 20D 力矩 (0-255), 不传默认 100
        """
        if steps > 1:
            if self._last_hand is None:
                js = self.get_joint_states()
                if js and js.get("hand"):
                    self._last_hand = list(js["hand"]["position"])

            if self._last_hand is not None:
                current = np.array(self._last_hand, dtype=float)
                target = np.array([float(x) for x in hand_values], dtype=float)
                interval = 1.0 / fps
                for i in range(1, steps + 1):
                    alpha = i / steps
                    vals = [int(round(x)) for x in (1 - alpha) * current + alpha * target]
                    self._send_hand_direct(vals, effort=effort)
                    time.sleep(interval)
                self._last_hand = [float(x) for x in hand_values]
                return True

        ok = self._send_hand_direct(hand_values, effort=effort)
        if ok:
            self._last_hand = [float(x) for x in hand_values]
        return ok

    def send_hand_radians(self, radians_20d, steps: int = 100, fps: float = 150.0, effort=None) -> bool:
        """将 20D 弧度转为 actuator 值后发送

        Args:
            radians_20d: 20D 弧度值
            steps: 插值步数, 1 = 不插值直接发送
            fps: 插值控制频率 (Hz)
            effort: 20D 力矩 (0-255), 不传默认 100
        """
        left_raw = OMNIHAND_LEFT.radians_to_actuator(list(radians_20d[:10]))
        right_raw = OMNIHAND_RIGHT.radians_to_actuator(list(radians_20d[10:]))
        return self.send_hand(left_raw + right_raw, steps=steps, fps=fps, effort=effort)

    def send_eef(self, eef_values) -> bool:
        """发送 EEF 位姿 (14D), 绕过服务端插值"""
        return self._send_eef_direct(eef_values)

    def send_head(self, shake: float, nod: float) -> bool:
        try:
            resp = self._post("/send_head", {"shake": float(shake), "nod": float(nod)})
            return resp.status_code == 200
        except Exception:
            return False

    def send_loco(self, forward: float, lateral: float, angular: float, mode: int = 1) -> bool:
        try:
            resp = self._post("/send_loco", {"forward": float(forward), "lateral": float(lateral),
                                              "angular": float(angular), "mode": mode})
            return resp.status_code == 200
        except Exception:
            return False

    def send_waist(self, z: float = 0, pitch: float = 0, yaw: float = 0,
                   x: float = 0, y: float = 0, roll: float = 0) -> bool:
        try:
            resp = self._post("/send_waist", {"x": float(x), "y": float(y), "z": float(z),
                                               "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)})
            return resp.status_code == 200
        except Exception:
            return False

    # ==================== 状态获取 ====================

    def get_joint_states(self) -> Optional[dict]:
        try:
            resp = self._get("/get_joint_states")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def get_eef_state(self) -> Optional[dict]:
        try:
            resp = self._get("/get_eef_state")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def close(self):
        try:
            self._obs_executor.shutdown(wait=False)
        except Exception:
            pass
        self.session.close()


class A3RobotInterface(A2RobotInterface):
    """A3 机器人高层 HTTP 客户端。

    与 A2 的区别:
      - A3 相机是 sensor_msgs/Image (server 端 cv_bridge 解码后再用 JPEG 下发),
        共 10 路: head_stereo_left/right, head_left/right/rear, chest_front,
        waist_front, wrist_left/right, armpit_right. get_observation() 会并发
        把这 10 路全部取回, 同时保留 A2 字段 (head_rgb / fish_left / ...) 为 None,
        让上层共用代码 (A2 字段缺失时上层应已经 fallback)。
      - 新增 get_ta_whole_body_command()：查询 server 端缓存的最近一帧
        TA 全身指令（来自 /ta/whole_body_command），已 protobuf parse 成 dict。
      - 末端按 ``hand_kind`` 切换 O10 灵巧手 (20D, 默认) 或 AgiClaw 双指夹爪 (2D)。
        必须与 server 启动时 ``ros2 run a3_server server -- --hand-kind <hand|gripper>``
        保持一致。
      - 腰部走 protobuf MotionControlMoveWaistChannel, 4D [pitch, roll, yaw, height]。
        ``send_waist(values)`` 兼容 state 顺序 [yaw, roll, pitch, (height)] 输入。
      - 腿部命令走 sensor_msgs/JointState, topic `/body_drive/leg_joint_command_ros2`
        (当前固件可能尚未启用)。

    用法 (VLA 推理):
        robot = A3RobotInterface("10.42.10.11", hand_kind="hand")   # A3 server 跑在 ADU
        obs = robot.get_observation()
        head = obs["head_stereo_left"]   # numpy BGR
        ta = robot.get_ta_whole_body_command()
        action = model.predict(obs)
        robot.step(action)
    """

    # 与 a3_server.server_node.CAMERA_TOPICS 一一对应
    CAMERA_NAMES = (
        "head_stereo_left", "head_stereo_right",
        "head_left", "head_right", "head_rear",
        "chest_front", "waist_front",
        "wrist_left", "wrist_right",
        "armpit_right",
    )

    HAND_KINDS = ("hand", "gripper")

    def __init__(self, robot_ip: str = "192.168.2.50", port: int = 5050,
                 timeout: float = 2.0, target_fps: float = 150.0,
                 hand_kind: str = "hand"):
        if hand_kind not in self.HAND_KINDS:
            raise ValueError(f"hand_kind must be one of {self.HAND_KINDS}, got {hand_kind!r}")
        self.hand_kind = hand_kind
        # gripper 是 2D, 灵巧手是 20D; 用于 step() 时跳过 rad→actuator 转换。
        self.hand_dim = 2 if hand_kind == "gripper" else 20

        super().__init__(robot_ip=robot_ip, port=port, timeout=timeout, target_fps=target_fps)

        # A3 手臂控制模式 — bool, 决定 send_arm / send_eef 走哪条通道:
        #   False (默认, "不插值")  pnc_arm 处于 PASSIVE 模式, send_arm 走
        #                          /send_arm (mc 通道, 不插值, 直接透传)。
        #                          send_eef 不可用 (mc 通道无 EEF)。
        #   True ("插值")          pnc_arm 切到 ONLINE_TRAJECTORY 模式,
        #                          send_arm 走 /send_arm_interp (PncArmInterpolateChannel,
        #                          机上插值), send_eef 走 /send_eef_interp (同 channel SE3)。
        # 调 set_mode(interp=True/False) 切换。
        self._interp_mode = False

        # 重建线程池: 10 路相机 + 1 路关节, 比 A2 大
        self._obs_executor.shutdown(wait=False)
        from concurrent.futures import ThreadPoolExecutor
        self._obs_executor = ThreadPoolExecutor(
            max_workers=len(self.CAMERA_NAMES) + 2,
            thread_name_prefix="a3_obs",
        )
        # HTTPAdapter 池也同步加大, 让 10 路图像真正并发
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def list_cameras(self) -> Optional[dict]:
        """查询 server 端订阅的相机列表。"""
        try:
            resp = self._get("/list_cameras")
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    # ==================== A3 手臂控制模式 (插值 / 不插值) ====================

    def set_action_mode(self, mode: str, timeout: float = 180.0) -> bool:
        """A3 上不使用父类的 change_action_mode_a2.sh (那是 A2 Orin→x86 跳板)。

        想切插值 / 不插值控制请用 set_mode(interp=True|False)。
        """
        raise NotImplementedError(
            "A3 不用 set_action_mode (A2 专属); 请用 set_mode(interp=True|False)"
        )

    def set_mode(self, interp: bool, timeout: float = 60.0) -> bool:
        """设置 A3 手臂控制模式 (插值 vs 不插值)。

        - interp=False (默认 boot 状态): 不插值。send_arm 直接发到 mc 的
          /motion/control/pnc/arm_joint_command (joint_msgs/JointCommand),
          命令 100% 透传, 没有任何平滑。
          send_eef 不可用 (mc 通道没有 EEF)。
        - interp=True: 机上插值。send_arm / send_eef 都走 pnc_arm 的
          /pnc_arm/motion/interpolate (PncArmInterpolateChannel), pnc_arm 模块
          自己做高频平滑。EEF 控制只在此模式下可用。

        切换通过本地 scripts/utils/change_pnc_arm_mode_a3.sh ssh 到 ADU 跑
        S_SetControlMode.py 改 pnc_arm 模块的内部模式
        (PASSIVE ↔ ONLINE_TRAJECTORY), 耗时 ~2-5 秒。本地需安装 sshpass。

        Args:
            interp:  True = 启用机上插值 (pnc_arm trajectory); False = 关闭 (pnc_arm passive)
            timeout: 子进程最长等待秒数

        Returns: True = 切换成功或已是目标模式; False = 失败 (内部状态不更新)
        """
        target_alias = "trajectory" if interp else "passive"
        import os
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "scripts", "utils", "change_pnc_arm_mode_a3.sh")
        if not os.path.isfile(script):
            print(f"[set_mode] 找不到脚本: {script}")
            return False
        try:
            result = subprocess.run(["bash", script, target_alias], timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[set_mode] 超时 ({timeout}s), 切换未完成")
            return False
        if result.returncode != 0:
            return False
        self._interp_mode = bool(interp)
        return True

    # ==================== 覆写 send_arm / send_eef 按 _interp_mode 路由 ====================

    def send_arm(self, arm_values, steps: int = 100, fps: float = 150.0) -> bool:
        """发送 14D 双臂命令, 按 set_mode 设置的模式路由:

        - 插值模式  (set_mode(True)):  → /send_arm_interp (机上插值, 不本地插值)
        - 不插值模式 (set_mode(False)): → /send_arm?raw    (机上不插值, 本地仍按 steps 插值)

        EEF 控制走 send_eef(...) (只在插值模式下可用)。
        参数 steps/fps 仅不插值模式生效, 插值模式下机上自己插值, 上层
        按推理频率发即可。
        """
        if self._interp_mode:
            try:
                resp = self._post("/send_arm_interp",
                                   {"values": [float(x) for x in arm_values], "flag": 2})
                ok = resp.status_code == 200
                if ok:
                    self._last_arm = [float(x) for x in arm_values]
                return ok
            except Exception:
                return False
        # 不插值模式: 走父类 (mc /send_arm?raw + 本地插值)
        return super().send_arm(arm_values, steps=steps, fps=fps)

    def send_eef(self, eef_values) -> bool:
        """发送 14D EEF 双臂位姿 (left_pos+left_quat + right_pos+right_quat)。

        A3 上 EEF 只能在插值模式 (set_mode(True)) 下用 (走 /send_eef_interp,
        flag=102 SE3 双臂)。不插值模式直接拒绝并提示。
        """
        if not self._interp_mode:
            print("[send_eef] A3 EEF 控制需要先 set_mode(interp=True)")
            return False
        try:
            resp = self._post("/send_eef_interp",
                               {"values": [float(x) for x in eef_values], "flag": 102})
            return resp.status_code == 200
        except Exception:
            return False

    def get_camera(self, name: str) -> Optional[np.ndarray]:
        """取单路相机的最新一帧 BGR ndarray。"""
        return self._get_image(f"/get_camera/{name}")

    # NOTE: get_observation() moved further down in the class — the newer
    # version fetches IMU concurrently for sonic-a3. Keeping only one
    # definition (the last-defined one below would silently override anyway
    # but the duplicate was confusing to read).

    def get_ta_whole_body_command(self) -> Optional[dict]:
        """查询 server 端缓存的最近一帧 TA 全身指令。

        server 端订阅 /ta/whole_body_command/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel,
        把 RosMsgWrapper.data (序列化后的 TaWholeBodyCommandChannel) 用 protobuf
        反序列化后缓存; 该接口直接返回 MessageToDict 的结果。

        返回 dict (keys 与 .proto 一致):
            {
              "timestamp": <wall time, float>,
              "data_len":  <原始字节长度>,
              "decoded":   {
                "header": {"seq": ..., "timestamp": {...}, "frame_id": "..."},
                "data":   {  # TaWholeBodyCommand
                    "joint_layout":     "TaJointLayout_BODY_31" | "..._BODY_HANDS_55" | ...,
                    "pelvis_pose":      {"quat_wxyz":[...], "position_xyz":[...]},
                    "pelvis_velocity":  {"linear_xyz":[...], "angular_xyz":[...]},
                    "leg_command":      {"angles_rad":[12]},
                    "foot_contact":     {...},
                    "waist_command":    {"angles_rad":[3]},
                    "head_command":     {"angles_rad":[2]},
                    "arm_command":      {"angles_rad":[14], "velocities_rad_s":[14], ...},
                    "left_hand_command":  {"angles_rad":[12]},
                    "right_hand_command": {"angles_rad":[12]},
                    "joint_velocities":   {"velocities_rad_s":[31]},
                }
              }
            }
        没收到过任何 TA 帧时返回 None；server 端缺 ta_channel_pb2 也会返回 None。
        """
        try:
            resp = self._get("/get_ta_whole_body_command")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    # ==================== A3 专属命令 ====================

    def send_waist(self, *args, pitch: Optional[float] = None,
                   roll: Optional[float] = None,
                   yaw: Optional[float] = None,
                   height: Optional[float] = None,
                   **_legacy_kwargs) -> bool:
        """发送 A3 腰部命令 (走 protobuf MotionControlMoveWaistChannel)。

        参数范围 (与 a3u_tool/T_WaistMove.py 一致):
            pitch:  [-0.5, 0.5]   rad
            roll:   [-0.3, 0.3]   rad
            yaw:    [-1.57, 1.57] rad
            height: [-0.4, 0.0]   m

        三种调用方式:
            (a) 位置参数 + state 顺序的 list/tuple/np.ndarray:
                send_waist([yaw, roll, pitch])             # 长度 3, height 默认 0
                send_waist([yaw, roll, pitch, height])     # 长度 4
                与 /motion/control/waist_joint_state 输出顺序一致, 调试时
                ``s = robot.get_joint_states()["waist"]["position"]; s[2]+=0.1;
                robot.send_waist(s)`` 直接能用。

            (b) 关键字参数:
                send_waist(pitch=0.1, yaw=0.0, roll=0.0, height=0.0)

            (c) 兼容 A2 旧 6D [x,y,z,roll,pitch,yaw]: 仅取 roll/pitch/yaw, z 当 height,
                x/y 忽略 (A3 上腰部没有平移自由度)。
        """
        if args:
            v = args[0]
            try:
                v_list = list(v)
            except TypeError:
                raise TypeError(f"send_waist 位置参数应为 list/tuple/np.ndarray, 得到 {type(v).__name__}")
            n = len(v_list)
            if n == 3:
                payload = {"values": [float(v_list[0]), float(v_list[1]), float(v_list[2])]}
            elif n == 4:
                payload = {"values": [float(x) for x in v_list]}
            elif n == 6:
                # A2 6D: [x, y, z, roll, pitch, yaw]; z -> height
                payload = {"pitch": float(v_list[4]), "roll": float(v_list[3]),
                           "yaw": float(v_list[5]), "height": float(v_list[2])}
            else:
                raise ValueError(f"send_waist 位置参数长度需为 3/4/6, 得到 {n}")
        else:
            payload = {}
            if pitch is not None:  payload["pitch"]  = float(pitch)
            if roll is not None:   payload["roll"]   = float(roll)
            if yaw is not None:    payload["yaw"]    = float(yaw)
            if height is not None: payload["height"] = float(height)
            # 兼容旧调用 send_waist(z=, pitch=, yaw=) — z 当 height
            if "z" in _legacy_kwargs and "height" not in payload:
                payload["height"] = float(_legacy_kwargs["z"])
            if not payload:
                raise ValueError("send_waist 未提供任何参数")
        try:
            resp = self._post("/send_waist", payload)
            return resp.status_code == 200
        except Exception:
            return False

    def send_leg(self, values) -> bool:
        """A3 腿部命令: sensor_msgs/JointState 走 /body_drive/leg_joint_command_ros2。

        values: 长度 12, 顺序对应 server 端 LEG_JOINT_NAMES。
        当前固件版本可能尚未启用该 topic; 若发不动检查 `ros2 topic info` 订阅数。
        """
        try:
            resp = self._post("/send_leg", {"values": [float(x) for x in values]})
            return resp.status_code == 200
        except Exception:
            return False

    def send_loco(self, forward: float, lateral: float, angular: float,
                  mode: int = 0) -> bool:
        """A3 行走速度命令; mode 默认 0 (=默认), 1 (=导航), 与 T_LocomotionVelocity 一致。"""
        try:
            resp = self._post("/send_loco", {"forward": float(forward),
                                              "lateral": float(lateral),
                                              "angular": float(angular),
                                              "mode": int(mode)})
            return resp.status_code == 200
        except Exception:
            return False

    # ==================== step / step_chunk: A3 专属路由 ====================

    def _hand_chunk_to_actuator(self, values, hand_value: str) -> np.ndarray:
        """Normalize an A3 hand chunk to the actuator values used on ROS topics."""
        hand_2d = np.asarray(values, dtype=float)
        if hand_2d.ndim != 2 or hand_2d.shape[1] != self.hand_dim:
            raise ValueError(
                f"{self.hand_kind} chunk must be (H,{self.hand_dim}), "
                f"got {hand_2d.shape}"
            )

        hand_value = str(hand_value).lower()
        if hand_value not in ("raw", "rad"):
            raise ValueError(f"hand_value must be 'raw' or 'rad', got {hand_value!r}")
        if self.hand_kind != "hand" or hand_value == "raw":
            return hand_2d

        converted = []
        for row in hand_2d:
            left_raw = OMNIHAND_LEFT.radians_to_actuator(row[:10])
            right_raw = OMNIHAND_RIGHT.radians_to_actuator(row[10:])
            converted.append(left_raw + right_raw)
        return np.asarray(converted, dtype=float)

    def _pick_reset_hand_pose(self, hand_pose: str, hand_open):
        """A3 hand_kind=gripper 时改用 VLA_GRIPPER_OPEN_POS / CLOSE_POS (2D)。

        hand_pose:
            "open"  → VLA_GRIPPER_OPEN_POS  ([4096, 4096], 张开)
            "fist"  → VLA_GRIPPER_CLOSE_POS ([0, 0], 闭合)
        hand_kind=hand 时直接复用父类逻辑 (20D O10 灵巧手)。
        """
        if self.hand_kind == "gripper":
            from config import VLA_GRIPPER_OPEN_POS, VLA_GRIPPER_CLOSE_POS
            if hand_pose == "fist":
                return list(VLA_GRIPPER_CLOSE_POS), "夹爪闭合"
            return list(VLA_GRIPPER_OPEN_POS), "夹爪张开"
        return super()._pick_reset_hand_pose(hand_pose, hand_open)

    def step(self, action: dict, wait=None, settle_ms=None) -> None:
        """A3 step: 在 super().step 前把 A3 专属的 waist/leg 命令发掉,
        并在 hand_kind=gripper 时强制 hand_value=raw (gripper 没有 rad 含义)。
        """
        if "waist" in action or "leg" in action or self.hand_kind == "gripper":
            action = dict(action)
            if "waist" in action:
                self.send_waist(action.pop("waist"))
            if "leg" in action:
                self.send_leg(action.pop("leg"))
            if self.hand_kind == "gripper" and action.get("hand_value") == "rad":
                action["hand_value"] = "raw"
        super().step(action, wait=wait, settle_ms=settle_ms)

    def step_chunk(self, chunk: dict, chunk_fps: float = 30.0,
                   wait: bool = True, settle_ms: float = 0.0,
                   timeout_ms: float = 30000.0,
                   **kwargs):
        """A3 step_chunk. Two modes, auto-detected by chunk keys:

        (A) legacy A2-inherited path: chunk has arm/hand/eef (and optionally
            waist/leg via the old per-part endpoints). Falls through to
            super().step_chunk. waist/leg still go through /send_waist and
            /send_leg respectively for backward-compat.

        (B) sonic-a3 whole-body path (RECOMMENDED for VLA inference): chunk
            has pelvis_quat_wxyz + arm + leg + waist + optional s_used_local.
            Server does atomic swap + publishes the chunk to the robot via one
            of two wire paths (see emit_mode). Return value is a dict with
            server-computed actual_delay etc. (matches a2_server.swap_arm_chunk_atomic response).

        Auto-detection: if the chunk contains 'pelvis_quat_wxyz' OR the
        caller passes 's_used_local' in kwargs, we go through mode (B).

        kwargs supports (for mode B):
            s_used_local:   int, for server-atomic swap actual_delay math
                            in the pre-upsampling source/policy coordinate.
            s_used_local_wire: exact wire-buffer counterpart of s_used_local.
                            Optional transport detail used when source and wire
                            fps differ; normal RTC callers need not use it.
            source_fps:     pre-upsampling action rate.  With a wire-rate
                            payload, server returns actual_delay in this
                            source coordinate plus actual_delay_wire for
                            internal buffer bookkeeping.
            chunk_id:       int, echoed back for diagnostics
            adaptive_transition: bool, off by default (sonic smooths downstream)
            emit_mode:      "ta_cmd" (default) → 60Hz single-frame
                            /ta/whole_body_command (MC consumes directly);
                            "reference_window" → 50Hz 10-frame
                            /gr00t/reference_window (sonic consumes).
                            Both share the same chunk buffer + swap, so
                            actual_delay is identical.
            ta_cmd_hz:      float, emit rate for the ta_cmd path (default 60).
                            Server linearly interpolates between chunk
                            waypoints (chunk_fps, e.g. 20/30Hz) up to this.
        """
        # ---- (C) UPPER-BODY server-atomic chunk (arm + hand + waist) ----
        # Explicit mode="upper_body": send arm(H,14)+hand(H,20/2)+waist(H,4) to
        # a3_server /send_chunk with s_used_local. Handled BEFORE the wb_mode
        # heuristic so s_used_local here does NOT route into the whole-body path
        # (which needs leg+pelvis). Returns {"ok", "actual_delay"}.
        if kwargs.get("mode") == "upper_body":
            return self._send_upper_chunk(
                chunk, chunk_fps=chunk_fps, wait=wait,
                settle_ms=settle_ms, timeout_ms=timeout_ms, **kwargs)

        wb_mode = (
            "pelvis_quat_wxyz" in chunk
            or kwargs.get("s_used_local") is not None
            or (kwargs.get("mode") == "whole_body")
        )
        if not wb_mode:
            # Legacy path — original A2-inherited behaviour.
            if "waist" in chunk or "leg" in chunk or self.hand_kind == "gripper":
                chunk = dict(chunk)
                if "waist" in chunk:
                    w = chunk.pop("waist")
                    w_arr = np.asarray(w, dtype=float)
                    if w_arr.ndim == 1:
                        self.send_waist(w_arr.tolist())
                    else:
                        import threading as _th
                        def _waist_pace(rows, fps):
                            interval = 1.0 / max(fps, 1.0)
                            t0 = time.time()
                            for i, row in enumerate(rows):
                                target = t0 + i * interval
                                sleep_s = target - time.time()
                                if sleep_s > 0:
                                    time.sleep(sleep_s)
                                self.send_waist(list(row))
                        _th.Thread(target=_waist_pace,
                                   args=(w_arr.tolist(), chunk_fps),
                                   daemon=True).start()
                if "leg" in chunk:
                    leg = np.asarray(chunk.pop("leg"), dtype=float)
                    if leg.ndim == 1:
                        self.send_leg(leg.tolist())
                    else:
                        self.send_leg(leg[-1].tolist())
                if self.hand_kind == "gripper" and chunk.get("hand_value") == "rad":
                    chunk["hand_value"] = "raw"
            # Strip A3-specific kwargs so super() doesn't choke.
            for _k in ("s_used_local", "s_used_local_wire", "source_fps", "chunk_id", "adaptive_transition",
                       "mode", "emit_mode", "ta_cmd_hz"):
                kwargs.pop(_k, None)
            return super().step_chunk(chunk, chunk_fps=chunk_fps, wait=wait,
                                      settle_ms=settle_ms, timeout_ms=timeout_ms,
                                      **kwargs)

        # ---- Mode (B): whole-body chunk → /send_chunk sonic path ----
        payload: dict = {
            "chunk_fps": float(chunk_fps),
            "mode": "whole_body",
        }
        # Required whole-body fields
        for k in ("arm", "leg", "waist", "pelvis_quat_wxyz"):
            if k not in chunk:
                raise ValueError(
                    f"whole-body step_chunk requires '{k}' in chunk dict; got keys={list(chunk)}"
                )
            payload[k] = np.asarray(chunk[k], dtype=float).tolist()
        # Optional hand chunk still routed (server sends on /motion/control/hand_joint_command).
        if "hand" in chunk:
            hand_raw = self._hand_chunk_to_actuator(
                chunk["hand"], chunk.get("hand_value", "raw")
            )
            payload["hand"] = hand_raw.tolist()
            # The a3_server and the machine ROS topic only accept actuator units.
            payload["hand_value"] = "raw"
            if "hand_effort" in chunk:
                payload["hand_effort"] = list(chunk["hand_effort"])
        # RTC swap fields
        if kwargs.get("s_used_local") is not None:
            payload["s_used_local"] = int(kwargs["s_used_local"])
        if kwargs.get("s_used_local_wire") is not None:
            payload["s_used_local_wire"] = int(kwargs["s_used_local_wire"])
        if kwargs.get("source_fps") is not None:
            payload["source_fps"] = float(kwargs["source_fps"])
        if kwargs.get("chunk_id") is not None:
            payload["chunk_id"] = int(kwargs["chunk_id"])
        if kwargs.get("adaptive_transition") is not None:
            payload["adaptive_transition"] = bool(kwargs["adaptive_transition"])
        # emit_mode selects the server's wire path for this chunk:
        #   "ta_cmd"           → 60Hz single-frame /ta/whole_body_command (MC direct)
        #   "reference_window" → 50Hz 10-frame /gr00t/reference_window (sonic)
        # Both share the same wb chunk buffer + s_used_local atomic swap, so the
        # returned actual_delay is identical. ta_cmd_hz tunes the /ta emit rate
        # (ta_cmd mode only); useful when chunk_fps is 20 or 30 — server
        # interpolates between waypoints up to ta_cmd_hz.
        if kwargs.get("emit_mode") is not None:
            payload["emit_mode"] = str(kwargs["emit_mode"])
        if kwargs.get("ta_cmd_hz") is not None:
            payload["ta_cmd_hz"] = float(kwargs["ta_cmd_hz"])

        # POST + return the server's response as a dict.
        query = "?"
        if wait:
            query += f"wait=true&settle_ms={settle_ms}&timeout_ms={timeout_ms}"
        try:
            resp = self.session.post(
                f"{self.base_url}/send_chunk{query if len(query) > 1 else ''}",
                json=payload,
                timeout=(timeout_ms + settle_ms + 5000.0) / 1000.0,
            )
            if resp.status_code != 200:
                return {"ok": False, "actual_delay": 0,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            body = resp.json()
            # server returns {"ok": True, "actual_delay": ..., ...} on wb path.
            return body if isinstance(body, dict) else {"ok": True, "actual_delay": 0}
        except Exception as e:
            return {"ok": False, "actual_delay": 0, "error": str(e)}

    def _send_upper_chunk(self, chunk: dict, chunk_fps: float = 30.0,
                          wait: bool = False, settle_ms: float = 0.0,
                          timeout_ms: float = 30000.0, **kwargs) -> dict:
        """UPPER-BODY server-atomic chunk send (arm + hand + waist).

        Posts to a3_server /send_chunk with ``mode="upper_body"``:
            arm   (H, 14)     joint radians   → JointCommand /motion/control/pnc/arm_joint_command
            hand  (H, 20/2)   actuator/raw    → mc hand / gripper command
            waist (H, 4)      [yaw,roll,pitch,height] → server-paced /send_waist
        plus ``s_used_local`` so the server computes ``actual_delay`` and slices
        all three atomically (see interp_publisher.swap_upper_chunk_atomic). This
        is the arm/hand equivalent of a2_server's server-atomic swap; waist is
        the A3-only addition. hand ``rad`` values are converted to actuator here
        (dexterous hand only; gripper stays raw).

        Returns the server dict {"ok", "actual_delay"} (actual_delay=0 on a
        server that predates this route, so the caller degrades to
        soft-continuous automatically).
        """
        payload: dict = {"mode": "upper_body", "chunk_fps": float(chunk_fps)}
        hand_value = chunk.get("hand_value", "raw")

        if chunk.get("arm") is not None:
            arm_2d = np.asarray(chunk["arm"], dtype=float)
            if arm_2d.ndim != 2:
                raise ValueError(f"upper arm chunk must be 2D, got {arm_2d.shape}")
            payload["arm"] = arm_2d.tolist()

        if chunk.get("waist") is not None:
            waist_2d = np.asarray(chunk["waist"], dtype=float)
            if waist_2d.ndim != 2 or waist_2d.shape[1] != 4:
                raise ValueError(
                    f"upper waist chunk must be (H,4) [yaw,roll,pitch,height], "
                    f"got {waist_2d.shape}")
            payload["waist"] = waist_2d.tolist()

        if chunk.get("hand") is not None:
            hand_raw = self._hand_chunk_to_actuator(chunk["hand"], hand_value)
            payload["hand"] = hand_raw.tolist()
            payload["hand_value"] = "raw"
            if chunk.get("hand_effort") is not None:
                payload["hand_effort"] = [float(x) for x in chunk["hand_effort"]]

        if kwargs.get("s_used_local") is not None:
            payload["s_used_local"] = int(kwargs["s_used_local"])
        if kwargs.get("chunk_id") is not None:
            payload["chunk_id"] = int(kwargs["chunk_id"])

        query = ""
        if wait:
            query = f"?wait=true&settle_ms={float(settle_ms)}&timeout_ms={float(timeout_ms)}"
        try:
            resp = self.session.post(
                f"{self.base_url}/send_chunk{query}", json=payload,
                timeout=(timeout_ms + settle_ms + 5000.0) / 1000.0,
            )
            if resp.status_code != 200:
                return {"ok": False, "actual_delay": 0,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            body = resp.json()
            return body if isinstance(body, dict) else {"ok": True, "actual_delay": 0}
        except Exception as e:
            return {"ok": False, "actual_delay": 0, "error": str(e)}
    def push_reference_window(self, window: dict, timeout_ms: float = 1000.0) -> dict:
        """直发一个 reference window 到 a3_server(/push_reference_window)。

        sim direct-50Hz replay 用:客户端已把 episode 插值到 50Hz 且自己滑窗,
        这里把一个(通常 10 帧)窗口原样 POST 给 server,server 立即组
        TaWholeBodyReferenceWindow 发到 /gr00t/reference_window(不插值/不 RTC)。
        发布节拍由调用方(driver 的 50Hz 循环)决定。

        window 必含: leg(H,12) waist(H,3) arm(H,14) pelvis_quat_wxyz(H,4)
        可选: dq31(H,31) 关节速度; head(H,2)。返回 server 的 dict {"ok", "frames"}。
        """
        payload: dict = {}
        for k in ("leg", "waist", "arm", "pelvis_quat_wxyz"):
            if k not in window:
                raise ValueError(
                    f"push_reference_window 需要 '{k}';got keys={list(window)}")
            payload[k] = np.asarray(window[k], dtype=float).tolist()
        for k in ("dq31", "head"):
            if window.get(k) is not None:
                payload[k] = np.asarray(window[k], dtype=float).tolist()
        try:
            resp = self.session.post(
                f"{self.base_url}/push_reference_window",
                json=payload,
                timeout=timeout_ms / 1000.0,
            )
            if resp.status_code != 200:
                return {"ok": False,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            body = resp.json()
            return body if isinstance(body, dict) else {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def push_token(self, token, timeout_ms: float = 1000.0) -> dict:
        """直发一个 64D token 到 a3_server(/push_token)。

        sim external-token replay 用:server 组 SonicTokenChannel 发到
        /sonic/token_input,sonic 按 token_io.mode(external_token_pre_fsq /
        external_token_post_fsq)解释。发布节拍由调用方(driver)决定。

        token: (64,) pre_fsq 连续 latent 或 post_fsq 量化码。返回 {"ok": ...}。
        """
        tok = np.asarray(token, dtype=float).reshape(-1)
        if tok.shape[0] != 64:
            return {"ok": False, "error": f"token expected (64,), got {tok.shape}"}
        try:
            resp = self.session.post(
                f"{self.base_url}/push_token",
                json={"token": tok.tolist()},
                timeout=timeout_ms / 1000.0,
            )
            if resp.status_code != 200:
                return {"ok": False,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
            body = resp.json()
            return body if isinstance(body, dict) else {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_imu(self) -> Optional[dict]:
        """Fetch latest pelvis + torso IMU samples from a3_server (/get_imu).

        Returns ``{"pelvis": {...}, "torso": {...}}`` where each dict has:
            orientation_xyzw: [4]     — quat xyzw (scipy-native)
            angular_velocity: [3]     — rad/s in IMU body frame
            linear_acceleration: [3]  — m/s² in IMU body frame
            gravity_dir: [3]          — body-frame unit vector pointing WORLD-DOWN
            timestamp: float          — server-side receive wall time
        pelvis / torso is None if that IMU hasn't published anything yet.
        Returns None if the HTTP call itself fails.
        """
        try:
            resp = self._get("/get_imu")
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def set_state_source(self, source: str) -> bool:
        """Select this session's state chain on a3_server (/set_state_source).

        Call ONCE after the pipeline kind is known (detect_embodiment_kind):
          - "whole_body" → server lazily creates the /wbc/whole_body_state
            subscription; body state (leg/waist/neck/arm + pelvis/torso IMU)
            then flows through get_whole_body_state(). Hand stays on the
            scattered chain.
          - "upper_body" → no-op on the server (wbc sub never created); state
            keeps coming from the scattered /get_joint_states + /get_imu.

        Idempotent server-side. Returns True on HTTP 200 ack. Old a3_server
        without this endpoint returns non-200 → False (caller can ignore and
        fall back to the scattered chain).
        """
        if source not in ("whole_body", "upper_body"):
            raise ValueError(f"source must be 'whole_body'|'upper_body', got {source!r}")
        try:
            resp = self._post("/set_state_source", {"source": source})
            return resp.status_code == 200
        except Exception:
            return False

    def get_whole_body_state(self) -> Optional[dict]:
        """Fetch the whole_body state chain from a3_server (/get_whole_body_state).

        Returns ``{"joints": {"leg","waist","neck","arm","hand"}, "imu":
        {"pelvis","torso"}, "timestamp"}`` where body joints + IMU come from
        /wbc/whole_body_state and hand comes from the scattered hand sub. Any
        field is None until its source has published. Returns None if the HTTP
        call fails or the server lacks the endpoint (old a3_server).
        """
        try:
            resp = self._get("/get_whole_body_state")
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def measure_rtt(self, num_samples: int = 30, warmup: int = 5,
                    timeout: float = 0.5) -> Optional[float]:
        """Trivial RTT probe (fallback for a3_server which has /measure_rtt).

        Kept as a3-side helper so callers don't need to know whether the
        A2 super class already had it. Median of round-trip millis.
        """
        try:
            for _ in range(max(0, warmup)):
                self.session.get(f"{self.base_url}/measure_rtt", timeout=timeout)
            samples = []
            for _ in range(max(1, num_samples)):
                t0 = time.monotonic()
                r = self.session.get(f"{self.base_url}/measure_rtt", timeout=timeout)
                t1 = time.monotonic()
                if r.status_code == 200:
                    samples.append(t1 - t0)
            if not samples:
                return None
            samples.sort()
            return samples[len(samples) // 2]
        except Exception:
            return None

    def get_chunk_progress(self, timeout: float = 0.2) -> Optional[dict]:
        """Return the server-reported played index into the current chunk.

        Response dict:
            {"arm": <float>, "wb": <float>, "eef": None, "hand": None}
        Where "arm" and "wb" are the SAME whole-body chunk played_idx (arm key
        kept for legacy client compatibility).
        """
        try:
            resp = self.session.get(f"{self.base_url}/get_chunk_progress",
                                    timeout=timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def get_observation(self, hand_rad: bool = False,
                        cameras: Optional[list] = None,
                        include_imu: bool = True,
                        state_source: Optional[str] = None) -> dict:
        """A3 observation with optional IMU fetch (concurrent with camera+joints).

        Extends the parent CAMERA_NAMES + joints layout by adding an "imu"
        field ``{"pelvis": {...}, "torso": {...}}`` (or None if HTTP fails).
        Set include_imu=False if the model doesn't need IMU state to skip
        the extra HTTP round-trip.

        state_source selects where joints + IMU come from:
          - None (default) → scattered chain: /get_joint_states + /get_imu
            (upper_body, and back-compat).
          - "whole_body_state" → WBC chain: joints (leg/waist/neck/arm/hand) +
            imu (pelvis/torso) come from a single /get_whole_body_state call
            (body from /wbc/whole_body_state, hand from the scattered hand sub).
            Requires a prior set_state_source("whole_body"). Cameras are still
            fetched the normal way.
        """
        cam_list = list(self.CAMERA_NAMES) if cameras is None else list(cameras)
        futures = {
            name: self._obs_executor.submit(self._get_image, f"/get_camera/{name}")
            for name in cam_list
        }
        use_wb_state = (state_source == "whole_body_state")
        if use_wb_state:
            futures["wb_state"] = self._obs_executor.submit(self.get_whole_body_state)
        else:
            futures["joints"] = self._obs_executor.submit(self.get_joint_states)
            if include_imu:
                futures["imu"] = self._obs_executor.submit(self.get_imu)

        results = {k: f.result() for k, f in futures.items()}
        if use_wb_state:
            wb = results.pop("wb_state") or {}
            joints = wb.get("joints") or {}
            imu = (wb.get("imu") if include_imu else None)
        else:
            joints = results.pop("joints")
            imu = results.pop("imu", None)

        if (hand_rad and self.hand_kind == "hand"
                and joints and joints.get("hand")):
            raw = joints["hand"].get("position")
            if raw and len(raw) == 20:
                left_rad = OMNIHAND_LEFT.actuator_to_radians(raw[:10])
                right_rad = OMNIHAND_RIGHT.actuator_to_radians(raw[10:])
                joints["hand"]["position"] = left_rad + right_rad

        out = dict(results)
        # A2 compat null fields so upstream code that expects them doesn't KeyError.
        for k in ("head_rgb", "fish_left", "fish_right",
                  "hand_left_rgb", "hand_right_rgb"):
            out.setdefault(k, None)
        out["joints"] = joints or {}
        out["imu"] = imu
        out["timestamp"] = time.time()
        return out

    def get_observation_with_progress(self, hand_rad: bool = False,
                                       cameras: Optional[list] = None,
                                       include_imu: bool = True,
                                       timeout: float = 5.0,
                                       state_source: Optional[str] = None) -> Optional[dict]:
        """A3 原子快照: 全部相机 + 关节 + IMU + whole-body chunk 进度, 一次 HTTP。

        复用父类同名方法的思路, 但适配 A3:
        - 10 路相机 (而非 5 路), names 走 CAMERA_TOPICS;
        - server 返回的 chunk_progress 是 {arm, wb, eef, hand}, wb 即 whole-body
          played_idx (arm 为兼容别名);
        - hand_rad 仅 hand_kind="hand" 时转弧度 (gripper 是 actuator counts)。

        server 不支持该 endpoint (老版本 a3_server) 时回退到旧路径:
            get_observation() + get_chunk_progress()。

        Args:
            cameras:  指定只回哪些相机 (None=全部 CAMERA_NAMES)
            include_imu: False 时跳过 IMU 字段
            timeout:  HTTP 超时 (秒); base64 10 图打包体积大, 默认 5s
            state_source: ``whole_body_state`` 时从 WBC state 链路取关节和 IMU
        """
        import base64
        cam_list = list(self.CAMERA_NAMES) if cameras is None else list(cameras)
        # /get_observation_with_progress?cameras=a,b,c&include_imu=true
        params = {
            "cameras": ",".join(cam_list),
            "include_imu": "true" if include_imu else "false",
            "state_source": state_source or "",
        }
        try:
            resp = self.session.get(
                f"{self.base_url}/get_observation_with_progress",
                params=params, timeout=timeout)
            if resp.status_code != 200:
                return self._fallback_obs_with_progress(
                    hand_rad, cameras, include_imu, state_source)
            data = resp.json()
        except Exception:
            return self._fallback_obs_with_progress(
                hand_rad, cameras, include_imu, state_source)

        # An older A3 endpoint predates the WBC state chain and silently ignores
        # state_source. Fall back instead of feeding scattered joints to a
        # whole-body checkpoint.
        if (state_source == "whole_body_state"
                and data.get("state_source") != "whole_body_state"):
            return self._fallback_obs_with_progress(
                hand_rad, cameras, include_imu, state_source)

        def _b64_to_img(b64):
            if b64 is None:
                return None
            try:
                arr = np.frombuffer(base64.b64decode(b64), np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                return None

        out = {name: _b64_to_img(data.get(name)) for name in cam_list}
        for k in ("head_rgb", "fish_left", "fish_right",
                  "hand_left_rgb", "hand_right_rgb"):
            out.setdefault(k, None)

        joints = data.get("joints") or {}
        if (hand_rad and self.hand_kind == "hand"
                and joints and joints.get("hand")):
            raw = joints["hand"].get("position")
            if raw and len(raw) == 20:
                left_rad = OMNIHAND_LEFT.actuator_to_radians(raw[:10])
                right_rad = OMNIHAND_RIGHT.actuator_to_radians(raw[10:])
                joints = dict(joints)
                joints["hand"] = dict(joints["hand"])
                joints["hand"]["position"] = left_rad + right_rad
        out["joints"] = joints
        out["imu"] = data.get("imu") if include_imu else None
        out["chunk_progress"] = data.get("chunk_progress")
        out["timestamp"] = data.get("timestamp", time.time())
        return out

    def _fallback_obs_with_progress(self, hand_rad, cameras, include_imu,
                                    state_source=None):
        """老版本 a3_server 无 /get_observation_with_progress 时的回退路径。"""
        obs = self.get_observation(hand_rad=hand_rad, cameras=cameras,
                                   include_imu=include_imu,
                                   state_source=state_source)
        if obs is None:
            return None
        prog = self.get_chunk_progress()
        obs["chunk_progress"] = prog
        return obs
