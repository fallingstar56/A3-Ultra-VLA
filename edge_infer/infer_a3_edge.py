"""
Config-driven GR00T inference for the A3 humanoid — train-time RTC edition,
REMOTE ZMQ model variant (``infer_a3_rtc_zmq.py``).

HOT-SWAP: the pipeline is chosen automatically at startup from the
checkpoint's action layout (``get_rtc_metadata`` → ``detect_embodiment_kind``):
  • whole_body — sonic_a3 (body+pelvis_quat6d) or a3_config
    (leg+waist+arm+pelvis) → A3ObsBuilder / A3ActionDecoder +
    step_chunk(mode=whole_body) server-atomic RTC (see below).
  • upper_body — examples/A3 dex/gripper configs
    (hand|gripper + arm + waist + waist_height) → A3UpperBodyObsBuilder /
    A3UpperBodyActionDecoder + CHUNK send via robot.step_chunk(mode="upper_body")
    to the THREE mc per-part topics (arm/hand as server-paced chunks,
    waist server-paced per-frame). Mirrors A2/infer_a2_rtc.py rtc_chunk:
    virtual-tick RTC + SERVER-ATOMIC swap (client sends s_used_local, server
    slices arm/hand/waist by actual_delay — see a3_server
    interp_publisher.swap_upper_chunk_atomic). Degrades to soft-continuous
    (actual_delay=0) on an a3_server without the upper_body route.
    The waist command is 4D [yaw,roll,pitch,height]; the 4th dim (pelvis
    height h) is padded with 0 (configs did not train it) unless
    --use_model_waist_height. See AsyncRTCUpperBodyRunner / build_upper_chunk_dict.

The whole-body pipeline below is unchanged:

Differs from ``infer_a3_rtc.py`` (which loads a LOCAL Gr00tPolicy) in exactly
two ways, per the wholebody-human design:

  1. MODEL LINK — A2-style remote ZMQ. Instead of loading Gr00tPolicy in
     process, this client connects to a running Gr00tPolicyServer
     (``gr00t/eval/run_gr00t_server.py``) via ``PolicyClient``. The heavy
     model runs on the server; this process only does obs build + action
     decode + chunk send. Front-end link mirrors infer_a2_rtc.py verbatim:
       PolicyClient(host, port) → get_rtc_metadata() → per-frame
       get_action(obs, options={rtc_mode,rtc_delay,action_prefix}) → feed
       info["action_pred_normalized"] back as the next prefix.

  2. END TRANSPORT — 30Hz send, server-side 50Hz interpolation. The client
     sends the policy's native 30Hz whole-body chunk to A3ServerNode, either
     through the legacy HTTP a3_server or the default subproc shared-memory
     transport
     (chunk_fps=source_fps=30, emit_mode="reference_window"). a3_server's
     50Hz timer now interpolates 30Hz→50Hz (joints linear, pelvis SLERP)
     and publishes a true 50Hz×10 TaWholeBodyReferenceWindow on
     /wbc/infer/reference_window (see robointerface master_wholebody_human,
     InterpolationPublisher.wb_snapshot_for_reference_window). The client
     NO LONGER upsamples 30→50; it works entirely on the 30Hz policy axis.

Everything else — A3ObsBuilder, A3ActionDecoder, DeltaFrameReanchor (incl.
the SO(3) pelvis reanchor keyed on action_format==ROT6D), KeyStateMachine,
server-atomic swap, record-chunks — is identical to infer_a3_rtc.py and
reused verbatim. Pelvis reanchoring is already spherical (SO(3) composition
in DeltaFrameReanchor._reanchor_rot6d), unchanged.

Pipeline (client + local engine + A3ServerNode):

  LocalEnginePolicy (VLA @ 30Hz, sonic_a3 recipe)
    → SubprocA3Robot (shared memory observation + local command pipe), or
      A3RobotInterface HTTP compatibility path
    → A3ServerNode atomic swap + 50Hz TaWholeBodyReferenceWindow (30→50 interp)
      on ``/wbc/infer/reference_window``
    → sonic (gr00t_inference on) → WBC → MC

Two modes (same as infer_a3_rtc.py):
  --mode rtc_chunk (default) — pi's chunk_train recipe. Virtual tick +
      server-atomic swap + per-timestep DeltaFrameReanchor.
  --mode standard — synchronous cold-play (one chunk at a time, no RTC).

Runtime keys (via KeyStateMachine, same as feat/a3 deploy_gr00t.py):
  s   start pushing chunks (RUNNING)
  p   pause (IDLE, sonic holds last frame)
  1..9  switch task by index into prompt.yaml
  l   list tasks
  Ctrl+C  exit

Human-in-the-loop (--human_in_loop true): the same RUNNING/IDLE state machine
is driven over HTTP instead of the keyboard, so a teleop station can hand
control back and forth with the model:
  POST|GET /start  → RUNNING — fresh cold-start (prefix 为空的第一次推理),
                     then RTC prefix from the 2nd chunk on.
  POST|GET /stop   → IDLE — cancel_chunk: model stops sending, a3_server stops
                     executing the current chunk (holds last frame) → 摇操接管.
  GET  /status     → current state.
--human_in_loop false (default) is ordinary inference (keyboard-gated).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Callable, Optional

# The verified exp70 open-loop path uses a deliberate mixed environment:
# torch/torchvision/transformers come from torch_build, while TensorRT comes
# from trt_venv.  Keep that import order here too; importing trt_venv's
# torchvision first raises `operator torchvision::nms does not exist`.
_TORCH_SITE = os.environ.get(
    "A3_TORCH_SITE",
    "/agibot/torch_build/venv/lib/python3.12/site-packages",
)


def _first_existing_path(env_name: str, candidates: tuple[str, ...]) -> str:
    """Resolve a deploy path without silently preferring a stale checkout."""
    configured = os.environ.get(env_name)
    if configured:
        return configured
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


_TRT_SITE = _first_existing_path(
    "A3_TRT_SITE",
    (
        "/agibot/edge_deploy/trt_venv/lib/python3.12/site-packages",
        "/agibot/torch_build/venv/lib/python3.12/site-packages",
    ),
)
_VIDEO_SITE = "/agibot/fengtianli/a3_lerobot/.venv/lib/python3.12/site-packages"
for _site in (_TRT_SITE, _TORCH_SITE):
    while _site in sys.path:
        sys.path.remove(_site)
sys.path.insert(0, _TORCH_SITE)

# Import these before TensorRT so GR00T resolves the same compatible stack as
# the already-verified open-loop runner.
import torch  # noqa: E402
import torchvision  # noqa: E402
import transformers  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, _TRT_SITE)
import tensorrt as trt  # noqa: E402
# The deployment venv carries OpenCV built for NumPy 2.x.  The robot system
# cv2 is built against NumPy 1.x and cannot be loaded by the Thor torch stack
# (NumPy 2.5).  Import cv2 while TRT_SITE is temporarily visible, after the
# correct torch/NumPy stack has already been selected.
import cv2  # noqa: E402
if not trt.__version__.startswith("10.13"):
    raise RuntimeError(f"wrong TensorRT loaded: {trt.__version__} from {trt.__file__}")
sys.path.remove(_TRT_SITE)
if _VIDEO_SITE not in sys.path:
    sys.path.insert(0, _VIDEO_SITE)

import yaml
from scipy.spatial.transform import Rotation, Slerp

_THIS_DIR = Path(__file__).resolve().parent
_REPO_CANDIDATES = list(_THIS_DIR.parents)
REPO = next(
    (p for p in _REPO_CANDIDATES if (p / "gr00t").is_dir()),
    _THIS_DIR.parents[min(2, len(_THIS_DIR.parents) - 1)],
)
sys.path.insert(0, str(REPO))

# A3 RoboInterface (pi fork) — has get_imu, get_chunk_progress, measure_rtt,
# whole-body step_chunk (mode=whole_body + s_used_local + emit_mode).
# 路径可用 A3_ROBOINTERFACE_DIR 覆盖;默认指向本机 RoboInterface。
_ROBOINTERFACE_DIR = _first_existing_path(
    "A3_ROBOINTERFACE_DIR",
    (
        "/agibot/edge_deploy/RoboInterface",
        "/agibot/robotinterface/RoboInterface",
        "/agibot/RoboInterface",
    ),
)
sys.path.insert(0, _ROBOINTERFACE_DIR)
from interface import A3RobotInterface  # noqa: E402

_ROS_SERVER_DIR = _first_existing_path(
    "A3_ROS_SERVER_DIR",
    (
        str(_THIS_DIR.parent / "ros_server"),
        "/agibot/ros_server",
    ),
)

# 端侧: 本地 engine policy 替代远程 ZMQ PolicyClient
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_engine_policy import LocalEnginePolicy  # noqa: E402


def _require_wholebody_robointerface_protocol() -> None:
    """Fail before robot motion when a stale RoboInterface checkout is loaded."""
    interface_py = Path(_ROBOINTERFACE_DIR) / "interface.py"
    try:
        source = interface_py.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"cannot read RoboInterface client at {interface_py}; set "
            "A3_ROBOINTERFACE_DIR to robotinterface/RoboInterface from "
            "master_wholebody_human"
        ) from exc
    required = (
        '"mode": "whole_body"',
        'payload["source_fps"]',
        'payload["emit_mode"]',
        'def set_state_source',
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(
            f"RoboInterface at {interface_py} is incompatible with whole-body RTC "
            f"(missing {missing}). Deploy robotinterface/master_wholebody_human "
            "or set A3_ROBOINTERFACE_DIR to that checkout."
        )
    print(f"[ws] RoboInterface whole-body RTC protocol: {interface_py}", flush=True)


class _NumpyImageBridge:
    """Drop-in replacement for the parts of CvBridge that A3ServerNode uses.

    ROS ships cv_bridge's boost extension compiled against NumPy 1.x; calling
    it from this process (NumPy 2.x, required by the torch_build stack)
    segfaults. The camera topics are plain uncompressed frames, so the
    conversion is a reshape plus an optional channel swap — no extension
    module needed.
    """

    _DTYPES = {"8U": np.uint8, "8S": np.int8, "16U": np.uint16,
               "16S": np.int16, "32S": np.int32, "32F": np.float32,
               "64F": np.float64}

    @classmethod
    def _decode(cls, msg) -> np.ndarray:
        encoding = (msg.encoding or "").lower()
        if encoding in ("bgr8", "rgb8"):
            dtype, channels = np.uint8, 3
        elif encoding in ("bgra8", "rgba8"):
            dtype, channels = np.uint8, 4
        elif encoding == "mono8":
            dtype, channels = np.uint8, 1
        elif encoding == "mono16":
            dtype, channels = np.uint16, 1
        else:
            depth, _, count = encoding.partition("c")
            dtype = cls._DTYPES.get(depth.upper())
            if dtype is None:
                raise ValueError(f"unsupported image encoding {msg.encoding!r}")
            channels = int(count or 1)
        buf = np.frombuffer(bytes(msg.data), dtype=dtype)
        img = buf.reshape(int(msg.height), int(msg.width), channels)
        return img[:, :, 0] if channels == 1 else img

    def imgmsg_to_cv2(self, msg, desired_encoding: str = "passthrough") -> np.ndarray:
        img = self._decode(msg)
        src = (msg.encoding or "").lower()
        want = (desired_encoding or "passthrough").lower()
        if want in ("passthrough", src) or img.ndim != 3:
            return img
        if {src, want} == {"rgb8", "bgr8"} or {src, want} == {"rgba8", "bgra8"}:
            return img[:, :, ::-1].copy() if img.shape[2] == 3 else img[:, :, [2, 1, 0, 3]].copy()
        if src == "rgba8" and want == "bgr8":
            return img[:, :, [2, 1, 0]].copy()
        if src == "bgra8" and want == "bgr8":
            return img[:, :, :3].copy()
        raise ValueError(f"cannot convert {msg.encoding!r} to {desired_encoding!r}")


class SubprocA3Robot:
    """A3 robot access with ROS running in a separate process.

    Same interface as InProcessA3Robot, but A3ServerNode lives in a child
    process (see a3_ros_worker.py) so its camera callbacks and the 150Hz
    interpolation loop hold their own GIL, not ours. Measured in-process:
    get_action 340-400ms vs 203ms standalone; the callbacks were the
    difference. Frames arrive through shared memory, so fetching an
    observation is still a memcpy — no HTTP, no JPEG.
    """

    def __init__(self, hand_kind: str, cameras: list,
                 embodiment_kind: str = "upper_body",
                 robot_ip: str = "192.168.100.100",
                 worker_python: str = "/usr/bin/python3"):
        import atexit as _atexit
        import glob as _glob
        import subprocess
        import uuid
        from multiprocessing import shared_memory

        import a3_ros_worker as W

        self._W = W
        # Sweep segments orphaned by a previous run. Ctrl-C can land before
        # close() unlinks them, and each run leaks ~18MB of /dev/shm.
        stale = [f for pat in ("a3meta_*", "a3joints_*", "a3cam*_*")
                 for f in _glob.glob(f"/dev/shm/{pat}")]
        for path in stale:
            try:
                os.unlink(path)
            except OSError:
                pass
        if stale:
            print(f"[subproc] cleaned {len(stale)} stale shm segment(s)", flush=True)
        self.hand_kind = hand_kind
        self.embodiment_kind = embodiment_kind
        self.cameras = list(cameras)
        self._last_speed_hz = 30.0
        self._send_lock = threading.Lock()
        self._replies_cv = threading.Condition()
        self._next_request_id = 1
        self._pending_requests = set()
        self._replies_by_id = {}
        self._worker_error = None
        tag = uuid.uuid4().hex[:8]

        self._meta_shm = shared_memory.SharedMemory(
            create=True, size=W.meta_size(len(self.cameras)) * 8,
            name=f"a3meta_{tag}")
        self._joints_shm = shared_memory.SharedMemory(
            create=True, size=W.joints_size() * 8, name=f"a3joints_{tag}")
        self._cam_shms = [
            shared_memory.SharedMemory(create=True, size=W.MAX_FRAME_BYTES,
                                       name=f"a3cam{i}_{tag}")
            for i in range(len(self.cameras))
        ]
        self._meta = np.ndarray((W.meta_size(len(self.cameras)),),
                                dtype=np.int64, buffer=self._meta_shm.buf)
        self._joints = np.ndarray((W.joints_size(),), dtype=np.float64,
                                  buffer=self._joints_shm.buf)
        self._cam_views = [
            np.ndarray((W.MAX_FRAME_BYTES,), dtype=np.uint8, buffer=s.buf)
            for s in self._cam_shms
        ]
        self._meta[:] = 0
        self._joints[:] = 0.0
        _state_offsets = W.state_offsets()
        self._joints[_state_offsets["progress_arm"]] = np.nan
        self._joints[_state_offsets["progress_wb"]] = np.nan
        self._meta[2] = len(self.cameras)

        cmd = [
            worker_python, "-u",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "a3_ros_worker.py"),
            "--shm-meta", self._meta_shm.name,
            "--shm-joints", self._joints_shm.name,
            "--shm-cams", ",".join(s.name for s in self._cam_shms),
            "--cameras", ",".join(self.cameras),
            "--hand-kind", hand_kind,
            "--embodiment", embodiment_kind,
            "--robot-ip", robot_ip,
            "--roboiface-dir", _ROBOINTERFACE_DIR,
        ]
        # The child runs the system python, so it must NOT see this process's
        # torch_build venv (numpy 2.x breaks ROS's compiled extensions). But
        # the ROS setup scripts the user sourced put ros2_plugin_proto and the
        # colcon workspace on PYTHONPATH — dropping those leaves
        # A3ServerNode._RosMsgWrapper None and every publisher unbuilt. So:
        # inherit, then filter out only the venv entries and append ours.
        env = dict(os.environ)
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)
        env["PYTHONNOUSERSITE"] = "1"
        inherited = [e for e in env.get("PYTHONPATH", "").split(":")
                     if e and "/torch_build/venv/" not in e
                     and "/torch_venv/" not in e
                     and "/trt_venv/" not in e]
        extra = [
            os.path.join(_ROS_SERVER_DIR, "python_deps"),
            os.path.join(_ROS_SERVER_DIR, "_pb_gen"),
            os.path.join(
                _ROS_SERVER_DIR,
                "install/a3_server/lib/python3.12/site-packages",
            ),
            os.path.join(
                _ROS_SERVER_DIR,
                "install/ros2_plugin_proto/lib/python3.12/site-packages",
            ),
            os.path.join(
                _ROS_SERVER_DIR,
                "install/joint_msgs/lib/python3.12/site-packages",
            ),
            _ROBOINTERFACE_DIR,
            "/opt/ros/jazzy/lib/python3.12/site-packages",
        ]
        seen, merged = set(), []
        # The ROS worker must resolve this deployment's generated protobufs
        # before any stale aimdk namespace inherited from the operator shell.
        for entry in extra + inherited:
            if entry not in seen:
                seen.add(entry)
                merged.append(entry)
        env["PYTHONPATH"] = ":".join(merged)
        print(f"[subproc] launching ROS worker: {worker_python}", flush=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
        _atexit.register(self.close)

        reply = W.recv_msg(self.proc.stdout)
        if not reply or reply[0] != "ready":
            raise RuntimeError(f"ROS worker failed to start: {reply!r}")
        print(f"[subproc] ROS worker ready (cameras={reply[1]})", flush=True)
        # Start routing replies only after the handshake, so the reader thread
        # cannot swallow the "ready" message.
        threading.Thread(target=self._reader_loop, name="ros-worker-reader",
                         daemon=True).start()

    # ---- command channel ----
    #
    # A wait reply can arrive after a later cancel reply, and chunk send can
    # overlap with cancel. Match every reply to its request ID; a cancel ack
    # must never be inferred from another command's successful response.
    def _reader_loop(self):
        while True:
            try:
                reply = self._W.recv_msg(self.proc.stdout)
            except Exception:                            # noqa: BLE001
                reply = None
            with self._replies_cv:
                if reply is None:
                    self._worker_error = "ROS worker exited"
                elif (not isinstance(reply, tuple) or len(reply) < 3
                      or reply[0] != "rpc" or not isinstance(reply[1], int)):
                    self._worker_error = f"ROS worker protocol mismatch: {reply!r}"
                elif reply[1] in self._pending_requests:
                    self._replies_by_id[reply[1]] = reply[2:]
                # Late replies to timed-out calls are discarded.
                self._replies_cv.notify_all()
            if self._worker_error is not None:
                break

    def _call(self, *msg, timeout: float = 60.0):
        with self._send_lock:
            with self._replies_cv:
                if self._worker_error is not None:
                    raise RuntimeError(self._worker_error)
                request_id = self._next_request_id
                self._next_request_id += 1
                self._pending_requests.add(request_id)
            try:
                self._W.send_msg(self.proc.stdin, ("rpc", request_id, *msg))
            except Exception:
                with self._replies_cv:
                    self._pending_requests.discard(request_id)
                raise
        deadline = time.monotonic() + timeout
        with self._replies_cv:
            try:
                while request_id not in self._replies_by_id:
                    if self._worker_error is not None:
                        raise RuntimeError(self._worker_error)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(f"ROS worker timed out on {msg[0]!r}")
                    self._replies_cv.wait(timeout=remaining)
                reply = self._replies_by_id.pop(request_id)
            finally:
                self._pending_requests.discard(request_id)
        if reply[0] == "error":
            raise RuntimeError(f"ROS worker: {reply[1]}")
        if reply[0] != "ok":
            raise RuntimeError(f"ROS worker invalid reply to {msg[0]!r}: {reply!r}")
        return reply

    @property
    def hand_dim(self) -> int:
        return 2 if self.hand_kind == "gripper" else 20

    def set_speed(self, hz: float) -> None:
        self._last_speed_hz = float(hz)
        self._call("set_speed", float(hz))

    def cancel_chunk(self) -> bool:
        self._call("cancel")
        return True

    # ---- observation ----
    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Seqlock read: retry while the writer is mid-update (odd counter)."""
        base = self._W.META_HEAD + self._W.META_PER_CAM * idx
        for _ in range(50):
            seq0 = int(self._meta[base])
            if seq0 == 0 or seq0 % 2:
                time.sleep(0.001)
                continue
            h, w, c = (int(self._meta[base + 1]), int(self._meta[base + 2]),
                       int(self._meta[base + 3]))
            if h <= 0 or w <= 0 or c <= 0:
                return None
            n = h * w * c
            frame = self._cam_views[idx][:n].copy()
            if int(self._meta[base]) == seq0:
                return frame.reshape(h, w, c)
        return None

    def _read_state(self) -> dict:
        for _ in range(50):
            seq0 = int(self._joints[0])
            if seq0 == 0 or seq0 % 2:
                time.sleep(0.001)
                continue
            snapshot = self._joints.copy()
            if int(self._joints[0]) == seq0:
                return self._W.unpack_state(snapshot)
        return self._W.unpack_state(self._joints.copy())

    def _read_joints(self) -> dict:
        return self._read_state()["joints"]

    def _capture_whole_body_snapshot(self) -> dict:
        if self.embodiment_kind != "whole_body":
            raise RuntimeError("whole-body snapshot requested from upper-body worker")
        self._call("snapshot", timeout=5.0)
        return self._read_state()

    def _hand_to_rad(self, joints: dict) -> dict:
        """Actuator counts -> radians for the 20D dex hand, matching what the
        checkpoint's state expects (A3RobotInterface's hand_rad=True)."""
        hand = joints.get("hand")
        if not hand or self.hand_kind != "hand":
            return joints
        pos = hand.get("position") or []
        if len(pos) != 20:
            return joints
        from config import OMNIHAND_LEFT, OMNIHAND_RIGHT
        joints = dict(joints)
        joints["hand"] = {"position": (
            list(OMNIHAND_LEFT.actuator_to_radians(list(pos[:10])))
            + list(OMNIHAND_RIGHT.actuator_to_radians(list(pos[10:]))))}
        return joints

    def get_observation(self, hand_rad: bool = False,
                        cameras: Optional[list] = None, **_kwargs) -> dict:
        if self.embodiment_kind == "whole_body":
            state = self._capture_whole_body_snapshot()
            joints = state["joints"]
        else:
            state = None
            joints = self._read_joints()
        if hand_rad:
            joints = self._hand_to_rad(joints)
        out = {
            "joints": joints,
            "imu": (state["imu"] if state is not None
                    else {"pelvis": None, "torso": None}),
        }
        for i, name in enumerate(self.cameras):
            if cameras is None or name in cameras:
                out[name] = self._read_frame(i)
        if state is not None:
            out["timestamp"] = state["timestamp"]
            out["snapshot_atomic"] = True
        return out

    def get_observation_with_progress(
        self,
        hand_rad: bool = False,
        cameras: Optional[list] = None,
        include_imu: bool = True,
        state_source: Optional[str] = None,
        **_kwargs,
    ) -> dict:
        if self.embodiment_kind != "whole_body":
            out = self.get_observation(hand_rad=hand_rad, cameras=cameras)
            out["chunk_progress"] = self.get_chunk_progress()
            out["snapshot_atomic"] = False
            return out
        if state_source not in (None, "whole_body_state"):
            raise ValueError(
                "whole-body subproc requires state_source=whole_body_state"
            )
        state = self._capture_whole_body_snapshot()
        joints = state["joints"]
        if hand_rad:
            joints = self._hand_to_rad(joints)
        out = {
            "joints": joints,
            "imu": state["imu"] if include_imu else None,
            "chunk_progress": state["chunk_progress"],
            "timestamp": state["timestamp"],
            "snapshot_atomic": True,
        }
        wanted = self.cameras if cameras is None else cameras
        for name in wanted:
            out[name] = (
                self._read_frame(self.cameras.index(name))
                if name in self.cameras else None
            )
        return out

    def get_joint_states(self, hand_rad: bool = False) -> dict:
        joints = self._read_joints()
        return self._hand_to_rad(joints) if hand_rad else joints

    # ---- action ----
    def step_chunk(self, chunk: dict, chunk_fps: float = 30.0, wait: bool = True,
                   settle_ms: float = 0.0, timeout_ms: float = 30000.0,
                   s_used_local=None, **kwargs) -> dict:
        payload = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in chunk.items()}
        if self.embodiment_kind == "whole_body":
            options = {
                "s_used_local": s_used_local,
                "s_used_local_wire": kwargs.get("s_used_local_wire"),
                "source_fps": kwargs.get("source_fps", chunk_fps),
                "chunk_id": kwargs.get("chunk_id", -1),
                "adaptive_transition": kwargs.get("adaptive_transition", False),
                "emit_mode": kwargs.get("emit_mode", "reference_window"),
                "ta_cmd_hz": kwargs.get("ta_cmd_hz"),
            }
            reply = self._call(
                "wb_chunk", payload, float(chunk_fps), options,
                timeout=(float(timeout_ms) + float(settle_ms)) / 1000.0 + 5.0,
            )
            resp = dict(reply[1] or {})
            resp.update({"ok": True, "transport": "subproc"})
            if wait:
                self.wait_for_done(settle_ms=settle_ms, timeout_ms=timeout_ms)
            return resp
        if s_used_local is not None:
            # RTC: the child does the server-atomic swap and reports how many
            # frames elapsed during inference.
            reply = self._call("swap", payload, float(chunk_fps), int(s_used_local))
            return {"ok": True, "transport": "subproc",
                    "actual_delay": int(reply[1])}
        self._call("chunk", payload, float(chunk_fps))
        if wait:
            self.wait_for_done(settle_ms=settle_ms, timeout_ms=timeout_ms)
        return {"ok": True, "transport": "subproc", "actual_delay": 0}

    def get_chunk_progress(self, timeout: float = 0.2) -> Optional[dict]:
        """Live played index, mirrored into shm by the child at 200Hz.

        RTC calls this every tick; reading shm keeps it off the command
        channel so it never queues behind a chunk install.
        """
        raw = int(self._meta[self._W.META_PLAYED_IDX])
        if raw < 0:
            return None
        played = raw / float(self._W.PLAYED_SCALE)
        if self.embodiment_kind == "whole_body":
            return {"arm": played, "wb": played, "eef": None, "hand": None}
        return {"arm": played}

    def measure_rtt(self, num_samples: int = 30, warmup: int = 5) -> float:
        """Command-channel round trip. Orders of magnitude below HTTP, but the
        RTC runner logs it, so report the real number."""
        command = ("ping",) if self.embodiment_kind == "whole_body" else (
            "set_speed", float(self._last_speed_hz)
        )
        for _ in range(max(0, warmup)):
            self._call(*command)
        t0 = time.time()
        n = max(1, num_samples)
        for _ in range(n):
            self._call(*command)
        return (time.time() - t0) / n

    def wait_for_done(self, settle_ms: float = 0.0,
                      timeout_ms: float = 30000.0) -> None:
        """Block until the chunk has finished playing.

        The child does the waiting on interp_pub itself — the same place
        a3_server does it for /send_chunk?wait=true in the reference client.
        Polling the shm mirror here instead read a 5ms-stale zero and returned
        immediately, so the next chunk landed on one still playing.
        """
        self._call("wait", float(settle_ms), float(timeout_ms),
                   timeout=(float(timeout_ms) + float(settle_ms)) / 1000.0 + 5.0)

    def ready(self, cameras: list, require_hand: bool = True,
              require_imu: bool = False) -> tuple:
        if self.embodiment_kind == "whole_body":
            try:
                state = self._capture_whole_body_snapshot()
            except Exception as exc:                   # noqa: BLE001
                return False, f"whole-body snapshot: {exc}"
        else:
            state = None
        missing = [c for c in cameras
                   if c in self.cameras
                   and self._read_frame(self.cameras.index(c)) is None]
        missing += [c for c in cameras if c not in self.cameras]
        joints = state["joints"] if state is not None else self._read_joints()
        if self.embodiment_kind == "whole_body":
            for name, dim in (("leg", 12), ("waist", 3), ("arm", 14)):
                pos = ((joints.get(name) or {}).get("position") or [])
                if len(pos) < dim:
                    missing.append(f"{name} joints ({len(pos)}/{dim})")
            if require_imu and not (state["imu"] or {}).get("pelvis"):
                missing.append("pelvis IMU")
        elif not (joints.get("arm") and joints.get("waist")):
            missing.append("arm/waist joints")
        if require_hand and not joints.get("hand"):
            missing.append("hand joints")
        return (not missing, ", ".join(missing))

    def keep_only_cameras(self, wanted: list) -> None:
        return   # the worker already subscribes to just these

    def close(self) -> None:
        if getattr(self, "proc", None) is None:
            return
        try:
            if self.proc.poll() is None:
                self._call("shutdown", timeout=5.0)
                self.proc.wait(timeout=5)
        except Exception:                                # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                            # noqa: BLE001
                pass
        self.proc = None
        for shm in (*self._cam_shms, self._joints_shm, self._meta_shm):
            try:
                shm.close()
                shm.unlink()
            except Exception:                            # noqa: BLE001
                pass


class InProcessA3Robot:
    """Direct in-process adapter for the Thor A3ServerNode.

    Unlike ``A3RobotInterface``, this never serializes observations or sends
    HTTP requests. ROS callbacks update the node's in-memory caches and the
    interpolation publisher owns the 150 Hz command publication.
    """

    def __init__(self, hand_kind: str, robot_ip: str = "192.168.100.100"):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from a3_server.joint_config import HandKind
        from a3_server.server_node import A3ServerNode

        self._rclpy = rclpy
        if not rclpy.ok():
            # Keep rclpy's SIGINT handler out of the way: it tears the context
            # down immediately, so the reset-end chunk would fail to publish.
            # Ctrl-C then surfaces as a plain KeyboardInterrupt and main()'s
            # finally-block can still park the arm safely.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self._owns_context = True
        else:
            self._owns_context = False
        kind = HandKind.HAND if hand_kind == "hand" else HandKind.GRIPPER
        self.node = A3ServerNode(robot_ip=robot_ip, hand_kind=kind)
        # Swap in the NumPy-2-safe decoder before spinning; the camera
        # callbacks would otherwise segfault inside cv_bridge's extension.
        self.node.cv_bridge = _NumpyImageBridge()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, name="a3-ros-spin", daemon=True
        )
        self._spin_thread.start()
        self.hand_kind = hand_kind
        print(f"[inproc] A3ServerNode ready (robot_ip={robot_ip}, hand_kind={hand_kind})", flush=True)

    def keep_only_cameras(self, wanted: list) -> None:
        """Drop camera subscriptions the checkpoint never reads.

        A3ServerNode subscribes to all 10 topics. Their callbacks decode every
        frame in Python (10 x 30Hz = 300 decodes/s) and hold the GIL, which is
        what inflates get_action from ~203ms to ~1050ms in this process. The
        policy only needs a few of them, so destroy the rest.
        """
        from a3_server.server_node import CAMERA_TOPICS

        keep_topics = {CAMERA_TOPICS[c] for c in wanted if c in CAMERA_TOPICS}
        cam_topics = {t.lstrip("/") for t in CAMERA_TOPICS.values()}
        keep_norm = {t.lstrip("/") for t in keep_topics}
        # Resolve every name BEFORE destroying anything: reading .topic_name on
        # a destroyed handle raises InvalidHandle.
        subs = [(sub, sub.topic_name.lstrip("/")) for sub in self.node.subscriptions]
        dropped = []
        for sub, topic in subs:
            if not any(topic.endswith(t) for t in cam_topics):
                continue   # not a camera subscription — leave joints/IMU alone
            if any(topic.endswith(t) for t in keep_norm):
                continue
            self.node.destroy_subscription(sub)
            dropped.append(topic)
        # Callbacks may still hold a stale frame for a dropped topic.
        with self.node._cam_lock:
            for name, topic in CAMERA_TOPICS.items():
                if name not in wanted:
                    self.node.latest_cameras[name] = None
        print(f"[inproc] cameras kept={list(wanted)} dropped={len(dropped)} "
              f"(unsubscribed to keep their callbacks off the GIL)", flush=True)

    def set_speed(self, hz: float) -> None:
        self.node.interp_pub.set_send_fps(float(hz))

    @property
    def hand_dim(self) -> int:
        return 2 if self.hand_kind == "gripper" else 20

    def _hand_rad_to_actuator(self, rows: np.ndarray) -> np.ndarray:
        """20D radians -> raw actuator counts, per 10D half.

        The mc hand topic takes actuator units; A3RobotInterface does this via
        ``hand_value="rad"``. Skipping it feeds ~1.5 (radians) where ~4096
        (fully open) is expected, i.e. a permanently clenched fist.
        """
        from config import OMNIHAND_LEFT, OMNIHAND_RIGHT
        out = np.empty_like(rows, dtype=np.float64)
        for i, row in enumerate(rows):
            left = OMNIHAND_LEFT.radians_to_actuator(list(row[:10]))
            right = OMNIHAND_RIGHT.radians_to_actuator(list(row[10:]))
            out[i] = np.asarray(list(left) + list(right), dtype=np.float64)
        return out

    def _hand_actuator_to_rad(self, position: list) -> list:
        from config import OMNIHAND_LEFT, OMNIHAND_RIGHT
        left = OMNIHAND_LEFT.actuator_to_radians(list(position[:10]))
        right = OMNIHAND_RIGHT.actuator_to_radians(list(position[10:]))
        return list(left) + list(right)

    def _hand_joints(self, hand_rad: bool) -> Optional[dict]:
        """Cached hand joints, converted to radians when the checkpoint's state
        is in radians (dex hand). Gripper counts stay raw."""
        hand = self.node.latest_hand_joints
        if not (hand_rad and self.hand_kind == "hand" and hand):
            return hand
        pos = hand.get("position")
        if not pos or len(pos) != 20:
            return hand
        converted = dict(hand)
        converted["position"] = self._hand_actuator_to_rad(pos)
        return converted

    def get_joint_states(self, hand_rad: bool = False) -> dict:
        """Same shape as the HTTP /get_joint_states payload, read from cache."""
        return {
            "arm": self.node.latest_arm_joints,
            "hand": self._hand_joints(hand_rad),
            "waist": self.node.latest_waist_joints,
            "leg": self.node.latest_leg_joints,
            "neck": self.node.latest_neck_joints,
        }

    def _pick_reset_hand_pose(self, hand_pose: str, hand_open):
        """Mirrors A2/A3RobotInterface._pick_reset_hand_pose so reset_upper_body_arm
        can park the end effector on the way out. Values are raw actuator units."""
        from config import (  # RoboInterface/config.py
            VLA_GRIPPER_CLOSE_POS, VLA_GRIPPER_OPEN_POS, VLA_HAND_FIST_POS,
        )
        if self.hand_kind == "gripper":
            return ((VLA_GRIPPER_CLOSE_POS, "夹爪闭合") if hand_pose == "fist"
                    else (VLA_GRIPPER_OPEN_POS, "夹爪张开"))
        if hand_pose == "fist":
            return VLA_HAND_FIST_POS, "握拳"
        return hand_open, "张开手"

    def wait_for_done(self, settle_ms: float = 0.0, timeout_ms: float = 30000.0) -> None:        self.node.interp_pub.wait_chunk_done(
            settle_sec=max(0.0, float(settle_ms)) / 1000.0,
            timeout_sec=max(0.1, float(timeout_ms)) / 1000.0,
        )

    def get_observation(self, hand_rad: bool = False,
                        cameras: Optional[list] = None, **_kwargs) -> dict:
        """Snapshot the node's in-memory caches — no serialization, no HTTP.

        ``cameras`` limits the copy to what the checkpoint reads (obs_builder
        passes it via fetch_kwargs). Copying all 10 topics would move ~27MB of
        720p frames per step, most of it never used, while holding the lock the
        camera callbacks need to write.
        """
        wanted = self.node.latest_cameras.keys() if cameras is None else cameras
        with self.node._cam_lock:
            frames = {
                key: (None if self.node.latest_cameras.get(key) is None
                      else np.asarray(self.node.latest_cameras[key]).copy())
                for key in wanted
            }
        with getattr(self.node, "_imu_lock", threading.Lock()):
            pelvis_imu = getattr(self.node, "latest_pelvis_imu", None)
            torso_imu = getattr(self.node, "latest_torso_imu", None)
        joints = {
            "arm": self.node.latest_arm_joints,
            "hand": self._hand_joints(hand_rad),
            "waist": self.node.latest_waist_joints,
            "leg": self.node.latest_leg_joints,
            "neck": self.node.latest_neck_joints,
        }
        return {
            "joints": joints,
            "imu": {"pelvis": pelvis_imu, "torso": torso_imu},
            **frames,
        }

    def step_chunk(self, chunk: dict, chunk_fps: float = 30.0, wait: bool = True,
                   settle_ms: float = 0.0, timeout_ms: float = 30000.0,
                   **_kwargs) -> dict:
        """Install arm/hand/waist chunks straight into the 150 Hz interpolator.

        Accepts the same payload build_upper_chunk_dict / reset_upper_body_arm
        produce, including a 1-D waist target (promoted to a single-row chunk)
        and ``hand_value="rad"`` (converted to actuator units here, exactly as
        A3RobotInterface.step_chunk does).
        """
        def _rows(value, width: int) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float64)
            return arr.reshape(1, width) if arr.ndim == 1 else arr

        arm = chunk.get("arm")
        hand = chunk.get("hand")
        waist = chunk.get("waist")
        if arm is not None:
            self.node.interp_pub.set_arm_chunk(_rows(arm, 14), chunk_fps=float(chunk_fps))
        if hand is not None:
            hand_rows = _rows(hand, self.hand_dim)
            if chunk.get("hand_value") == "rad" and self.hand_kind == "hand":
                hand_rows = self._hand_rad_to_actuator(hand_rows)
            self.node.interp_pub.set_hand_chunk(hand_rows, chunk_fps=float(chunk_fps))
        if waist is not None:
            self.node.interp_pub.set_waist_chunk(_rows(waist, 4), chunk_fps=float(chunk_fps))
        if wait:
            self.node.interp_pub.wait_chunk_done(
                settle_sec=max(0.0, float(settle_ms)) / 1000.0,
                timeout_sec=max(0.1, float(timeout_ms)) / 1000.0,
            )
        return {"ok": True, "transport": "inproc"}

    def cancel_chunk(self) -> None:
        self.node.interp_pub.cancel_chunk()

    def close(self) -> None:
        try:
            self.cancel_chunk()
        finally:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.executor.shutdown(timeout_sec=1.0)
            self._spin_thread.join(timeout=2.0)
            if self._owns_context and self._rclpy.ok():
                self._rclpy.shutdown()

    def ready(self, cameras: list[str], require_hand: bool = True,
              require_imu: bool = False) -> tuple[bool, str]:
        missing = [name for name in cameras if self.node.latest_cameras.get(name) is None]
        if self.node.latest_arm_joints is None or self.node.latest_waist_joints is None:
            missing.append("arm/waist joints")
        if require_hand and self.node.latest_hand_joints is None:
            missing.append("hand joints")
        if require_imu and self.node.latest_pelvis_imu is None:
            missing.append("pelvis IMU")
        return (not missing, ", ".join(missing))


# ============================================================================
# Sonic modality key → A3 camera name
#
# Client-side hardware mapping — abstract sonic key (as it appears in the
# training modality.json) → A3RobotInterface.CAMERA_NAMES. If a checkpoint
# lists a video key not present here, we fail fast with a clear error. The
# head-front target is CLI-overridable via --head-cam-source since sonic_a3
# checkpoints differ on which stereo eye they were trained on.
# ============================================================================

VIDEO_CAMERA_MAP_DEFAULT: dict[str, str] = {
    "head_front":        "head_stereo_left",
    # task_16903 records the physical camera names directly.
    "head_stereo_left":  "head_stereo_left",
    "head_stereo_right": "head_stereo_right",
    "head_front_right":  "head_stereo_right",
    "head_side_left":    "head_left",
    "head_side_right":   "head_right",
    "head_rear":         "head_rear",
    "chest_front":       "chest_front",
    "waist_front":       "waist_front",
    "wrist_left":        "wrist_left",
    "wrist_right":       "wrist_right",
    "armpit_right":      "armpit_right",
}

# BODY_31 → BODY_29 index map (drop neck idx 15/16). Matches modality.json's
# "indices" for body_pos / body_vel / body in every sonic_a3 config.
BODY_INDICES_29: list[int] = list(range(0, 15)) + list(range(17, 31))

# Checkpoint image_target_size / shortest_image_edge (server transform reruns
# a smallest-max-size at 256, so feeding native 256 avoids a resample).
IMG_SIZE: int = 256


# ============================================================================
# Rotation helpers
# ============================================================================


def _quat_xyzw_to_rot6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """xyzw quat → 6D rot (first two rows of R flattened). scipy-native."""
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-8:
        return np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    s = n ** 0.5
    R = Rotation.from_quat([qx / s, qy / s, qz / s, qw / s]).as_matrix()
    return R[:2, :].reshape(6).astype(np.float32)


def _rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(H, 6) → (H, 4) wxyz via Gram-Schmidt. Canonical w >= 0 so adjacent
    smooth rotations don't flip sign because scipy picked a different hemisphere.
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(-1, 2, 3)
    out = np.zeros((r.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(r):
        row1, row2 = m[0], m[1]
        b1 = row1 / max(np.linalg.norm(row1), 1e-8)
        proj = float(np.dot(b1, row2))
        b2 = row2 - proj * b1
        b2 = b2 / max(np.linalg.norm(b2), 1e-8)
        b3 = np.cross(b1, b2)
        R = np.stack([b1, b2, b3], axis=0)
        q_xyzw = Rotation.from_matrix(R).as_quat().astype(np.float32)
        if q_xyzw[3] < 0:
            q_xyzw = -q_xyzw
        out[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return out


def _preprocess_image(img: Optional[np.ndarray], size: int = IMG_SIZE) -> np.ndarray:
    """BGR (or None) → RGB uint8 (H, W, 3). **不做方形 resize**。

    ckpt 的 eval_image_transform 首步是 LetterBoxPad(p=1.0)——按原图长宽比补边
    成方图后再 SmallestMaxSize/crop。这里若先 cv2.resize(img,(256,256)) 把非方图
    强行拉伸成方,LetterBoxPad 对已是方图的输入等于失效,模型看到的是被压扁的画面
    (几何失真)→ 推理动作异常。故必须喂**原生长宽比**,由服务端 transform 做
    letterbox+resize+crop(与 feat/a3 send_dataset_obs/deploy_gr00t 的原生喂法一致)。
    """
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================================
# Robot joints → BODY_31 layout
# ============================================================================


def _joint_slice(joints: dict, name: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    j = joints.get(name) or {}
    p = np.asarray(j.get("position") or [0.0] * dim, dtype=np.float64)
    v = np.asarray(j.get("velocity") or [0.0] * dim, dtype=np.float64)
    if p.size < dim:
        p = np.concatenate([p, np.zeros(dim - p.size)])
    if v.size < dim:
        v = np.concatenate([v, np.zeros(dim - v.size)])
    return p[:dim], v[:dim]


def _assemble_body31(joints: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concat robot joints into BODY_31 layout (leg12+waist3+neck2+arm14).
    Returns (q31, dq31, leg, waist, arm) — the last three are the raw slices,
    useful for the old ``leg/waist/arm`` state keys.
    """
    leg_p, leg_v = _joint_slice(joints, "leg", 12)
    waist_p, waist_v = _joint_slice(joints, "waist", 3)
    neck_p, neck_v = _joint_slice(joints, "neck", 2)
    arm_p, arm_v = _joint_slice(joints, "arm", 14)
    q31 = np.zeros(31, dtype=np.float64)
    q31[0:12] = leg_p
    q31[12:15] = waist_p
    q31[15:17] = neck_p
    q31[17:31] = arm_p
    dq31 = np.zeros(31, dtype=np.float64)
    dq31[0:12] = leg_v
    dq31[12:15] = waist_v
    dq31[15:17] = neck_v
    dq31[17:31] = arm_v
    return q31, dq31, leg_p.astype(np.float32), waist_p.astype(np.float32), arm_p.astype(np.float32)


# ============================================================================
# Config-driven obs builder
# ============================================================================
#
# Each supported state key maps to a builder that reads from
# (q31, dq31, leg, waist, arm, imu) → (D,) np.float32. Adding a new state key
# = adding a new entry here. video keys route through VIDEO_CAMERA_MAP.


def _sk_leg(q31, dq31, leg, waist, arm, imu, hand):            return leg
def _sk_waist(q31, dq31, leg, waist, arm, imu, hand):          return waist
def _sk_arm(q31, dq31, leg, waist, arm, imu, hand):            return arm
def _sk_leg_vel(q31, dq31, leg, waist, arm, imu, hand):        return dq31[0:12].astype(np.float32)
def _sk_waist_vel(q31, dq31, leg, waist, arm, imu, hand):      return dq31[12:15].astype(np.float32)
def _sk_arm_vel(q31, dq31, leg, waist, arm, imu, hand):        return dq31[17:31].astype(np.float32)
def _sk_hand(q31, dq31, leg, waist, arm, imu, hand):
    pos = np.asarray((hand or {}).get("position") or [], dtype=np.float32).reshape(-1)
    if pos.size != 20:
        raise RuntimeError(f"checkpoint requires 20D hand state, got {pos.size}D")
    return pos
def _sk_gripper(q31, dq31, leg, waist, arm, imu, hand):
    pos = np.asarray((hand or {}).get("position") or [], dtype=np.float32).reshape(-1)
    if pos.size != 2:
        raise RuntimeError(f"checkpoint requires 2D gripper state, got {pos.size}D")
    return pos
def _sk_hand_opening(q31, dq31, leg, waist, arm, imu, hand):
    """task_16903 uses the index_bent_1 value (index 4) of each 10D hand."""
    pos = np.asarray((hand or {}).get("position") or [], dtype=np.float32).reshape(-1)
    if pos.size < 20:
        raise RuntimeError(
            f"checkpoint requires 20D hand state for hand_opening, got {pos.size}D"
        )
    return pos[[4, 14]]
def _sk_body_pos(q31, dq31, leg, waist, arm, imu, hand):
    return q31[BODY_INDICES_29].astype(np.float32)
def _sk_body_vel(q31, dq31, leg, waist, arm, imu, hand):
    return dq31[BODY_INDICES_29].astype(np.float32)
def _sk_pelvis_gravity(q31, dq31, leg, waist, arm, imu, hand):
    pelvis = (imu.get("pelvis") if isinstance(imu, dict) else None) or {}
    g = pelvis.get("gravity_dir") or [0.0, 0.0, -1.0]
    return np.asarray(g[:3], dtype=np.float32)
def _sk_pelvis_orient6d(q31, dq31, leg, waist, arm, imu, hand):
    pelvis = (imu.get("pelvis") if isinstance(imu, dict) else None) or {}
    q = pelvis.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0]
    return _quat_xyzw_to_rot6d(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


STATE_BUILDERS: dict[str, Callable] = {
    "leg":              _sk_leg,
    "leg_pos":          _sk_leg,
    "leg_vel":          _sk_leg_vel,
    "waist":            _sk_waist,
    "waist_pos":        _sk_waist,
    "waist_vel":        _sk_waist_vel,
    "arm":              _sk_arm,
    "arm_pos":          _sk_arm,
    "arm_vel":          _sk_arm_vel,
    "hand":             _sk_hand,
    "hand_opening":     _sk_hand_opening,
    "gripper":          _sk_gripper,
    "gripper_opening":  _sk_gripper,
    "body_pos":         _sk_body_pos,
    "body_vel":         _sk_body_vel,
    "pelvis_gravity":   _sk_pelvis_gravity,
    "pelvis_orient6d":  _sk_pelvis_orient6d,
}


class A3ObsBuilder:
    """Turn a robot_obs dict into the GR00T obs the local policy expects,
    driven by the checkpoint's modality schema (fetched at startup via
    ``Gr00tPolicy.get_rtc_metadata``).

    Only fields the checkpoint actually consumes are read from robot_obs —
    cameras / IMU are skipped when the model doesn't need them.
    """

    def __init__(
        self,
        metadata: dict,
        video_camera_map: dict[str, str],
        state_source: Optional[str] = "whole_body_state",
        hand_kind: str = "hand",
    ):
        self.video_camera_map = dict(video_camera_map)
        self.state_source = state_source
        self.hand_kind = hand_kind
        self.video_keys = list(metadata.get("video_keys") or [])
        self.state_keys = list(metadata["state_keys"])
        self.ref_only_keys = list(metadata.get("reference_only_keys") or [])
        self.language_keys = list(
            metadata.get("language_keys") or ["annotation.human.task_description"]
        )

        # State keys the loader materialises (encoder input + reference-only),
        # preserving the config's own order for repeatability.
        seen: set[str] = set()
        self._all_state: list[str] = []
        for k in self.state_keys + self.ref_only_keys:
            if k not in seen:
                seen.add(k)
                self._all_state.append(k)

        for k in self.video_keys:
            if k not in self.video_camera_map:
                raise RuntimeError(
                    f"video key {k!r} not in VIDEO_CAMERA_MAP; add it via "
                    "--head-cam-source or extend VIDEO_CAMERA_MAP_DEFAULT."
                )
        for k in self._all_state:
            if k not in STATE_BUILDERS:
                raise RuntimeError(
                    f"state key {k!r} has no client-side builder. Add it to "
                    "STATE_BUILDERS."
                )

        self.cameras_needed = [self.video_camera_map[k] for k in self.video_keys]
        self.needs_imu = any(
            k in ("pelvis_gravity", "pelvis_orient6d") for k in self._all_state
        )
        self.needs_hand = bool(
            {"hand", "hand_opening", "gripper", "gripper_opening"}
            & set(self._all_state)
        )
        # Video/state/language expected dims (for one-line startup log).
        self.state_dims = {k: int(metadata["state_dims"].get(k, -1)) for k in self.state_keys}

    def fetch_kwargs(self) -> dict:
        """Kwargs to pass to A3RobotInterface.get_observation / _with_progress
        so we only pull the cameras + IMU the checkpoint actually needs.

        state_source="whole_body_state" routes joints + IMU through the WBC
        chain (/get_whole_body_state ← /wbc/whole_body_state); None uses the
        scattered /get_joint_states + /get_imu chain."""
        return {
            # a3_server's hand_joint_state contract is actuator units on both
            # robot and sim. Convert to the radians used by the training data.
            "hand_rad": self.needs_hand and self.hand_kind == "hand",
            "cameras": list(self.cameras_needed),
            "include_imu": self.needs_imu,
            "state_source": self.state_source,
        }

    def build(self, robot_obs: dict, task: str) -> tuple[dict, np.ndarray]:
        """robot_obs → (gr00t_obs, current_body31). current_body31 is used by
        the decoder to restore the neck (idx 15/16) when the checkpoint drops
        it from the action layout."""
        joints = robot_obs.get("joints") or {}
        imu = robot_obs.get("imu") or {}
        q31, dq31, leg, waist, arm = _assemble_body31(joints)
        hand = joints.get("hand") or {}

        video = {}
        for k in self.video_keys:
            cam_name = self.video_camera_map[k]
            frame = _preprocess_image(robot_obs.get(cam_name))
            video[k] = frame[None, None, ...]  # (1, T=1, H, W, C)

        state = {}
        for k in self._all_state:
            arr = STATE_BUILDERS[k](q31, dq31, leg, waist, arm, imu, hand)
            state[k] = np.asarray(arr, dtype=np.float32).reshape(1, 1, -1)

        language = {self.language_keys[0]: [[task]]} if self.language_keys else {}
        obs = {"video": video, "state": state, "language": language}
        return obs, q31

    def hand_position_rad(self, robot_obs: dict) -> np.ndarray | None:
        """Return the current hand/gripper state used by hand action decoding."""
        if not self.needs_hand:
            return None
        hand = ((robot_obs.get("joints") or {}).get("hand") or {}).get("position") or []
        hand = np.asarray(hand, dtype=np.float32).reshape(-1)
        expected_dim = 20 if self.hand_kind == "hand" else 2
        if hand.size != expected_dim:
            raise RuntimeError(
                f"checkpoint requires {expected_dim}D {self.hand_kind} state, "
                f"got {hand.size}D"
            )
        return hand


# ============================================================================
# hand_opening → 5-finger spread
# ============================================================================
#
# task_16903 only emits index_bent_1 (one scalar per hand). The checkpoint was
# trained with middle/ring/pinky held closed (~1.48 rad, see config.py's
# VLA_HAND_INIT_POS comment), so by default only the index finger moves.
#
# HAND_SPREAD_FROM_INDEX maps that single scalar onto all five fingers: the
# index value is normalised against its own limit into an opening ratio
# t in [0, 1], then t interpolates each target joint across *its own* limits.
# Per-joint limits matter — thumb_bent_2 spans 0.8416 rad where the four finger
# bends span 1.48, and the left hand's limits are sign-mirrored
# (left_pos_direction[2] = -1). Copying the index radian value verbatim would
# drive the thumb past its limit and the left thumb the wrong way.
#
# Limits below mirror OmnihandCtrl.active_joint_{min,max} in
# RoboInterface/config.py; keep them in sync if that file changes.

# per-hand 10D joint index → (lo, hi) active-joint radian limits, right hand
_HAND_LIMITS_R: dict[int, tuple[float, float]] = {
    2: (0.0, 0.8416),   # thumb_bent_2
    4: (0.0, 1.48),     # index_bent_1  (the driving joint)
    5: (0.0, 1.48),     # middle_bent
    7: (0.0, 1.48),     # ring_bent_1
    9: (0.0, 1.48),     # pinky_bent_1
}
# left hand mirrors any joint whose left_pos_direction is -1 (idx 2 among these)
_HAND_MIRROR_LEFT = {2}
_HAND_DRIVER_IDX = 4
# joints written from the driver ratio, in per-hand 10D index space
_HAND_SPREAD_TARGETS = (2, 5, 7, 9)


def _hand_limits(joint_idx: int, is_left: bool) -> tuple[float, float]:
    lo, hi = _HAND_LIMITS_R[joint_idx]
    if is_left and joint_idx in _HAND_MIRROR_LEFT:
        return -hi, -lo
    return lo, hi


def _spread_index_to_five_fingers(hand20: np.ndarray) -> np.ndarray:
    """In-place: drive thumb/middle/ring/pinky bends from each hand's index bend.

    ``hand20`` is (H, 20) active-joint radians, left hand 0:10, right 10:20.
    """
    for hand_slot, is_left in ((0, True), (10, False)):
        d_lo, d_hi = _hand_limits(_HAND_DRIVER_IDX, is_left)
        driver = hand20[:, hand_slot + _HAND_DRIVER_IDX]
        span = d_hi - d_lo
        if abs(span) < 1e-9:
            continue
        # opening ratio: 0 = straight/open end of the driver, 1 = fully bent
        t = np.clip((driver - d_lo) / span, 0.0, 1.0)
        for j in _HAND_SPREAD_TARGETS:
            lo, hi = _hand_limits(j, is_left)
            hand20[:, hand_slot + j] = lo + t * (hi - lo)
    return hand20


# ============================================================================
# Config-driven action decoder
# ============================================================================


class A3ActionDecoder:
    """Server per-key action chunks → the 33D whole-body chunk dict expected
    by A3RobotInterface.step_chunk(mode=whole_body):

        {"leg": (H, 12), "waist": (H, 3), "arm": (H, 14),
         "pelvis_quat_wxyz": (H, 4)}

    Two supported layouts (auto-detected from metadata.action_keys):

    - **sonic_a3** — ``body(29)`` + ``pelvis_quat6d(6)`` = 35D. body's 29D is
      reinserted into BODY_31 (neck idx 15/16 held from the robot's current
      neck state), then split into leg/waist/arm. pelvis rot6d → wxyz quat.

    - **a3_config (33D)** — ``leg(12)`` + ``waist(3)`` + ``arm(14)`` +
      (``pelvis_quat(4)`` or ``pelvis_quat6d(6)``). The current A3 datasets
      use the equivalent ``leg_pos`` / ``waist_pos`` / ``arm_pos`` names and
      may additionally include ``hand_opening``. Direct passthrough (rot6d
      converted to quat if needed).
    """

    def __init__(self, metadata: dict, hand_spread: bool = True):
        self.action_keys = list(metadata["action_keys"])
        keys = set(self.action_keys)
        self.hand_action_key = next(
            (key for key in ("hand", "hand_opening", "gripper", "gripper_opening")
             if key in keys),
            None,
        )
        self.has_hand_opening = self.hand_action_key == "hand_opening"
        self.hand_value = (
            "raw" if self.hand_action_key in ("gripper", "gripper_opening") else "rad"
        )
        self.hand_mapping = None
        if self.hand_action_key == "hand_opening":
            try:
                from gr00t.eval.real_robot.hand_mapping import HandOpeningMapping
                self.hand_mapping = HandOpeningMapping.from_metadata(
                    metadata.get("hand_mapping")
                    or metadata.get("hand_mapping_sample")
                )
            except (ImportError, ValueError) as exc:
                print(
                    f"[decoder] WARNING: checkpoint hand mapping unavailable ({exc}); "
                    "falling back to the live hand pose mapping",
                    flush=True,
                )
        # drive all five fingers from the single index-bend action
        self.hand_spread = hand_spread
        if {"body", "pelvis_quat6d"}.issubset(keys):
            self.kind = "sonic_body29"
        elif {"leg", "waist", "arm"}.issubset(keys) and (
            "pelvis_quat" in keys or "pelvis_quat6d" in keys
        ):
            self.kind = "split33"
            self.split_keys = ("leg", "waist", "arm")
        elif {"leg_pos", "waist_pos", "arm_pos"}.issubset(keys) and (
            "pelvis_quat" in keys or "pelvis_quat6d" in keys
        ):
            self.kind = "split33"
            self.split_keys = ("leg_pos", "waist_pos", "arm_pos")
        else:
            raise RuntimeError(
                f"Unsupported action layout: {self.action_keys}. Expected "
                "sonic_a3 (body + pelvis_quat6d) or a3_config "
                "(leg/leg_pos + waist/waist_pos + arm/arm_pos + pelvis_quat[6d])."
            )

    def decode(
        self,
        action_dict: dict,
        robot_body31: np.ndarray,
        robot_hand20_rad: np.ndarray | None = None,
    ) -> dict:
        if self.kind == "sonic_body29":
            body29 = np.asarray(action_dict["body"][0], dtype=np.float32)          # (H, 29)
            rot6d = np.asarray(action_dict["pelvis_quat6d"][0], dtype=np.float32)  # (H, 6)
            H = body29.shape[0]
            body31 = np.zeros((H, 31), dtype=np.float32)
            body31[:, BODY_INDICES_29] = body29
            body31[:, 15:17] = robot_body31[15:17].astype(np.float32)[None, :]
            out = {
                "leg":               body31[:, 0:12],
                "waist":             body31[:, 12:15],
                "arm":               body31[:, 17:31],
                "pelvis_quat_wxyz":  _rot6d_to_quat_wxyz(rot6d),
            }
            return self._add_hand_chunk(out, action_dict, robot_hand20_rad)

        # split33: old a3_config
        leg_key, waist_key, arm_key = self.split_keys
        leg = np.asarray(action_dict[leg_key][0], dtype=np.float32)
        waist = np.asarray(action_dict[waist_key][0], dtype=np.float32)
        arm = np.asarray(action_dict[arm_key][0], dtype=np.float32)
        if "pelvis_quat6d" in action_dict:
            rot6d = np.asarray(action_dict["pelvis_quat6d"][0], dtype=np.float32)
            pelvis_quat = _rot6d_to_quat_wxyz(rot6d)
        else:
            pelvis_quat = np.asarray(action_dict["pelvis_quat"][0], dtype=np.float32)
        out = {
            "leg": leg,
            "waist": waist,
            "arm": arm,
            "pelvis_quat_wxyz": pelvis_quat,
        }
        return self._add_hand_chunk(out, action_dict, robot_hand20_rad)

    def _add_hand_chunk(
        self,
        out: dict,
        action_dict: dict,
        robot_hand20_rad: np.ndarray | None,
    ) -> dict:
        """Attach full-hand, mapped hand-opening, or gripper actions."""
        if self.hand_action_key is None:
            return out
        if self.hand_action_key in ("hand", "gripper", "gripper_opening"):
            hand = np.asarray(
                action_dict[self.hand_action_key][0], dtype=np.float32
            )
            expected_dim = 20 if self.hand_action_key == "hand" else 2
            if hand.ndim != 2 or hand.shape[1] != expected_dim:
                raise RuntimeError(
                    f"{self.hand_action_key} action must be (H,{expected_dim}), "
                    f"got {hand.shape}"
                )
            out["hand"] = hand
            return out

        # hand_opening: expand the two index-bend actions back to a 20D hand.
        current = np.asarray(robot_hand20_rad, dtype=np.float32).reshape(-1)
        if current.size != 20:
            raise RuntimeError(
                f"hand_opening action requires current 20D hand radians, got {current.size}D"
            )
        opening = np.asarray(action_dict["hand_opening"][0], dtype=np.float32)
        if opening.ndim != 2 or opening.shape[1] != 2:
            raise RuntimeError(f"hand_opening action must be (H, 2), got {opening.shape}")
        try:
            from gr00t.eval.real_robot.hand_mapping import expand_hand_opening_to_hand20
            out["hand"] = expand_hand_opening_to_hand20(
                opening,
                current,
                mapping=self.hand_mapping,
            )
            return out
        except ImportError:
            pass
        hand = np.repeat(current[None, :], opening.shape[0], axis=0)
        hand[:, 4] = opening[:, 0]
        hand[:, 14] = opening[:, 1]
        if self.hand_spread:
            hand = _spread_index_to_five_fingers(hand)
        out["hand"] = hand
        return out


# ============================================================================
# 30Hz → 50Hz upsampling on decoded whole-body chunks
# ============================================================================
#
# NOTE: this ZMQ variant does NOT upsample on the client — a3_server does the
# 30→50Hz interpolation for /wbc/infer/reference_window (joints linear, pelvis
# SLERP; see robointerface master_wholebody_human,
# InterpolationPublisher.wb_snapshot_for_reference_window). This helper is
# retained for diagnostic / standard-mode parity with infer_a3_rtc.py and for
# any future client-side resampling need. Ported from
# feat/a3:scripts_crp/deploy_gr00t.py upsample_action_chunk. Keys match
# A3RobotInterface.step_chunk(mode=whole_body) — ``pelvis_quat_wxyz``
# instead of feat/a3's ``pelvis_quat``.


def upsample_whole_body_chunk(chunk: dict, src_fps: float, dst_fps: float) -> dict:
    """joint action chunk from src_fps to dst_fps.

    - joints: per-column ``np.interp`` linear
    - pelvis: scipy Slerp (wxyz → xyzw → slerp → wxyz)
    - dst_fps <= src_fps (or difference < 1e-6): shallow-copy, no resample
    - H < 2: shallow-copy (single frame, nothing to interpolate)
    - H_new = round((H-1) * dst_fps / src_fps) + 1 over the same [0, (H-1)/src_fps]
    """
    if abs(src_fps - dst_fps) < 1e-6 or dst_fps < src_fps:
        return dict(chunk)

    leg = np.asarray(chunk["leg"], dtype=np.float64)
    waist = np.asarray(chunk["waist"], dtype=np.float64)
    arm = np.asarray(chunk["arm"], dtype=np.float64)
    quat_wxyz = np.asarray(chunk["pelvis_quat_wxyz"], dtype=np.float64)

    h = leg.shape[0]
    if h < 2:
        return dict(chunk)

    duration = (h - 1) / src_fps
    h_new = int(round(duration * dst_fps)) + 1
    t_src = np.arange(h) / src_fps
    t_dst = np.clip(np.linspace(0.0, duration, h_new), t_src[0], t_src[-1])

    def _interp_cols(arr: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(t_dst, t_src, arr[:, d]) for d in range(arr.shape[1])], axis=1
        )

    leg_new = _interp_cols(leg)
    waist_new = _interp_cols(waist)
    arm_new = _interp_cols(arm)

    # scipy Rotation quaternion uses xyzw; wire/policy is wxyz.
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(t_src, Rotation.from_quat(quat_xyzw))
    quat_new_wxyz = slerp(t_dst).as_quat()[:, [3, 0, 1, 2]]

    return {
        "leg": leg_new.astype(np.float32),
        "waist": waist_new.astype(np.float32),
        "arm": arm_new.astype(np.float32),
        "pelvis_quat_wxyz": quat_new_wxyz.astype(np.float32),
    }


# ============================================================================
# Delta-frame reanchor (per-timestep aware, config-driven via
# metadata.action_state_key)
# ============================================================================


class DeltaFrameReanchor:
    """Cross-chunk shift for RELATIVE action dims in NORMALIZED space.

    For each RELATIVE key with an ``action_state_key`` in the checkpoint's
    action_configs (surfaced by get_rtc_metadata), we reanchor the tail of
    the previous chunk from S_prev-frame to S_new-frame at NEW-chunk positions
    [0..L-1], so the model's RTC-pinned prefix decodes to
    (S_new + delta) at flow-matching time t=1, not (S_prev + delta).

    Two representations, dispatched on ``action_format``:

    - **DEFAULT (joint vectors)** — additive. Reanchor in unnormalized space:
          raw_delta_wrt_S_prev = norm_prev[p] * scale[t']  + offset[t']
          raw_delta_wrt_S_new  = raw_delta_wrt_S_prev + (S_prev - S_new)
          norm_new_prefix[p]   = (raw_delta_wrt_S_new  - offset[p]) / scale[p]
      Matches JointActionChunk.to_absolute_chunking (ref.joints + rel.joints).

    - **ROT6D (SO(3) pelvis orientation)** — multiplicative. The previous
      chunk's RELATIVE rot6d is R_rel_prev = R_prev^-1 @ R_action_prev (what
      the model produced, pinned to the OLD frame). We need the prefix to
      represent R_rel_new = R_new^-1 @ R_action_prev (same absolute target,
      new frame). So:
          R_rel_new = R_new^-1 @ R_prev @ R_rel_prev
      Done in UNNORMALIZED rot6d space (denorm → SO(3) compose → renorm),
      because rot6d vectors aren't a vector space — Gram-Schmidt orthonormal-
      izes on the way back to a matrix. The naive additive shift
      (raw + (S_prev - S_new)) is wrong here: it treats 6 rotation coords as
      Euclidean and produces a non-orthonormal result, so the decoded absolute
      pelvis pose jumps at every chunk swap.

    Both representations handle 1D ``(D,)`` and 2D ``(H, D)`` norm stats.

    NB: reanchor operates on the ORIGINAL policy time axis (H at
    policy_output_fps). Upsampling to 50Hz happens AFTER decode, so this class
    is unaffected.
    """

    def __init__(self, metadata: dict, use_it: bool = True):
        self.action_horizon = int(metadata["action_horizon"])
        self.action_keys = list(metadata["action_keys"])
        self.action_dims = metadata["action_dims"]
        self.action_reps = metadata["action_reps"]
        self.action_state_key_map = metadata.get("action_state_key") or {}
        self.action_norm = metadata["action_norm"]
        # Format + type per key, surfaced by get_rtc_metadata so we can
        # dispatch joint-vector math vs SO(3) composition. Older servers
        # that don't surface these fall back to DEFAULT (joint) — same
        # behaviour as before this field existed, which is correct for every
        # RELATIVE key that ISN'T a pure rotation.
        self.action_format = metadata.get("action_format") or {}
        self.action_type = metadata.get("action_type") or {}
        self.state_keys = set(metadata.get("state_keys") or []) | set(
            metadata.get("reference_only_keys") or []
        )
        self.use_relative_action = bool(metadata.get("use_relative_action", False))
        self.enabled = use_it and self.use_relative_action

        # Cache per-RELATIVE-key (slice, scale (H,D), offset (H,D), state_key,
        # fmt). fmt ∈ {"DEFAULT","ROT6D"} picks the reanchor branch.
        self._relative_slices: list[
            tuple[str, slice, np.ndarray, np.ndarray, str, str]
        ] = []
        offset_flat = 0
        for key in self.action_keys:
            dim = int(self.action_dims[key])
            sl = slice(offset_flat, offset_flat + dim)
            offset_flat += dim
            if self.action_reps.get(key, "ABSOLUTE") != "RELATIVE":
                continue
            if not self.enabled:
                continue
            # Prefer explicit action_state_key (e.g. body → body_pos,
            # pelvis_quat6d → pelvis_orient6d); fall back to key name.
            state_key = self.action_state_key_map.get(key) or key
            if state_key not in self.state_keys:
                print(
                    f"[DeltaReanchor] RELATIVE key {key!r} has state_key={state_key!r} "
                    f"but no matching state entry — skipping shift for it."
                )
                continue
            scale, off = self._extract_scale_offset(self.action_norm[key], dim)
            fmt = str(self.action_format.get(key, "DEFAULT")).upper()
            if fmt not in ("DEFAULT", "ROT6D"):
                # Unknown format — safest to fall back to additive, which is
                # correct for any Euclidean vector and only wrong for SO(3).
                print(
                    f"[DeltaReanchor] key {key!r} has unknown action_format "
                    f"{fmt!r} — using DEFAULT (additive) shift. Verify this "
                    f"is right for the modality."
                )
                fmt = "DEFAULT"
            self._relative_slices.append((key, sl, scale, off, state_key, fmt))

        self.flat_action_dim = offset_flat
        if self.enabled and self._relative_slices:
            summary = ", ".join(
                f"{k}(state={sk},fmt={fm},scale.shape={s.shape})"
                for k, _, s, _, sk, fm in self._relative_slices
            )
            print(f"[DeltaReanchor] enabled — {summary}")
        elif use_it and not self.use_relative_action:
            print("[DeltaReanchor] policy reports use_relative_action=False — no shift")
        elif self.enabled and not self._relative_slices:
            print("[DeltaReanchor] no RELATIVE keys with matching state — nothing to shift")

    def _extract_scale_offset(self, norm: dict, dim: int) -> tuple[np.ndarray, np.ndarray]:
        H = self.action_horizon
        if norm["kind"] == "meanstd":
            scale = np.asarray(norm["std"], dtype=np.float32)
            off = np.asarray(norm["mean"], dtype=np.float32)
        else:
            mn = np.asarray(norm["min"], dtype=np.float32)
            mx = np.asarray(norm["max"], dtype=np.float32)
            scale = (mx - mn) / 2.0
            off = (mx + mn) / 2.0
        if scale.ndim == 1:
            assert scale.shape == (dim,), f"1D stats: expected ({dim},), got {scale.shape}"
            scale = np.broadcast_to(scale, (H, dim)).copy()
            off = np.broadcast_to(off, (H, dim)).copy()
        elif scale.ndim == 2:
            assert scale.shape == (H, dim), f"2D stats: expected ({H}, {dim}), got {scale.shape}"
        else:
            raise ValueError(f"norm stats must be 1D or 2D, got ndim={scale.ndim}")
        scale = np.where(np.abs(scale) < 1e-8, np.float32(1.0), scale).astype(np.float32)
        return scale, off.astype(np.float32)

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """(..., 6) rot6d → (..., 3, 3) rotation matrix via Gram-Schmidt.

        Mirrors gr00t/data/stats.py::_rot6d_to_matrix_batch (rows = first two
        rows of R, third row = cross product). We keep a local copy so the
        client doesn't import the data layer.
        """
        r = np.asarray(rot6d, dtype=np.float64).reshape(*rot6d.shape[:-1], 2, 3)
        row1 = r[..., 0, :]
        row2 = r[..., 1, :]
        row1 = row1 / np.linalg.norm(row1, axis=-1, keepdims=True).clip(min=1e-12)
        dot = np.sum(row1 * row2, axis=-1, keepdims=True)
        row2 = row2 - dot * row1
        row2 = row2 / np.linalg.norm(row2, axis=-1, keepdims=True).clip(min=1e-12)
        row3 = np.cross(row1, row2)
        return np.stack([row1, row2, row3], axis=-2)

    @staticmethod
    def _matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
        """(..., 3, 3) → (..., 6) — first two rows flattened."""
        return rot_mat[..., :2, :].reshape(*rot_mat.shape[:-2], 6)

    def _reanchor_rot6d(
        self,
        prev_norm: np.ndarray,        # (L, 6) normalized
        scale_prev: np.ndarray,       # (L, 6)
        off_prev: np.ndarray,         # (L, 6)
        scale_new: np.ndarray,        # (L, 6)
        off_new: np.ndarray,          # (L, 6)
        s_prev_rot6d: np.ndarray,     # (6,) reference frame, UNNORMALIZED rot6d
        s_new_rot6d: np.ndarray,      # (6,) reference frame, UNNORMALIZED rot6d
    ) -> np.ndarray:
        """SO(3) reanchor for a RELATIVE rot6d key.

        prev_norm decodes to R_rel_prev = R_prev^-1 @ R_action_prev (the model's
        old-frame relative target). We want R_rel_new = R_new^-1 @ R_action_prev
        so the same absolute pose lands at the new frame:

            R_rel_new = R_new^-1 @ R_prev @ R_rel_prev

        All composition in SO(3); denorm before, renorm after. Reference frames
        S_prev/S_new come in as UNNORMALIZED rot6d (they're state values, in
        the same physical units the action unnormalizes to).
        """
        # Denormalize the prefix to UNNORMALIZED rot6d.
        rel_prev_raw = prev_norm.astype(np.float64) * scale_prev + off_prev
        R_rel_prev = self._rot6d_to_matrix(rel_prev_raw)            # (L, 3, 3)
        R_prev = self._rot6d_to_matrix(s_prev_rot6d.astype(np.float64))  # (3, 3)
        R_new = self._rot6d_to_matrix(s_new_rot6d.astype(np.float64))    # (3, 3)
        # R_new^-1 @ R_prev @ R_rel_prev  (R_new, R_prev broadcast over L)
        R_new_inv = R_new.T  # SO(3): inverse == transpose
        comp = R_new_inv @ R_prev @ R_rel_prev                       # (L, 3, 3)
        rel_new_raw = self._matrix_to_rot6d(comp)                    # (L, 6)
        # Renormalize with the NEW-chunk per-position stats.
        new_norm = (rel_new_raw - off_new) / scale_new
        return new_norm.astype(np.float32)

    def reanchor_prefix_slice(
        self,
        prev_slice: np.ndarray,
        trigger_idx: int,
        s_prev: dict[str, np.ndarray],
        s_new: dict[str, np.ndarray],
    ) -> np.ndarray:
        L, D_pad = prev_slice.shape
        out = prev_slice.astype(np.float32, copy=True)
        if not self._relative_slices or L == 0:
            return out
        H = self.action_horizon
        assert 0 <= trigger_idx <= H
        assert trigger_idx + L <= H
        for key, sl, scale, offset, sk, fmt in self._relative_slices:
            D = sl.stop - sl.start
            sp = np.asarray(s_prev[sk], dtype=np.float32).reshape(-1)[-D:]
            sn = np.asarray(s_new[sk], dtype=np.float32).reshape(-1)[-D:]
            scale_prev = scale[trigger_idx:trigger_idx + L]
            off_prev = offset[trigger_idx:trigger_idx + L]
            scale_new = scale[:L]
            off_new = offset[:L]
            prev_norm = out[:, sl]
            if fmt == "ROT6D":
                # SO(3) composition path. sp / sn are UNNORMALIZED rot6d state
                # (pelvis_orient6d lives in the same units the action
                # unnormalizes to), exactly what _reanchor_rot6d expects.
                out[:, sl] = self._reanchor_rot6d(
                    prev_norm, scale_prev, off_prev, scale_new, off_new, sp, sn,
                )
            else:
                # Additive joint-vector path (DEFAULT). Matches
                # JointActionChunk.to_absolute_chunking: ref + rel.
                raw = prev_norm * scale_prev + off_prev
                raw = raw + (sp - sn)
                new_norm = (raw - off_new) / scale_new
                out[:, sl] = new_norm.astype(np.float32)
        return out


def _state_dict_from_obs(obs: dict) -> dict[str, np.ndarray]:
    """Squeeze each state field to (D,) in RAW (unnormalized) physical units.

    These are the reference frames the reanchor composes against: for joint
    keys the additive shift does `raw + (S_prev - S_new)` (Euclidean, so raw
    units are fine), and for ROT6D keys the SO(3) path needs UNNORMALIZED
    rot6d state — which is exactly what the obs builder produces (the IMU
    quaternion → rot6d conversion in _sk_pelvis_orient6d is physical, not
    normalized). Squeezing to (D,) keeps the slice bookkeeping uniform.
    """
    out = {}
    for k, v in obs["state"].items():
        arr = np.asarray(v, dtype=np.float32)
        out[k] = arr.reshape(-1, arr.shape[-1])[-1]
    return out


# ============================================================================
# Runtime keys — TaskHolder + KeyStateMachine (from feat/a3 deploy_gr00t.py)
# ============================================================================


def _print_start_prompt() -> None:
    """Print only after the selected runner has finished initialization."""
    print("\n[ws] 请输入 s 启动推理；输入 p 暂停推理", flush=True)


class TaskHolder:
    """Thread-safe live task string. worker reads current() each infer tick,
    KeyStateMachine's background reader calls switch() on digit keys."""

    def __init__(self, tasks: list[str]) -> None:
        self._tasks = list(tasks) if tasks else ["do the task"]
        self._idx = 0
        self._lock = threading.Lock()

    def current(self) -> str:
        with self._lock:
            return self._tasks[self._idx]

    def switch(self, i: int) -> str | None:
        with self._lock:
            if 0 <= i < len(self._tasks):
                self._idx = i
                return self._tasks[i]
            return None

    def listing(self) -> str:
        with self._lock:
            cur = self._idx
            tasks = list(self._tasks)
        return "\n".join(
            f"  {n + 1}{'*' if n == cur else ' '} {t!r}" for n, t in enumerate(tasks)
        )


class KeyStateMachine:
    """IDLE/RUNNING; s→RUNNING, p→IDLE. Background raw-stdin thread; main
    loop polls is_running(). With a task_holder, 1..9 switches preset tasks
    and 'l' lists them."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"

    def __init__(
        self,
        task_holder: "TaskHolder | None" = None,
        *,
        auto_run: bool = False,
        grasp_stop_enabled: bool = False,
    ) -> None:
        self.state = self.IDLE
        self.task_holder = task_holder
        self.grasp_stop_enabled = grasp_stop_enabled
        self._state_lock = threading.Lock()
        self.grasp_state = "UNARMED" if grasp_stop_enabled else "DISABLED"
        self.stop_epoch = 0
        self.cancel_ack_epoch = 0
        self.cancel_failed = False
        self.grasp_heartbeat_at = 0.0
        self._stop = False
        # 'r' sets this; the main loop consumes it and runs the reset itself.
        # Resetting from the reader thread would race the inference loop for
        # the robot's chunk channel.
        self._reset_requested = False
        # 'm' toggles standard <-> rtc_chunk. Same reasoning: the runner swap
        # happens in main(), not here.
        self._mode_switch_requested = False
        try:
            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._tty_ok = True
        except (termios.error, ValueError, OSError):
            self._fd = -1
            self._old_termios = None
            self._tty_ok = False
            if auto_run:
                print("[ws] stdin is not a TTY, --auto-run set → RUNNING")
                self.state = self.RUNNING
            else:
                print(
                    "[ws] stdin is not a TTY — keyboard control disabled. "
                    "Staying IDLE (chunks NOT pushed until state → RUNNING). "
                    "Pass --auto-run to start immediately in headless setups."
                )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._print_help()

    def _print_help(self) -> None:
        print(f"[ws] state={self.state}")
        if self.task_holder is not None:
            print(f"[ws] task[current]={self.task_holder.current()!r}")
            print(self.task_holder.listing())

    def _reader(self) -> None:
        if not self._tty_ok:
            return
        while not self._stop:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "s":
                self.set_running("keyboard s")
            elif ch == "p":
                self.force_idle("keyboard p")
            elif ch == "r":
                if self.grasp_stop_enabled:
                    print("\n[grasp-stop] reset is disabled during a guarded task")
                    continue
                if self.state == self.RUNNING:
                    print("\n[ws] 'r' 忽略: 请先按 p 暂停再复位")
                else:
                    self._reset_requested = True
                    print("\n[ws] → 复位到初始位姿 (dataset frame 0) ...")
            elif ch == "m":
                if self.grasp_stop_enabled:
                    print("\n[grasp-stop] mode switching is disabled")
                    continue
                if self.state == self.RUNNING:
                    print("\n[ws] 'm' 忽略: 请先按 p 暂停再切模式")
                else:
                    self._mode_switch_requested = True
            elif self.task_holder is not None and ch == "l":
                print("\n[ws] tasks (* = current):")
                print(self.task_holder.listing())
            elif self.task_holder is not None and ch in "123456789":
                if self.grasp_stop_enabled:
                    print("\n[grasp-stop] task switching is disabled")
                    continue
                new = self.task_holder.switch(int(ch) - 1)
                if new is not None:
                    print(f"\n[ws] → task[{ch}]: {new!r}(next infer tick)")

    def is_running(self) -> bool:
        with self._state_lock:
            if (self.grasp_stop_enabled and self.grasp_state == "ARMED"
                    and self.state == self.RUNNING
                    and time.monotonic() - self.grasp_heartbeat_at > 0.25):
                self.grasp_state = "SENSOR_UNAVAILABLE"
                self.state = self.IDLE
                self.stop_epoch += 1
                print("[grasp-stop] monitor heartbeat expired; cancelling chunk")
            return self.state == self.RUNNING and (
                not self.grasp_stop_enabled or self.grasp_state == "ARMED"
            )

    def set_running(self, reason: str) -> bool:
        with self._state_lock:
            if self.grasp_stop_enabled and (
                self.grasp_state != "ARMED"
                or time.monotonic() - self.grasp_heartbeat_at > 0.25
            ):
                print(f"[grasp-stop] start refused: {self.grasp_state} / no heartbeat")
                return False
            changed = self.state != self.RUNNING
            self.state = self.RUNNING
        if changed:
            print(f"\n[ws] → RUNNING ({reason})")
        return True

    def force_idle(self, reason: str) -> None:
        with self._state_lock:
            changed = self.state != self.IDLE
            if changed:
                self.stop_epoch += 1
            if self.grasp_stop_enabled and self.grasp_state in ("UNARMED", "ARMED"):
                # Also latch a stop between /grasp/arm and /start, or a delayed
                # start from the monitor could undo the operator's stop.
                self.grasp_state = (
                    "HUMAN_ABORT" if reason.startswith(("keyboard", "http /stop"))
                    else "SAFETY_ABORT"
                )
                changed = True
            self.state = self.IDLE
        if changed:
            print(f"\n[ws] → IDLE ({reason})")

    def arm_grasp(self) -> bool:
        with self._state_lock:
            if not self.grasp_stop_enabled or self.grasp_state != "UNARMED" or self.state != self.IDLE:
                return False
            self.grasp_state = "ARMED"
            self.grasp_heartbeat_at = time.monotonic()
            return True

    def heartbeat_grasp(self) -> bool:
        with self._state_lock:
            if not self.grasp_stop_enabled or self.grasp_state != "ARMED":
                return False
            self.grasp_heartbeat_at = time.monotonic()
            return True

    def latch_grasp(self, result: str) -> bool:
        with self._state_lock:
            if not self.grasp_stop_enabled or self.grasp_state != "ARMED":
                return False
            self.grasp_state = result
            if self.state == self.RUNNING:
                self.stop_epoch += 1
            self.state = self.IDLE
        print(f"[grasp-stop] latched {result}; stop_epoch={self.stop_epoch}")
        return True

    def ack_cancel(self, ok: bool) -> None:
        with self._state_lock:
            if ok:
                self.cancel_ack_epoch = self.stop_epoch
            else:
                self.cancel_failed = True

    def status_snapshot(self) -> dict:
        with self._state_lock:
            return {
                "state": self.state,
                "grasp_state": self.grasp_state,
                "stop_epoch": self.stop_epoch,
                "cancel_ack_epoch": self.cancel_ack_epoch,
                "cancel_failed": self.cancel_failed,
            }

    def take_reset_request(self) -> bool:
        """One-shot reset request consumed by the main loop."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    def take_mode_switch_request(self) -> bool:
        """One-shot mode switch request consumed by the main loop."""
        if self._mode_switch_requested:
            self._mode_switch_requested = False
            return True
        return False

    def close(self) -> None:
        self._stop = True
        if self._tty_ok and self._old_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except termios.error:
                pass


def _load_task_list(prompt_yaml: str | None, cli_task: str | None) -> list[str]:
    """prompt.yaml is single source of truth. --task, if given, is prepended
    as tasks[0] (boot default); tasks are deduped preserving order."""
    file_tasks: list[str] = []
    if prompt_yaml:
        p = Path(prompt_yaml).expanduser()
        if not p.is_absolute():
            p = REPO / p
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text()) or {}
                raw = data.get("tasks") if isinstance(data, dict) else None
                if isinstance(raw, list):
                    file_tasks = [str(t) for t in raw if isinstance(t, str) and t.strip()]
            except Exception as e:
                print(f"[ws] failed to read {p}: {e} (ignoring)")
        else:
            print(f"[ws] prompt yaml not found at {p}, falling back to --task only")

    seen: set[str] = set()
    tasks: list[str] = []
    for t in [cli_task, *file_tasks]:
        if t and t not in seen:
            seen.add(t)
            tasks.append(t)
    if not tasks:
        tasks = ["Walk forward. Move and avoid all obstacles"]
    return tasks


# ============================================================================
# Async RTC chunk-train runner (REMOTE ZMQ PolicyClient)
# ============================================================================


class AsyncRTCChunkTrainRunner:
    """Chunk-mode train-time RTC over REMOTE PolicyClient (ZMQ) + A3RobotInterface.

    Direct port of pi/inference/infer.py::AsyncRTCChunkTrainRunner. Two threads:

    - **Main thread**: virtual tick advances ``self.t`` on the model's native
      policy axis (30Hz). It does NOT call robot.step(); a3_server plays the
      30Hz chunk and publishes /wbc/infer/reference_window (interpolating 30→50Hz
      internally).
    - **Inference thread**: waits ``self.t >= s_min``, snapshots (s, A_prev_raw,
      S_prev), releases the lock, calls the remote policy server, then
      re-acquires the lock, installs the new chunk via
      ``A3RobotInterface.step_chunk(mode=whole_body,
      emit_mode="reference_window", s_used_local=...)``, and resets
      ``self.t = actual_delay`` (returned by server-atomic swap).

    No client-side upsampling: the client sends the 30Hz policy chunk verbatim;
    a3_server does the 30→50Hz interpolation (joints linear, pelvis SLERP) for
    the reference window. RTC's ``self.t``, ``s_min``, ``Q``, prefix, and
    server ``actual_delay`` all live on the 30Hz policy axis.
    """

    def __init__(
        self,
        policy_client: PolicyClient,
        robot: A3RobotInterface,
        obs_builder: A3ObsBuilder,
        decoder: A3ActionDecoder,
        reanchor: DeltaFrameReanchor,
        task_holder: TaskHolder,
        key_sm: KeyStateMachine,
        args,
        metadata: dict,
    ):
        self.policy = policy_client  # remote ZMQ PolicyClient
        self.robot = robot
        self.obs_builder = obs_builder
        self.decoder = decoder
        self.reanchor = reanchor
        self.task_holder = task_holder
        self.key_sm = key_sm
        self.args = args
        self.metadata = metadata

        # Horizon in POLICY time (H). Single axis: policy == wire (both 30Hz on
        # the client); a3_server's 50Hz interpolation is internal to it.
        self.policy_horizon = int(metadata["action_horizon"])
        self.action_horizon = self.policy_horizon
        self.wire_action_horizon = self.policy_horizon  # == policy (no client upsample)
        # rtc_max_delay from the checkpoint is in POLICY action positions
        # (see gr00t/model/gr00t_n1d7/gr00t_n1d7.py::_sample_actions_with_prefix
        # — `rtc_delay` indexes action positions in the model's H-length chunk).
        # This is the sole model-facing delay limit; transport upsampling does
        # not create additional RTC positions.
        self.max_delay_policy = int(metadata["rtc_max_delay"])
        if self.max_delay_policy <= 0 and getattr(args, "rtc_max_delay_override", None) is None:
            raise RuntimeError(
                "Checkpoint reports rtc_max_delay=0 — retrain with train-time "
                "RTC or pass --rtc_max_delay_override for a dry run (garbage "
                "output). See memory `gr00t-rtc-train-bug` for the setup.py "
                "fix history."
            )
        if getattr(args, "rtc_max_delay_override", None) is not None:
            self.max_delay_policy = int(args.rtc_max_delay_override)

        self.s_min = int(args.exec_steps)  # policy frames (same unit as RTC training)
        # policy_output_fps is the model/native + wire send rate (30Hz). a3_server
        # does the 30→50Hz interpolation for /wbc/infer/reference_window, so the client
        # works entirely on the 30Hz policy axis: chunk_fps == source_fps ==
        # policy_output_fps. wire_fps is retained only for diagnostics / the
        # reference_window publish rate on the server side.
        self.policy_output_fps = float(args.policy_output_fps)
        self.wire_fps = float(args.wire_fps)
        self.chunk_fps = self.policy_output_fps
        self.interval = 1.0 / self.policy_output_fps
        # Single axis now: policy == wire (both 30Hz on the client). The server's
        # 50Hz reference-window interpolation is internal to a3_server and does
        # not create extra RTC positions on the client side.
        self._policy_per_wire = 1.0
        self._wire_per_policy = 1.0

        self.action_keys = list(metadata["action_keys"])
        self.action_dims = {k: int(metadata["action_dims"][k]) for k in self.action_keys}
        self.flat_action_dim = sum(self.action_dims[k] for k in self.action_keys)

        # sync primitives
        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        # shared state
        self.t = 0
        self.A_cur_raw = None            # (1, policy_H, D_pad) normalized model output
        self.A_cur_state_dict = None     # {state_key: (D,)} — S_prev for reanchor
        self.A_cur_body31 = None         # (31,) robot BODY_31 at chunk-load time
        self._chunk_seq = 0
        self._last_infer_timing_ms = {}

        # Send/cancel generation guard. Bumped on every RUNNING→IDLE transition
        # (cancel) and captured by the inference thread BEFORE its /send_chunk
        # POST. After the POST returns we compare: if the generation changed,
        # a cancel landed while our send was in flight — the send may have
        # re-installed a chunk AFTER the cancel, defeating the pause. We then
        # re-cancel to restore the IDLE invariant. See _send_chunk + the
        # post-send check in _inference_loop / _cold_start_locked.
        self._cancel_gen = 0
        # True from immediately before a /send_chunk POST until its post-send
        # pause guard (including any compensating cancel) is complete.  A fast
        # p→s restart waits for this to clear before cold-starting, otherwise a
        # stale send's compensating cancel could erase the new run's first
        # chunk.
        self._send_in_flight = False
        self._deferred_cancel_ack = False

        # Delay ring buffer on the original policy axis. a3_server returns
        # this primary actual_delay in source_fps units; its wire delay is
        # retained separately only for server-buffer bookkeeping.
        self.delay_buf_size = 4
        self.Q = deque([min(self.s_min, self.max_delay_policy - 1)], maxlen=self.delay_buf_size)
        # Margin is in policy frames, exactly as used by the checkpoint's RTC
        # training objective.
        self._delay_safety_margin = int(getattr(args, "rtc_delay_margin", 2))
        self._server_adaptive_transition = (
            str(getattr(args, "server_transition", "fixed")).lower() == "adaptive"
        )
        # 关掉 RTC train-time prefix pin(sim 用):每次推理都纯 sample,不喂上一
        # chunk 的 reanchored 尾巴。RTC 的 chunk 原子替换(s_used_local/actual_delay)
        # 照常,只是相邻 chunk 独立采样;替换边界的 jump 交给 server adaptive_transition
        # 平滑(故 sim 里配 --server_transition adaptive)。真机默认 False(保留 pin)。
        self.no_rtc_prefix = bool(getattr(args, "no_rtc_prefix", False))
        # Exact start of the server's currently loaded *wire* buffer within
        # the logical unsliced wire chunk. This cannot be reconstructed from
        # policy delay after 30→50Hz rounding, so the server returns it as
        # ``actual_delay_wire`` alongside the primary policy actual_delay.
        self._server_chunk_start_idx_wire = 0
        self._server_chunk_start_idx_policy = 0

        # RTT probe (diagnostic; server-atomic swap doesn't use this)
        try:
            rtt_sec = robot.measure_rtt(num_samples=30, warmup=5)
            if rtt_sec is not None:
                print(
                    f"[RTT-Probe] median RTT={rtt_sec * 1000:.1f}ms, "
                    f"one-way≈{rtt_sec * 500:.1f}ms ≈ "
                    f"{rtt_sec / 2 * self.chunk_fps:.2f} frames @ {self.chunk_fps}Hz"
                )
            else:
                print("[RTT-Probe] unavailable")
        except Exception as e:
            print(f"[RTT-Probe] error: {e}")

        # ---- chunk 记录(默认开启,存到 $A3_LOG_DIR)----
        # 每次 send 记一条完整 wire chunk、其 policy-axis actual_delay，以及
        # server 内部实际切掉的 actual_delay_wire，供事后对齐检查。
        # 目录优先级:--record-chunks-dir > $A3_LOG_DIR > CWD;传 "off" 关闭。
        rec_dir = getattr(args, "record_chunks_dir", None)
        if rec_dir is None:
            rec_dir = os.environ.get("A3_LOG_DIR") or os.getcwd()
        self._record_enabled = bool(rec_dir) and str(rec_dir).lower() != "off"
        self._record_dir = rec_dir if self._record_enabled else None
        self._run_id = os.environ.get("A3_RUN_ID", "") or "run"
        self._chunk_log: list = []
        self._chunk_dumped = False
        # policy 轴(30Hz)记录:验证 RTC 冻结前缀 / 切换拼接用
        self._last_delay_policy = 0       # 传给模型的 delay(policy 帧,冻结数)
        self._last_s_used_policy = -1     # 前缀在上一 chunk 里的起始 index(policy 帧)
        # 三版本完整 chunk 原料(infer_chunk 暂存,_send_chunk 打进 rec):
        #   ② 归一化 flat (H, D_pad);③ 反归一化 per-key {body:(H,29), pelvis_quat6d:(H,6)}
        # ① 绝对角 = _send_chunk 收到的 chunks 参数(leg/waist/arm/pelvis_quat_wxyz)。
        # 切换段/prefix 段不物理存 —— HTML 里用 chunk[c-1] + s_used_policy[c] 事后切。
        self._last_norm_flat = None
        self._last_denorm = None

        # ---- normalized rot6d 中间量记录(pelvis RTC pin 诊断)----
        # 目标:区分"模型没 pin 住前缀" vs "reanchor/decode stats 不一致导致
        # abs_cur[0] != abs_prev[s]"。每次推理捕获四份 pelvis normalized:
        #   prev_norm     : 上一 chunk raw 输出 [s:s+L] 的 pelvis 列(reanchor 输入)
        #   new_norm      : reanchor 重写后填进 prefix_raw[0,:L] 的 pelvis 列
        #   model_out_norm: 模型本次返回 A_new_raw 的 pelvis 列(整 H)
        # 以及 S_prev/S_new 的 pelvis_orient6d state(reanchor 参考帧)。
        # 真值只有"ROT6D pelvis"这一列;非 RTC / 无 pelvis_quat6d 时各字段留 None。
        self._pelvis_norm_sl: slice | None = self._find_pelvis_norm_slice(metadata)
        self._last_prev_norm_pelvis = None       # (L,6) reanchor 前
        self._last_new_norm_pelvis = None        # (L,6) reanchor 后(=喂进 prefix)
        self._last_model_out_norm_pelvis = None  # (H,6) 模型输出
        self._last_s_prev_pelvis = None          # (6,) UNNORMALIZED rot6d state
        self._last_s_new_pelvis = None           # (6,)
        if self._record_enabled and self._pelvis_norm_sl is not None:
            print(
                "[record] pelvis normalized rot6d 中间量记录已开启 "
                f"(flat slice={self._pelvis_norm_sl.start}:{self._pelvis_norm_sl.stop})"
            )
        if self._record_enabled:
            os.makedirs(self._record_dir, exist_ok=True)
            print(
                f"[record] chunk 记录已开启 -> "
                f"{os.path.join(self._record_dir, f'chunks_{self._run_id}.npz')}"
            )
            atexit.register(self._dump_chunks)  # finally 之外的兜底

        self.running = False

    # ---------------- policy call ----------------

    @staticmethod
    def _find_pelvis_norm_slice(metadata: dict) -> slice | None:
        """Locate the pelvis rot6d columns in the flat normalized action tensor.

        The flat layout is action_keys concatenated in metadata order, each
        taking its action_dims[key] columns. We return the slice for the
        ``pelvis_quat6d`` key (the RELATIVE rot6d pelvis that RTC pins). None
        when this checkpoint has no such key (e.g. non-RTC / direct-quat).
        Used by the pin-diagnostic recorder to extract the normalized rot6d
        column without re-running the action decoder.
        """
        keys = list(metadata.get("action_keys") or [])
        dims = metadata.get("action_dims") or {}
        off = 0
        for k in keys:
            d = int(dims.get(k, 0))
            if k == "pelvis_quat6d":
                return slice(off, off + d)
            off += d
        return None

    def _reset_norm_record(self) -> None:
        """Clear the per-inference scratch fields (normalized-rot6d pin 诊断 +
        三版本 chunk 原料)。

        Called at the start of each inference tick (and on cold-start) so a
        stale previous-tick capture can never leak into the next chunk's npz
        record when the current tick aborts before _send_chunk runs.
        """
        self._last_prev_norm_pelvis = None
        self._last_new_norm_pelvis = None
        self._last_model_out_norm_pelvis = None
        self._last_s_prev_pelvis = None
        self._last_s_new_pelvis = None
        self._last_norm_flat = None
        self._last_denorm = None

    def _policy_infer(self, obs, prefix_raw, delay):
        """`delay` is in POLICY frames — see caller in _inference_loop
        (d_used_policy). Model's rtc_delay indexes action positions."""
        options = None
        if prefix_raw is not None and delay > 0:
            options = {
                "rtc_mode": "train_time",
                "rtc_delay": int(min(delay, self.max_delay_policy - 1)),
                "action_prefix": prefix_raw,
            }
        return self.policy.get_action(obs, options=options)

    def infer_chunk(self, robot_obs: dict, prefix_raw=None, delay: int = 0):
        """Call the policy server, decode into a whole-body 33D chunk dict AT
        POLICY FPS (30Hz). No client-side upsampling — a3_server does the
        30→50Hz interpolation for /wbc/infer/reference_window.

        Returns (chunks_policy, A_new_raw (1, policy_H, D_pad), state_dict,
        body31_at_infer). ``A_new_raw`` is the ORIGINAL policy-time-axis
        normalized output — DeltaFrameReanchor operates on this axis.
        """
        infer_start = time.perf_counter()
        stage_start = infer_start
        obs, body31 = self.obs_builder.build(robot_obs, self.task_holder.current())
        state_dict = _state_dict_from_obs(obs)
        obs_build_done = time.perf_counter()

        action_dict, info = self._policy_infer(obs, prefix_raw, delay)
        policy_done = time.perf_counter()

        raw = info.get("action_pred_normalized") if isinstance(info, dict) else None
        if raw is None:
            raise RuntimeError(
                "Policy server returned no info['action_pred_normalized']. "
                "Check gr00t/policy/gr00t_policy.py — get_action must echo the "
                "normalized action in info."
            )
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[None, ...]
        decode_start = time.perf_counter()
        chunks_policy = self.decoder.decode(
            action_dict, body31, self.obs_builder.hand_position_rad(robot_obs)
        )
        decode_done = time.perf_counter()
        if self._record_enabled:
            self._last_delay_policy = int(delay)
            # 记录三版本完整 chunk 的 ②③ 原料(① = chunks_policy 已由 _send_chunk 存):
            #   ② 归一化 flat: raw[0] (H, D_pad),含 body(29)+pelvis_quat6d(6) 归一化值
            #   ③ 反归一化 per-key: action_dict 各 key(body 29D 绝对角、pelvis rot6d 6D)
            # 存到实例变量供 _send_chunk 打进 rec(与 chunks_policy 同一次 send)。
            self._last_norm_flat = raw[0].astype(np.float32).copy()  # (H, D_pad)
            self._last_denorm = {
                k: np.asarray(action_dict[k][0], dtype=np.float32).copy()
                for k in self.action_keys
                if k in action_dict
            }
            # 模型输出整 H 的 pelvis normalized 列(pin 诊断:与 reanchor 后的
            # new_norm[:delay] 比对,相等=pin 住,不等=模型把前缀也去噪了)。
            if self._pelvis_norm_sl is not None:
                self._last_model_out_norm_pelvis = np.asarray(
                    raw[0, :, self._pelvis_norm_sl], dtype=np.float32
                ).copy()  # (H, 6)
        infer_done = time.perf_counter()
        obs_build_ms = (obs_build_done - stage_start) * 1000.0
        policy_ms = (policy_done - obs_build_done) * 1000.0
        decode_ms = (decode_done - decode_start) * 1000.0
        infer_total_ms = (infer_done - infer_start) * 1000.0
        self._last_infer_timing_ms = {
            "total": infer_total_ms,
            "obs_build": obs_build_ms,
            "policy": policy_ms,
            "decode": decode_ms,
            "other": max(
                0.0,
                infer_total_ms - obs_build_ms - policy_ms - decode_ms,
            ),
        }
        # No client-side upsampling — a3_server interpolates 30→50Hz for the
        # reference window. chunk_fps == source_fps == policy_output_fps (30).
        return chunks_policy, raw, state_dict, body31

    # ---------------- server progress ----------------

    def _server_progress_from_snapshot(
        self, chunk_prog_raw
    ) -> Optional[tuple[float, float]]:
        """Return ``(global_policy, global_policy)`` server progress.

        Single axis in this ZMQ variant: the client sends 30Hz chunks, so the
        server stores 30Hz waypoints and ``get_chunk_progress``'s ``wb`` is
        already a 30Hz (policy) played index. The server's 30→50Hz interpolation
        for /wbc/infer/reference_window is internal and doesn't affect this index.
        Both tuple entries are the same value (``_policy_per_wire == 1.0``); the
        pair is kept for signature parity with the local-upsample variant.
        """
        if not isinstance(chunk_prog_raw, dict):
            return None
        played = chunk_prog_raw.get("wb")
        if played is None:
            played = chunk_prog_raw.get("arm")
        if played is None:
            return None
        global_policy = float(played) + float(self._server_chunk_start_idx_policy)
        return global_policy, global_policy

    # ---------------- chunk send ----------------

    def _send_chunk(
        self,
        chunks: dict,
        s_used_local: Optional[int],
        is_init: bool = False,
    ) -> Optional[int]:
        """POST the 30Hz whole-body chunk to a3_server. Server does the atomic
        slice in its interp_pub lock AND the 30→50Hz interpolation for
        /wbc/infer/reference_window (see robointerface master_wholebody_human).
        ``actual_delay`` is returned on the original policy (30Hz) axis; since
        chunk_fps == source_fps (== policy_output_fps), actual_delay_wire ==
        actual_delay and no wire bookkeeping is needed on the client.

        Returns None unless the server explicitly acknowledges ``ok=True``.
        A timeout/transport failure is ambiguous (the server may have installed
        the chunk before its response was lost), so callers must cancel and
        return to IDLE instead of committing local RTC state optimistically.
        """
        sul = None if is_init else s_used_local
        chunk_id = self._chunk_seq
        self._chunk_seq += 1
        try:
            payload = {
                "leg": chunks["leg"],
                "waist": chunks["waist"],
                "arm": chunks["arm"],
                "pelvis_quat_wxyz": chunks["pelvis_quat_wxyz"],
            }
            if "hand" in chunks:
                payload["hand"] = chunks["hand"]
                payload["hand_value"] = self.decoder.hand_value
            resp = self.robot.step_chunk(
                payload,
                chunk_fps=self.policy_output_fps,
                wait=False,
                s_used_local=sul,
                source_fps=self.policy_output_fps,
                chunk_id=chunk_id,
                adaptive_transition=self._server_adaptive_transition,
                emit_mode="reference_window",
            )
            if not isinstance(resp, dict) or not resp.get("ok", False):
                print(f"  [ChunkSend] server did not acknowledge chunk_id={chunk_id}: {resp}")
                return None
            if "actual_delay_wire" not in resp:
                print(
                    "  [ChunkSend] incompatible a3_server: missing "
                    "actual_delay_wire; deploy the matching server revision"
                )
                return None
            actual_delay = int(resp.get("actual_delay", 0) or 0)
            actual_delay_wire = int(resp.get("actual_delay_wire", 0) or 0)
        except Exception as e:
            print(f"  [ChunkSend] chunk_id={chunk_id} exception: {e}")
            return None

        self._server_chunk_start_idx_wire = max(0, actual_delay_wire)
        self._server_chunk_start_idx_policy = max(0, actual_delay)
        if self._record_enabled:
            # 记录一条(全部 30Hz policy 轴,copy 防后续复用被改)。三个数据版本的
            # 完整 chunk 都存,HTML 里可切换查看 / 按 actual_delay / s_used_policy
            # 事后切出执行段、切换段、prefix 段:
            #   ① 绝对角 (chunks 参数,decode 后): leg(12)/waist(3)/arm(14) 绝对关节角
            #      + pelvis_quat_wxyz(4) 四元数
            #   ② 归一化 flat (raw[0], H×D_pad): 模型直接输出,含 body(29)+
            #      pelvis_quat6d(6) 归一化值(RTC prefix 就在这一层)
            #   ③ 反归一化 per-key (action_dict): body(29) 绝对关节角 + pelvis rot6d(6)
            #      (server decode_action 反归一化后、client 转 quat 前)
            # 另存 actual_delay(server 执行段起点)、s_used_policy(prefix 起点)、
            # delay_policy(模型冻结帧数) 等标量,以及 pelvis norm pin 诊断中间量。
            try:
                rec = {
                    "chunk_id": int(chunk_id),
                    "t_ns": time.monotonic_ns(),
                    "wall_ns": time.time_ns(),
                    "actual_delay": int(actual_delay),
                    "actual_delay_wire": int(actual_delay_wire),
                    "s_used_local": (-1 if sul is None else int(sul)),
                    "is_init": bool(is_init),
                    "delay_policy": int(self._last_delay_policy),
                    "s_used_policy": int(self._last_s_used_policy),
                    # ① 绝对角完整 chunk。
                    "leg": np.array(chunks["leg"], dtype=np.float32),
                    "waist": np.array(chunks["waist"], dtype=np.float32),
                    "arm": np.array(chunks["arm"], dtype=np.float32),
                    "pelvis_quat_wxyz": np.array(chunks["pelvis_quat_wxyz"], dtype=np.float32),
                }
                # ② 归一化 flat (H, D_pad)。
                if self._last_norm_flat is not None:
                    rec["norm_flat"] = np.array(self._last_norm_flat, dtype=np.float32)
                # ③ 反归一化 per-key。sonic_a3: body(H,29) + pelvis_quat6d(H,6)。
                # split33: leg/waist/arm/pelvis_quat[6d]。按实际 key 存。
                if self._last_denorm is not None:
                    for k, v in self._last_denorm.items():
                        rec[f"denorm_{k}"] = np.array(v, dtype=np.float32)
                # pelvis normalized rot6d 中间量(pin 诊断)。cold-start / 无前缀时为 None。
                if self._pelvis_norm_sl is not None:
                    if self._last_prev_norm_pelvis is not None:
                        rec["prev_norm_pelvis"] = np.array(
                            self._last_prev_norm_pelvis, dtype=np.float32
                        )  # (avail, 6)
                    if self._last_new_norm_pelvis is not None:
                        rec["new_norm_pelvis"] = np.array(
                            self._last_new_norm_pelvis, dtype=np.float32
                        )  # (avail, 6)
                    if self._last_model_out_norm_pelvis is not None:
                        rec["model_out_norm_pelvis"] = np.array(
                            self._last_model_out_norm_pelvis, dtype=np.float32
                        )  # (H, 6)
                    if self._last_s_prev_pelvis is not None:
                        rec["s_prev_pelvis"] = np.array(
                            self._last_s_prev_pelvis, dtype=np.float32
                        )  # (6,)
                    if self._last_s_new_pelvis is not None:
                        rec["s_new_pelvis"] = np.array(
                            self._last_s_new_pelvis, dtype=np.float32
                        )  # (6,)
                self._chunk_log.append(rec)
            except Exception as e:  # noqa: BLE001 — 记录失败绝不影响控制
                print(f"[record] append failed: {e}")
        if not is_init:
            H_policy = self.action_horizon
            sent = max(0, H_policy - actual_delay)
            print(
                f"  [ChunkSend] chunk_id={chunk_id} "
                f"server_actual_delay={actual_delay}(policy), "
                f"wire={actual_delay_wire}, s_used_local={sul}(policy) "
                f"effective={sent}/{H_policy}"
            )
        return actual_delay

    def _dump_chunks(self) -> None:
        """把记录的 chunk 落盘成 chunks_<RUN_ID>.npz(np.load 可读)。

        幂等:finally 与 atexit 都会调,靠 _chunk_dumped 防重复写。
        变长 H 按 Hmax 补零,另存每条真实 H。
        """
        if not self._record_enabled or self._chunk_dumped:
            return
        self._chunk_dumped = True
        log = self._chunk_log
        if not log:
            print("[record] 无 chunk 记录,跳过落盘")
            return
        N = len(log)
        Hs = [int(r["leg"].shape[0]) for r in log]
        Hmax = max(Hs)

        def _pad(key, D):
            a = np.zeros((N, Hmax, D), dtype=np.float32)
            for i, r in enumerate(log):
                h = r[key].shape[0]
                a[i, :h] = r[key]
            return a

        leg_names = [
            "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
        ]
        waist_names = ["waist_yaw", "waist_roll", "waist_pitch"]
        arm_names = [
            "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wr_roll", "L_wr_pitch", "L_wr_yaw",
            "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wr_roll", "R_wr_pitch", "R_wr_yaw",
        ]
        out = os.path.join(self._record_dir, f"chunks_{self._run_id}.npz")
        save = dict(
            chunk_id=np.array([r["chunk_id"] for r in log], dtype=np.int64),
            t_ns=np.array([r["t_ns"] for r in log], dtype=np.int64),
            wall_ns=np.array([r["wall_ns"] for r in log], dtype=np.int64),
            actual_delay=np.array([r["actual_delay"] for r in log], dtype=np.int64),
            actual_delay_wire=np.array(
                [r["actual_delay_wire"] for r in log], dtype=np.int64
            ),
            s_used_local=np.array([r["s_used_local"] for r in log], dtype=np.int64),
            is_init=np.array([r["is_init"] for r in log], dtype=bool),
            delay_policy=np.array([r.get("delay_policy", -1) for r in log], dtype=np.int64),
            s_used_policy=np.array([r.get("s_used_policy", -1) for r in log], dtype=np.int64),
            H=np.array(Hs, dtype=np.int64),
            # 推理输出的完整 30Hz chunk (N, Hmax, D)。
            leg=_pad("leg", 12),
            waist=_pad("waist", 3),
            arm=_pad("arm", 14),
            pelvis_quat_wxyz=_pad("pelvis_quat_wxyz", 4),
            wire_fps=np.float64(self.wire_fps),
            chunk_fps=np.float64(self.policy_output_fps),
            policy_fps=np.float64(self.policy_output_fps),
            policy_horizon=np.int64(self.policy_horizon),
            leg_names=np.array(leg_names, dtype=object),
            waist_names=np.array(waist_names, dtype=object),
            arm_names=np.array(arm_names, dtype=object),
        )

        # 变长键的通用补零打包(每条真实长度另存)。
        def _pad_var(keys_dims, len_key):
            """keys_dims: {out_name: (rec_key, D)}; 所有 rec_key 共享同一 Hmax
            和 len 数组(名为 len_key)。缺失条目 → 全零 + len=0。"""
            lens = [
                int(r[next(iter(keys_dims.values()))[0]].shape[0])
                if next(iter(keys_dims.values()))[0] in r else 0
                for r in log
            ]
            hmax = max(lens) if max(lens) > 0 else 1
            save[len_key] = np.array(lens, dtype=np.int64)
            for out_name, (rec_key, D) in keys_dims.items():
                a = np.zeros((N, hmax, D), dtype=np.float32)
                for i, r in enumerate(log):
                    if rec_key in r:
                        arr = r[rec_key]
                        a[i, : arr.shape[0]] = arr
                save[out_name] = a

        # ② 归一化 flat (N, Hmax, D_pad)。D_pad = 模型输出扁平维度(sonic_a3=35)。
        # 记录 action_keys / dims / 各 key 在 flat 里的 offset,HTML 侧据此切列。
        if any("norm_flat" in r for r in log):
            dpad = max(
                (r["norm_flat"].shape[1] for r in log if "norm_flat" in r),
                default=0,
            )
            nf = np.zeros((N, Hmax, dpad), dtype=np.float32)
            for i, r in enumerate(log):
                if "norm_flat" in r:
                    arr = r["norm_flat"]
                    nf[i, : arr.shape[0], : arr.shape[1]] = arr
            save["norm_flat"] = nf
            save["norm_flat_dpad"] = np.int64(dpad)
            # flat 布局: action_keys 顺序拼接,每 key 占 action_dims[key] 列。
            akeys = list(self.action_keys)
            adims = [int(self.action_dims[k]) for k in akeys]
            save["action_keys"] = np.array(akeys, dtype=object)
            save["action_dims"] = np.array(adims, dtype=np.int64)
            offs, o = [], 0
            for dd in adims:
                offs.append(o)
                o += dd
            save["action_offsets"] = np.array(offs, dtype=np.int64)

        # ③ 反归一化 per-key (denorm_<key>)。sonic_a3: denorm_body(H,29) +
        # denorm_pelvis_quat6d(H,6);split33: denorm_leg/waist/arm/pelvis_quat[6d]。
        # 变长 key 集合按实际记录到的 denorm_* 动态打包。
        denorm_keys = sorted({
            k for r in log for k in r.keys() if k.startswith("denorm_")
        })
        if denorm_keys:
            dmap = {}
            for dk in denorm_keys:
                D = max(
                    (r[dk].shape[1] for r in log if dk in r), default=0
                )
                dmap[dk] = (dk, D)
            _pad_var(dmap, "H_denorm")
            save["denorm_keys"] = np.array(denorm_keys, dtype=object)
        # pelvis normalized rot6d 中间量(pin 诊断)。三条前缀/输出数组都在
        # policy 轴,长度可变(prev/new = avail=H-s_used,model_out = H),
        # 按 Hpmax 补零,另存每条真实长度。S_prev/S_new 是单帧 (6,)。
        if self._pelvis_norm_sl is not None and any(
            "model_out_norm_pelvis" in r for r in log
        ):
            Hpn = [
                int(r["model_out_norm_pelvis"].shape[0])
                if "model_out_norm_pelvis" in r else 0
                for r in log
            ]
            Hpnmax = max(Hpn) if max(Hpn) > 0 else 1

            def _pad_norm(key, D):
                a = np.zeros((N, Hpnmax, D), dtype=np.float32)
                for i, r in enumerate(log):
                    if key in r:
                        arr = r[key]
                        a[i, : arr.shape[0]] = arr
                return a

            save["Hnorm"] = np.array(Hpn, dtype=np.int64)
            save["prev_norm_pelvis"] = _pad_norm("prev_norm_pelvis", 6)
            save["new_norm_pelvis"] = _pad_norm("new_norm_pelvis", 6)
            save["model_out_norm_pelvis"] = _pad_norm("model_out_norm_pelvis", 6)
            # S_prev/S_new: 单帧 (6,),缺失行留 NaN 以便前端识别。
            sp = np.full((N, 6), np.nan, dtype=np.float32)
            sn = np.full((N, 6), np.nan, dtype=np.float32)
            for i, r in enumerate(log):
                if "s_prev_pelvis" in r:
                    sp[i] = r["s_prev_pelvis"]
                if "s_new_pelvis" in r:
                    sn[i] = r["s_new_pelvis"]
            save["s_prev_pelvis"] = sp
            save["s_new_pelvis"] = sn
        try:
            np.savez_compressed(out, **save)
            print(f"[record] 已保存 {N} 条 chunk -> {out}")
        except Exception as e:  # noqa: BLE001
            print(f"[record] 落盘失败: {e}")

    # ---------------- threads ----------------

    def _cancel_and_reset_locked(self, reason: str) -> None:
        """RUNNING → IDLE handler: server-side cancel + local state clear.
        Caller must hold self.C (self.M).

        server:
            robot.cancel_chunk() → clears _wb_q31_chunk etc. →
            wb_snapshot_for_reference_window returns None →
            a3_server stops publishing /wbc/infer/reference_window this tick.
            Motor targets on the interpolator hold at the current position
            (per interp_publisher.cancel_chunk docstring).

        local RTC state:
            A_cur_raw / A_cur_state_dict / body31 / server_chunk_start_idx
            all wiped so the next RUNNING edge triggers a fresh cold-start.
            t is reset to 0 so the virtual clock re-arms cleanly.
            Q is left seeded at s_min so the first re-start uses the same
            conservative delay estimate the runner used at construction.

        race fix:
            Bumps ``_cancel_gen``. The inference thread captures the gen
            before its /send_chunk POST and re-checks after — if the gen
            advanced, a cancel was issued while our send was in flight and
            may have re-installed a chunk post-cancel. The caller re-cancels
            to restore the IDLE invariant. Without this, a p press that
            lands between the pre-send RUNNING check and the POST arrival
            would be silently undone by the in-flight send.
        """
        # Bump FIRST, under the lock, so any send that completes after this
        # point sees a stale gen and knows to re-cancel.
        self._cancel_gen += 1
        my_gen = self._cancel_gen
        # Cancel is a POST; release the lock while the HTTP call is in flight
        # to avoid holding the inference thread's Condition needlessly.
        self.C.release()
        try:
            ok = self.robot.cancel_chunk()
        except Exception as e:
            ok = False
            print(f"  [Cancel] exception: {e}")
        finally:
            self.C.acquire()
        if self._send_in_flight and ok:
            self._deferred_cancel_ack = True
        else:
            self.key_sm.ack_cancel(bool(ok))
        self.A_cur_raw = None
        self.A_cur_state_dict = None
        self.A_cur_body31 = None
        self._server_chunk_start_idx_wire = 0
        self._server_chunk_start_idx_policy = 0
        self.t = 0
        self.Q.clear()
        self.Q.append(min(self.s_min, self.max_delay_policy - 1))
        # 清切换段起点: p→s 重启后第一条 cold-start 的 s_used_policy 为 -1(无前缀),
        # HTML 侧据此不画切换段;上一个 run 的 chunk 不会被误当切换来源。
        self._last_s_used_policy = -1
        # notify_all so the inference thread's wait_for() re-evaluates and
        # drops out of any pending inference-post-processing early.
        self.C.notify_all()
        print(
            f"[ws] IDLE ({reason}) gen={my_gen} — server /cancel_chunk={ok}; "
            f"cleared RTC state (A_cur, Q, t); "
            f"next 's' will cold-start a fresh chunk"
        )

    def _post_send_pause_guard_locked(self, send_gen: int, context: str) -> bool:
        """Restore the IDLE invariant if pause raced with ``/send_chunk``.

        Caller holds ``self.C`` and captured ``send_gen`` immediately before
        releasing it for the HTTP POST.  A generation change means the
        execution thread issued a cancel while the send was in flight.  The
        key-state check also covers cold-start, whose POST runs on the
        execution thread itself and therefore prevents that thread from
        observing the RUNNING→IDLE edge until the POST returns.

        The extra cancel happens *after* ``/send_chunk`` has returned, so it is
        ordered after any chunk that the racing request could have installed.
        Returns True when the caller must discard the send result/state.
        """
        invalidated = (
            self._cancel_gen != send_gen
            or not self.key_sm.is_running()
            or not self.running
        )
        if not invalidated:
            return False

        print(
            f"  [PauseGuard] {context} send invalidated "
            f"(send_gen={send_gen}, cancel_gen={self._cancel_gen}, "
            f"key_running={self.key_sm.is_running()}, runner_running={self.running}); "
            "issuing post-send cancel"
        )
        self._cancel_and_reset_locked(f"{context} send crossed pause")
        return True

    def _cold_start_locked(self) -> bool:
        """IDLE → RUNNING handler: fresh cold-start (no prefix).

        Caller must hold self.C. Releases the lock during robot obs +
        inference (both are network-bound); re-acquires before mutating
        shared state.

        Returns True on success, False if obs failed or user hit p mid-way.
        Idempotent-ish: called every RUNNING edge, so a p→s cycle produces
        a fresh chunk anchored at the current robot pose instead of resuming
        a stale one.
        """
        # A previous run may still be completing a send that crossed the pause
        # edge. Its post-send guard owns the right to issue the final cancel.
        # Do not let a new cold-start chunk race ahead of that cancel.
        announced_wait = False
        while self._send_in_flight and self.running and self.key_sm.is_running():
            if not announced_wait:
                print("[cold-start] waiting for previous in-flight send to quiesce")
                announced_wait = True
            self.C.wait(timeout=0.1)
        if not self.running or not self.key_sm.is_running():
            return False

        self.C.release()
        try:
            if not self.key_sm.is_running() or not self.running:
                return False
            # Cold-start has no prior chunk, so only an observation is needed.
            robot_obs = self.robot.get_observation(
                **self.obs_builder.fetch_kwargs()
            )
            if robot_obs is None:
                print("[cold-start] no observation available")
                return False
            t0 = time.monotonic()
            if self._record_enabled:
                # Cold-start has no prefix — clear stale reanchor captures so
                # this chunk's npz record shows None (no prev_norm / no new_norm),
                # but still captures model_out_norm from infer_chunk.
                self._reset_norm_record()
            chunks_init, A_init_raw, S_init_dict, body31_init = self.infer_chunk(
                robot_obs, prefix_raw=None, delay=0,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            print(f"[cold-start] exception: {e}")
            return False
        finally:
            self.C.acquire()

        # After re-acquiring: if user flipped IDLE while we were computing,
        # discard this chunk rather than pushing it. sonic won't get a chunk
        # this cycle; the next 's' triggers another cold-start.
        if not self.key_sm.is_running() or not self.running:
            print("[cold-start] IDLE flipped mid-inference, discarding chunk")
            return False

        self.action_horizon = int(A_init_raw.shape[1])
        self.wire_action_horizon = int(chunks_init["arm"].shape[0])
        self.A_cur_raw = A_init_raw
        self.A_cur_state_dict = S_init_dict
        self.A_cur_body31 = body31_init
        self.t = 0

        # Release for the POST — _send_chunk hits the wire.
        self._last_s_used_policy = -1  # cold-start 无前缀
        send_gen = self._cancel_gen
        self._send_in_flight = True
        self.C.release()
        try:
            send_result = self._send_chunk(
                chunks_init, s_used_local=None, is_init=True,
            )
        finally:
            self.C.acquire()
        try:
            invalidated = self._post_send_pause_guard_locked(send_gen, "cold-start")
        finally:
            self._send_in_flight = False
            if self._deferred_cancel_ack:
                self.key_sm.ack_cancel(True)
                self._deferred_cancel_ack = False
            self.C.notify_all()
        if invalidated:
            return False
        if send_result is None:
            self.key_sm.force_idle("cold-start chunk send failed")
            self._cancel_and_reset_locked("cold-start chunk send failed")
            return False
        print(
            f"[cold-start] chunk installed in {elapsed_ms:.0f}ms "
            f"(H_policy={self.action_horizon}, H_wire={self.wire_action_horizon}); "
            f"s_min(policy)={self.s_min}"
        )
        return True

    def _execution_loop(self):
        """Virtual-tick loop. Also monitors KeyStateMachine edges:
          RUNNING → IDLE: robot.cancel_chunk() + wipe local RTC state
                          (see _cancel_and_reset_locked docstring).
          IDLE → RUNNING: fresh cold-start (see _cold_start_locked).
        IDLE freezes self.t. Server progress is sampled only by the inference
        loop after fetching an observation; this loop never polls progress.
        """
        next_tick = time.monotonic()
        was_running = False  # tracks previous KeyStateMachine state
        while self.running:
            is_running = self.key_sm.is_running()

            # Edge detection — do this OUTSIDE the tick to avoid dropping a
            # frame while server-side cancel / cold-start HTTP is happening.
            if is_running != was_running:
                with self.C:
                    if is_running:
                        # IDLE → RUNNING: fresh cold-start.
                        _ok = self._cold_start_locked()
                    else:
                        # RUNNING → IDLE: cancel + wipe.
                        self._cancel_and_reset_locked("user pressed p")
                was_running = is_running

            if is_running:
                with self.C:
                    if self.A_cur_raw is not None and self.t < self.action_horizon:
                        self.t += 1
                    self.C.notify_all()

            next_tick += self.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def _inference_loop(self):
        with self.C:
            while self.running:
                if not self.C.wait_for(
                    lambda: (not self.running)
                    or (self.key_sm.is_running() and self.t >= self.s_min),
                    timeout=0.5,
                ):
                    continue
                if not self.running:
                    break

                s = self.t
                inference_gen = self._cancel_gen
                A_prev_raw_full = self.A_cur_raw.copy() if self.A_cur_raw is not None else None
                S_prev_dict = (
                    dict(self.A_cur_state_dict)
                    if self.A_cur_state_dict is not None
                    else None
                )
                # All RTC/model indices are policy-frame indices (pre-upsample).
                d_est_policy = max(self.Q) + self._delay_safety_margin
                if A_prev_raw_full is not None:
                    upper_policy = max(1, min(
                        self.action_horizon - s - 1,
                        self.max_delay_policy - 1,
                    ))
                    d_est_policy = max(1, min(d_est_policy, upper_policy))
                else:
                    d_est_policy = 0

                t_start = time.monotonic()
                inference_error = None
                self.C.release()
                try:
                    robot_obs = self.robot.get_observation_with_progress(
                        **self.obs_builder.fetch_kwargs()
                    )
                    observation_done = time.monotonic()
                    if robot_obs is None:
                        print("  [Inference] obs failed, skipping")
                        continue

                    server_progress = self._server_progress_from_snapshot(
                        robot_obs.get("chunk_progress")
                    )
                    if server_progress is not None:
                        # Single axis (policy==wire, both 30Hz); both tuple
                        # entries are the same value. s_used_local = s_used_policy.
                        _server_prog_wire, server_prog_policy = server_progress
                        P_obs = max(s, int(np.floor(server_prog_policy)))
                        s_used_policy = min(P_obs, self.action_horizon - 1)
                    else:
                        # Match the upstream dev RTC behavior: when an older
                        # RoboInterface/a3_server cannot provide live progress,
                        # continue from the local policy-frame counter.
                        s_used_policy = min(s, self.action_horizon - 1)

                    prefix_raw = None
                    d_used_policy = 0
                    if self._record_enabled:
                        self._reset_norm_record()
                    if A_prev_raw_full is not None and not self.no_rtc_prefix:
                        _, H_p, D_pad = A_prev_raw_full.shape
                        avail = max(0, H_p - s_used_policy)
                        if avail > 0:
                            prev_slice = A_prev_raw_full[0, s_used_policy:s_used_policy + avail, :]
                            fresh_obs, _ = self.obs_builder.build(
                                robot_obs, self.task_holder.current()
                            )
                            S_new_dict = _state_dict_from_obs(fresh_obs)
                            if self._record_enabled and self._pelvis_norm_sl is not None:
                                # reanchor 输入:上一 chunk raw 输出的 pelvis 列。
                                self._last_prev_norm_pelvis = np.asarray(
                                    prev_slice[:, self._pelvis_norm_sl], dtype=np.float32
                                ).copy()  # (avail, 6)
                                sp_state = S_prev_dict.get("pelvis_orient6d") \
                                    if S_prev_dict is not None else None
                                sn_state = S_new_dict.get("pelvis_orient6d")
                                if sp_state is not None:
                                    self._last_s_prev_pelvis = np.asarray(
                                        sp_state, dtype=np.float32
                                    ).reshape(-1)[-6:].copy()
                                if sn_state is not None:
                                    self._last_s_new_pelvis = np.asarray(
                                        sn_state, dtype=np.float32
                                    ).reshape(-1)[-6:].copy()
                            if S_prev_dict is not None and self.reanchor.enabled:
                                prev_slice = self.reanchor.reanchor_prefix_slice(
                                    prev_slice, s_used_policy, S_prev_dict, S_new_dict,
                                )
                            if self._record_enabled and self._pelvis_norm_sl is not None:
                                # reanchor 输出(=喂进 prefix_raw 的 pelvis 列)。
                                self._last_new_norm_pelvis = np.asarray(
                                    prev_slice[:, self._pelvis_norm_sl], dtype=np.float32
                                ).copy()  # (avail, 6)
                            prefix_raw = np.zeros_like(A_prev_raw_full)
                            prefix_raw[0, :avail, :] = prev_slice
                            upper_p = min(
                                self.max_delay_policy - 1,
                                avail,
                                self.policy_horizon - s_used_policy - 1,
                                self.policy_horizon - 1,
                            )
                            d_used_policy = max(1, min(d_est_policy, upper_p))

                    infer_stage_start = time.monotonic()
                    chunks_new, A_new_raw, S_new_dict, body31_new = self.infer_chunk(
                        robot_obs, prefix_raw=prefix_raw, delay=d_used_policy,
                    )
                    infer_stage_done = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    inference_error = exc
                finally:
                    self.C.acquire()

                if inference_error is not None:
                    message = (
                        "RTC observation/inference failed: "
                        f"{type(inference_error).__name__}: {inference_error}"
                    )
                    print(f"  [Inference] ERROR: {message}")
                    self.key_sm.force_idle(message)
                    self._cancel_and_reset_locked(message)
                    continue

                if chunks_new is None:
                    continue

                # Second RUNNING check after inference finishes — if the user
                # pressed p during inference, the runner cancelled the server
                # chunk in _execution_loop; installing our freshly-computed
                # chunk now would immediately restart /wbc/infer/reference_window,
                # bypassing the pause. Drop it. `_cancel_and_reset_locked`
                # has already cleared A_cur_raw/etc.; leave them cleared.
                if (
                    self._cancel_gen != inference_gen
                    or not self.key_sm.is_running()
                    or not self.running
                ):
                    print(
                        "  [Inference] run generation/key state changed during "
                        f"inference (start_gen={inference_gen}, "
                        f"cancel_gen={self._cancel_gen}) — discarding chunk "
                        "(no send, no state update)"
                    )
                    continue

                self.A_cur_raw = A_new_raw
                self.A_cur_state_dict = S_new_dict
                self.A_cur_body31 = body31_new

                if A_prev_raw_full is not None:
                    s_used_local = max(
                        0, int(s_used_policy) - int(self._server_chunk_start_idx_policy)
                    )
                    s_used_local = min(s_used_local, self.action_horizon - 1)
                else:
                    s_used_local = None

                self.action_horizon = int(A_new_raw.shape[1])
                self.wire_action_horizon = int(chunks_new["arm"].shape[0])

                send_gen = self._cancel_gen
                self._send_in_flight = True
                self._last_s_used_policy = int(s_used_policy)  # 前缀起点(policy 帧)
                timing_chunk_id = self._chunk_seq
                send_start = time.monotonic()
                self.C.release()
                try:
                    send_result = self._send_chunk(
                        chunks_new, s_used_local, is_init=False,
                    )
                finally:
                    self.C.acquire()
                send_done = time.monotonic()

                try:
                    invalidated = self._post_send_pause_guard_locked(send_gen, "RTC")
                finally:
                    self._send_in_flight = False
                    if self._deferred_cancel_ack:
                        self.key_sm.ack_cancel(True)
                        self._deferred_cancel_ack = False
                    self.C.notify_all()
                if invalidated:
                    continue

                if send_result is None:
                    self.key_sm.force_idle("RTC chunk send failed")
                    self._cancel_and_reset_locked("RTC chunk send failed")
                    continue

                actual_delay = max(0, min(int(send_result), self.action_horizon - 1))
                self.t = actual_delay
                self.Q.append(actual_delay)

                timing_done = time.monotonic()
                t_ms = (timing_done - t_start) * 1000
                under = max(0, actual_delay - d_used_policy)
                under_str = f", UNDERESTIMATE={under}(policy)" if under > 0 else ""
                print(
                    f"  [Inference] {t_ms:.0f}ms, s_virt(policy)={s}, "
                    f"s_used(policy)={s_used_policy}, d_est(policy)={d_est_policy}, "
                    f"d_used(policy)={d_used_policy}, d_actual(policy)={actual_delay}, "
                    f"prev@raw[{s_used_policy}:H_p]{under_str}"
                )
                observation_ms = (observation_done - t_start) * 1000.0
                rtc_prep_ms = (infer_stage_start - observation_done) * 1000.0
                infer_ms = (infer_stage_done - infer_stage_start) * 1000.0
                send_ms = (send_done - send_start) * 1000.0
                bookkeeping_ms = max(
                    0.0,
                    (send_start - infer_stage_done + timing_done - send_done)
                    * 1000.0,
                )
                infer_parts = self._last_infer_timing_ms
                print(
                    f"  [Timing] chunk_id={timing_chunk_id} total={t_ms:.1f}ms | "
                    f"observation={observation_ms:.1f}ms | "
                    f"rtc_prep={rtc_prep_ms:.1f}ms | "
                    f"infer={infer_ms:.1f}ms "
                    f"(obs_build={infer_parts.get('obs_build', 0.0):.1f}ms, "
                    f"policy={infer_parts.get('policy', 0.0):.1f}ms, "
                    f"decode={infer_parts.get('decode', 0.0):.1f}ms, "
                    f"other={infer_parts.get('other', 0.0):.1f}ms) | "
                    f"send={send_ms:.1f}ms | "
                    f"bookkeeping={bookkeeping_ms:.1f}ms"
                )

    def run(self):
        self.running = True
        print(
            f"[RTC-Chunk-Train-A3] H_policy={self.policy_horizon}, "
            f"max_delay(policy)={self.max_delay_policy}, "
            f"s_min(policy)={self.s_min}, "
            f"policy_fps={self.policy_output_fps}Hz→wire_fps={self.wire_fps}Hz, "
            f"transition={'adaptive' if self._server_adaptive_transition else 'fixed'}, "
            f"emit_mode=reference_window, delta_reanchor={self.reanchor.enabled}"
        )
        _print_start_prompt()
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()
        try:
            self._execution_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            infer_thread.join(timeout=5.0)
            # 先把 chunk 记录落盘,再做 server 取消 —— 保证"先存好再退出"。
            self._dump_chunks()
            # Best-effort cancel on shutdown so the robot doesn't keep tracking
            # a stale chunk after the runner exits.
            try:
                self.robot.cancel_chunk()
            except Exception:
                pass


# ============================================================================
# Standard (non-RTC) sync mode — baseline comparison
# ============================================================================


def run_standard(
    policy_client: PolicyClient,
    robot: A3RobotInterface,
    obs_builder: A3ObsBuilder,
    decoder: A3ActionDecoder,
    task_holder: TaskHolder,
    key_sm: KeyStateMachine,
    args,
):
    """Sync mode: obs → infer → send 30Hz chunk (non-blocking install) → sleep
    chunk_duration → repeat. No RTC. a3_server does the 30→50Hz interpolation.
    Gated on key_sm.is_running() (press 's' to start)."""
    _print_start_prompt()
    try:
        while True:
            if not key_sm.is_running():
                time.sleep(0.1)
                continue
            robot_obs = robot.get_observation(**obs_builder.fetch_kwargs())
            obs, body31 = obs_builder.build(robot_obs, task_holder.current())
            t0 = time.time()
            action_dict, _ = policy_client.get_action(obs)
            infer_ms = (time.time() - t0) * 1000
            chunks = decoder.decode(
                action_dict, body31, obs_builder.hand_position_rad(robot_obs)
            )
            H = chunks["arm"].shape[0]
            if args.action_horizon is not None:
                H = min(H, args.action_horizon)
                chunks = {k: v[:H] for k, v in chunks.items()}
            payload = {
                "leg": chunks["leg"],
                "waist": chunks["waist"],
                "arm": chunks["arm"],
                "pelvis_quat_wxyz": chunks["pelvis_quat_wxyz"],
            }
            if "hand" in chunks:
                payload["hand"] = chunks["hand"]
                payload["hand_value"] = decoder.hand_value
            robot.step_chunk(
                payload,
                chunk_fps=args.policy_output_fps,
                wait=False,
                chunk_id=-1,
                emit_mode="reference_window",
            )
            duration = (H - 1) / args.policy_output_fps
            time.sleep(max(0.0, duration - infer_ms / 1000.0))
            print(f"  [Standard] infer={infer_ms:.0f}ms, chunk={H} ({duration*1000:.0f}ms)")
    except KeyboardInterrupt:
        print("\nStopping...")


# ============================================================================
# Entry point
# ============================================================================


# ============================================================================
# UPPER-BODY (mc per-part topics) pipeline — auto-selected for the
# examples/A3 dex / gripper configs (hand/gripper + arm + waist + waist_height).
# ============================================================================


# A3_WAIST_DEBUG=1 prints measured-vs-commanded waist each chunk.
_WAIST_DEBUG = os.environ.get("A3_WAIST_DEBUG", "") not in ("", "0")


STATE_KEY_TO_JOINT: dict[str, str] = {
    "hand":    "hand",
    "gripper": "hand",
    "arm":     "arm",
    "waist":   "waist",
}


# Upper-body 复位臂角 (14D: 左臂 7 + 右臂 7)。A2 的 robot.reset() 用预设 pose;
# A3 upper-body 这里用一个固定关节角,开始/结束各插值复位一次
# (见 reset_upper_body_arm)。改姿势直接改这里。
UPPER_BODY_RESET_ARM_POS: list[float] = [
    0.4,  0.25,  0.08, -0.6, 0.0, 0.0, 0.0,   # 左臂
    0.4, -0.25, -0.08, -0.6, 0.0, 0.0, 0.0,   # 右臂
]

# 复位腰部目标 (顺序 [yaw, roll, pitch, height], 与 send_waist 位置参数一致)。
# 全 0 = 直立 + height 顶端。设 None 则复位时不动腰。
UPPER_BODY_RESET_WAIST: "list[float] | None" = [0.0, 0.0, 0.0, 0.0]


class A3UpperBodyObsBuilder:
    """Turn a robot_obs dict into the GR00T obs the policy expects, driven by
    the checkpoint's modality schema (fetched via ``get_rtc_metadata``).

    For the two upper-body configs the state is hand/gripper + arm + waist,
    each read verbatim from ``robot_obs["joints"][<group>]["position"]``:
      - hand  (20D, radians)  — needs get_observation(hand_rad=True)
      - gripper (2D, raw)      — actuator counts
      - arm   (14D, radians)
      - waist (3D)
    Cameras come from the checkpoint's video_keys mapped through
    VIDEO_CAMERA_MAP.
    """

    def __init__(self, metadata: dict, video_camera_map: dict[str, str], hand_kind: str):
        self.video_camera_map = dict(video_camera_map)
        self.hand_kind = hand_kind
        self.video_keys = list(metadata.get("video_keys") or [])
        self.state_keys = list(metadata["state_keys"])
        self.ref_only_keys = list(metadata.get("reference_only_keys") or [])
        self.language_keys = list(
            metadata.get("language_keys") or ["annotation.human.task_description"]
        )
        self.state_dims = dict(metadata.get("state_dims") or {})

        # State keys the loader materialises (encoder input + reference-only),
        # preserving the config's own order for repeatability.
        seen: set[str] = set()
        self._all_state: list[str] = []
        for k in self.state_keys + self.ref_only_keys:
            if k not in seen:
                seen.add(k)
                self._all_state.append(k)

        for k in self.video_keys:
            if k not in self.video_camera_map:
                raise RuntimeError(
                    f"video key {k!r} not in VIDEO_CAMERA_MAP; add it to "
                    "VIDEO_CAMERA_MAP_DEFAULT."
                )
        for k in self._all_state:
            if k not in STATE_KEY_TO_JOINT:
                raise RuntimeError(
                    f"state key {k!r} has no joint mapping. This script targets "
                    f"the upper-body configs (hand/gripper + arm + waist); add "
                    f"{k!r} to STATE_KEY_TO_JOINT if you extended the recipe."
                )

        self.cameras_needed = [self.video_camera_map[k] for k in self.video_keys]
        # hand state is activejointpos radians for the dex hand; the gripper is
        # raw actuator counts (hand_rad is a no-op there, see A3RobotInterface).
        self.needs_imu = False

    def fetch_kwargs(self) -> dict:
        """Kwargs for A3RobotInterface.get_observation — only pull the cameras
        the checkpoint needs, request hand radians for the dex hand."""
        return {
            "hand_rad": self.hand_kind == "hand",
            "cameras": list(self.cameras_needed),
            "include_imu": False,
        }

    def hand_position_rad(self, robot_obs: dict) -> "np.ndarray | None":
        """Measured 20D hand position in radians, or None.

        The reference RTC runner hands this to the decoder; for the dex hand
        fetch_kwargs already requested radians, so it is a straight read.
        """
        if self.hand_kind != "hand":
            return None
        pos = ((robot_obs.get("joints") or {}).get("hand") or {}).get("position")
        if not pos or len(pos) != 20:
            return None
        return np.asarray(pos, dtype=np.float32)

    def _joint_state(self, joints: dict, key: str) -> np.ndarray:
        jkey = STATE_KEY_TO_JOINT[key]
        dim = int(self.state_dims.get(key, 0)) or None
        pos = (joints.get(jkey) or {}).get("position") or []
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if dim is not None:
            if arr.size < dim:
                arr = np.concatenate([arr, np.zeros(dim - arr.size, dtype=np.float32)])
            else:
                arr = arr[:dim]
        return arr

    def build(self, robot_obs: dict, task: str) -> dict:
        """robot_obs → gr00t_obs dict (video / state / language)."""
        joints = robot_obs.get("joints") or {}

        video = {}
        for k in self.video_keys:
            cam_name = self.video_camera_map[k]
            frame = _preprocess_image(robot_obs.get(cam_name))
            video[k] = frame[None, None, ...]  # (1, T=1, H, W, C)

        state = {}
        for k in self._all_state:
            arr = self._joint_state(joints, k)
            state[k] = np.asarray(arr, dtype=np.float32).reshape(1, 1, -1)

        language = {self.language_keys[0]: [[task]]} if self.language_keys else {}
        return {"video": video, "state": state, "language": language}


class A3UpperBodyActionDecoder:
    """Per-key action chunks → per-key ABSOLUTE chunk dict.

    Keys the checkpoint declares RELATIVE (``action_reps`` from
    get_rtc_metadata) come back as offsets from a reference state, so that
    state must be added to get a joint command. Sending a RELATIVE waist chunk
    as if it were absolute nudges the column the same way on every chunk
    install — on the robot that reads as the waist creeping in one direction
    with a jerk at each chunk boundary.

    Reference state per key is ``action_state_key`` when set, else the state
    group of the same name.

    Expected action keys (one of):
        {"hand"|"gripper", "arm", "waist", "waist_height"}
    """

    HAND_KEYS = ("hand", "gripper")

    def __init__(self, metadata: dict):
        self.action_keys = list(metadata["action_keys"])
        keys = set(self.action_keys)
        hand_key = next((k for k in self.HAND_KEYS if k in keys), None)
        if hand_key is None or "arm" not in keys or "waist" not in keys:
            raise RuntimeError(
                f"Unsupported action layout: {self.action_keys}. This script "
                f"expects an upper-body config: one of {self.HAND_KEYS} + 'arm' "
                f"+ 'waist' (+ optional 'waist_height')."
            )
        self.hand_key = hand_key
        # "rad" for the 20D dex hand (converted to actuator counts downstream),
        # "raw" for the 2D gripper. The reference runner reads this off the
        # decoder rather than re-deriving it.
        self.hand_value = "rad" if hand_key == "hand" else "raw"
        self.kind = f"upper_body:{hand_key}"
        self.has_waist_height = "waist_height" in keys
        reps = dict(metadata.get("action_reps") or {})
        # MEASURED on the robot: gr00t's get_action already resolves RELATIVE
        # keys against the observation state, so everything reaching us is
        # absolute. Adding the state again doubles the offset — tried it, the
        # waist swung noticeably harder. modality.json / action_reps describe
        # the TRAINING representation, not what get_action hands back.
        # A3_RELATIVE_KEYS=waist[,arm] re-enables the add-back for a checkpoint
        # whose policy really does return offsets.
        opt_in = {k.strip() for k in os.environ.get("A3_RELATIVE_KEYS", "").split(",")
                  if k.strip()}
        self.relative_keys = [k for k in self.action_keys if k in opt_in]
        self.state_key_for = dict(metadata.get("action_state_key") or {})
        print(f"[decoder] action_reps={reps} (training-time repr)", flush=True)
        print(f"[decoder] treating policy output as absolute; state added back "
              f"for: {self.relative_keys or 'none'}", flush=True)

    def _ref_state(self, obs: dict, key: str, dim: int) -> np.ndarray:
        """Reference state vector for a RELATIVE key, shaped (dim,)."""
        state = (obs or {}).get("state") or {}
        name = self.state_key_for.get(key) or key
        arr = state.get(name)
        if arr is None:
            raise RuntimeError(
                f"action key {key!r} is RELATIVE but the observation has no "
                f"state {name!r} to anchor it (state keys: {sorted(state)})."
            )
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.size < dim:
            raise RuntimeError(
                f"state {name!r} is {flat.size}D, need {dim}D to anchor "
                f"action {key!r}."
            )
        return flat[:dim]

    def decode(self, action_dict: dict, robot_hand20_rad=None) -> dict:
        """{action_key: (1, H, D)} → {action_key: (H, D)} float32.

        Signature matches the reference client. This checkpoint emits a 20D
        hand directly, so ``robot_hand20_rad`` (which the reference feeds to
        decode_hand_action_chunk for hand_opening-style mappings) is unused
        here; it is accepted so the runner code stays identical.

        Output is passed through as-is: gr00t's get_action has already
        resolved RELATIVE keys against the observation state.
        """
        out = {}
        for k in self.action_keys:
            arr = np.asarray(action_dict[k][0], dtype=np.float32)
            if k in self.relative_keys:
                ref = self._opt_in_ref_state(k, arr.shape[-1])
                if ref is not None:
                    arr = arr + ref[None, :]
            out[k] = arr
        return out

    def set_reference_state(self, obs: dict) -> None:
        """Stash the observation for the A3_RELATIVE_KEYS opt-in path."""
        self._last_obs = obs

    def _opt_in_ref_state(self, key: str, dim: int):
        obs = getattr(self, "_last_obs", None)
        if obs is None:
            return None
        try:
            return self._ref_state(obs, key, dim)
        except RuntimeError:
            return None


def build_upper_chunk_dict(
    chunk: dict,
    hand_key: str,
    hand_value: str,
    use_model_waist_height: bool = False,
    waist_target_idx: int | None = None,
) -> dict:
    """Per-key absolute chunk {key: (H, D)} -> A3RobotInterface.step_chunk() dict.

    Sends the WHOLE arm/hand chunk to the mc topics (a3_server paces them on its
    150Hz interpolator via /send_chunk legacy path):
        out["arm"]        (H, 14)     -> mc /motion/control/arm_joint_command
        out["hand"]       (H, 20/2)   -> mc hand / gripper joint command
        out["hand_value"] "rad"|"raw"

    WAIST: a3_server routes any /send_chunk payload containing ``waist`` (or
    ``s_used_local``) to the WHOLE-BODY branch (which needs leg+pelvis), so waist
    cannot ride the arm/hand chunk. A3RobotInterface.step_chunk instead POPS the
    ``waist`` key and paces it via /send_waist. To avoid overlapping per-row
    waist threads across successive RTC chunks (jitter), we default to a SINGLE
    waist target per chunk:
        waist_target_idx=None -> full (H, 4) trajectory (interface threads it
                                 per row at chunk_fps; only safe when chunks
                                 don't overlap, e.g. --mode standard full play)
        waist_target_idx=i    -> single 4D target = waist frame i (1D payload;
                                 interface sends it once)
    The waist 4th dim (pelvis height h) is padded 0 unless
    ``use_model_waist_height`` and the model emitted a ``waist_height`` group.
    """
    out: dict = {"hand_value": hand_value}
    if "arm" in chunk:
        out["arm"] = np.asarray(chunk["arm"], dtype=np.float32)          # (H, 14)
    if hand_key in chunk:
        out["hand"] = np.asarray(chunk[hand_key], dtype=np.float32)      # (H, 20/2)
    if "waist" in chunk:
        w = np.asarray(chunk["waist"], dtype=np.float32)                 # (H, 3)
        H = w.shape[0]
        if use_model_waist_height and "waist_height" in chunk:
            h = np.asarray(chunk["waist_height"], dtype=np.float32).reshape(H, 1)
        else:
            h = np.zeros((H, 1), dtype=np.float32)
        waist4 = np.concatenate([w[:, :3], h], axis=1)                   # (H, 4)
        if waist_target_idx is None:
            out["waist"] = waist4                                        # 2D -> threaded
        else:
            i = int(min(max(0, waist_target_idx), H - 1))
            out["waist"] = waist4[i]                                     # 1D -> single send
    return out


def load_dataset_task(dataset_path: str) -> Optional[str]:
    """Read the language prompt the checkpoint was trained with.

    The policy is language-conditioned: feeding a prompt that differs from the
    training task makes it emit unrelated (and on a real robot, unsafe) motion.
    Returns None when the dataset has no tasks.jsonl.
    """
    tasks_path = Path(dataset_path) / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return None
    for line in tasks_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        task = json.loads(line).get("task")
        if task:
            return str(task)
    return None


def load_dataset_initial_pose(dataset_path: str, episode_index: int = 0) -> dict[str, np.ndarray]:
    """Load the model-controlled upper-body pose from a LeRobot frame."""
    dataset = Path(dataset_path)
    info_path = dataset / "meta" / "info.json"
    modality_path = dataset / "meta" / "modality.json"
    if not info_path.is_file() or not modality_path.is_file():
        raise FileNotFoundError(
            f"dataset must contain meta/info.json and meta/modality.json: {dataset}"
        )
    info = json.loads(info_path.read_text())
    modality = json.loads(modality_path.read_text())
    state_meta = modality.get("state", {})
    required = {"hand", "arm", "waist"}
    if not required.issubset(state_meta):
        raise RuntimeError(f"dataset modality state is missing {required - set(state_meta)}")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        helper = os.environ.get(
            "A3_PARQUET_PYTHON",
            "/agibot/fengtianli/a3_lerobot/.venv/bin/python",
        )
        script = (
            "import json, sys, pyarrow.parquet as pq; "
            "t=pq.read_table(sys.argv[1], columns=['observation.state']); "
            "print(json.dumps(t['observation.state'][0].as_py()))"
        )
        try:
            out = subprocess.check_output(
                [helper, "-c", script, str(dataset / info["data_path"].format(
                    episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
                    episode_index=episode_index,
                ))],
                text=True,
                stderr=subprocess.STDOUT,
            )
            raw = np.asarray(json.loads(out), dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                "reading --dataset needs pyarrow or a valid A3_PARQUET_PYTHON helper"
            ) from exc
    else:
        chunk_size = int(info.get("chunks_size", 1000))
        chunk = episode_index // chunk_size
        pattern = info["data_path"]
        parquet_rel = pattern.format(episode_chunk=chunk, episode_index=episode_index)
        parquet_path = dataset / parquet_rel
        if not parquet_path.is_file():
            raise FileNotFoundError(f"episode parquet not found: {parquet_path}")
        table = pq.read_table(parquet_path, columns=["observation.state"])
        raw = np.asarray(table["observation.state"][0].as_py(), dtype=np.float32).reshape(-1)
    pose = {}
    for key in ("hand", "arm", "waist"):
        start = int(state_meta[key]["start"])
        end = int(state_meta[key]["end"])
        pose[key] = raw[start:end].copy()
    expected = {"hand": 20, "arm": 14, "waist": 3}
    bad = {key: value.size for key, value in pose.items() if value.size != expected[key]}
    if bad:
        raise RuntimeError(f"unexpected dataset pose dimensions: {bad}")
    print(
        f"[edge] dataset pose episode={episode_index} frame=0 "
        f"state_dim={raw.size} slices=hand[{state_meta['hand']['start']}:{state_meta['hand']['end']}], "
        f"arm[{state_meta['arm']['start']}:{state_meta['arm']['end']}], "
        f"waist[{state_meta['waist']['start']}:{state_meta['waist']['end']}]",
        flush=True,
    )
    for key, value in pose.items():
        print(
            f"[edge] dataset_pose.{key} shape={value.shape} "
            f"min={value.min():.6f} max={value.max():.6f}",
            flush=True,
        )
    return pose


def reset_upper_body_arm(
    robot,
    target_arm: list[float],
    target_waist: "list[float] | None" = None,
    target_hand: "list[float] | None" = None,
    reset_hand: bool = True,
    hand_pose: str = "open",
    from_last_cmd: bool = False,
    steps: int = 100,
    settle_ms: float = 200.0,
    label: str = "reset",
) -> bool:
    """把 upper-body 的 **手臂 + 手 + 腰** 复位到固定位姿,镜像 A2 infer_a2_rtc.py
    ``robot.reset()``(A2 复位臂+手,A3 这里再加腰)。开始与结束各调一次。
    复位失败只打印不抛,绝不 crash 主流程。

        target_arm     : 14D 关节角 (左臂 7 + 右臂 7)
        target_waist   : [yaw, roll, pitch, height];None 则不动腰
        reset_hand     : True 时把手/夹爪复位到 ``hand_pose`` ("open"/"fist"),
                         维度由 robot.hand_kind 决定 (灵巧手 20D / 夹爪 2D)
        from_last_cmd  : 过渡起点策略 (见下)
        steps          : 过渡用多少个 150Hz tick (≈ steps/150 秒)

    **过渡起点 —— 从"上一条 cmd"起,而不是从 measured state 起**:
    a3_server 的 150Hz interp 环在装 chunk 时,arm/hand/waist 都从各自的
    ``_*_current``(= 上一条 cmd 的保持位)平滑插到 chunk 首帧
    (见 interp_publisher._install_*_chunk_locked)。所以:
      • from_last_cmd=True (跑过 RTC, 有 cmd, 如 reset-end): 每组只发 **单帧目标**,
        chunk_fps=150/steps 使 server 用 steps 个 tick 从 ``_*_current``(上一条 cmd,
        被 cancel 时停住的保持位) 平滑到目标。客户端不需要知道上一条 cmd 是多少,
        也不会因 measured state 的跟踪误差先倒退一下。
      • from_last_cmd=False (还没发过 cmd, 如 reset-start): 读 measured state,
        构造 steps 帧 state→目标 轨迹,chunk_fps=150。即使 server ``_*_current``
        为 None (从没装过 chunk), 也会从 chunk[0]=state 起,安全。

    整段走一个 upper_body chunk 交给 server 而不是客户端逐帧直发, 是因为
    cancel_chunk 之后 server 仍把 ``_*_target`` 保持在模型最后一帧持续 publish;
    客户端自己 send_* 会和它抢同一组 mc topic (复位往回动一下又被拽回, 腰最明显)。
    把复位轨迹装进 server, 它才会把 ``_*_target`` 更新到复位位并保持。
    """
    TARGET_FPS = 150.0                          # a3_server interp 环频率
    N = int(max(2, steps))                      # 过渡 tick 数

    target = np.asarray(target_arm, dtype=np.float64).reshape(-1)
    if target.size != 14:
        print(f"[{label}] target_arm 必须是 14D, 收到 {target.size}D — 跳过复位")
        return False

    # 等 mc 守护起来、arm 关节可读(与 A2 reset 的等待逻辑一致)。
    js = None
    for _ in range(50):
        js = robot.get_joint_states()
        if js and js.get("arm") and js["arm"].get("position"):
            break
        time.sleep(0.1)
    else:
        print(f"[{label}] 未获取到 arm 关节状态 — 跳过复位")
        return False

    # _grp(current, target, D) → 该组要装进 chunk 的帧:
    #   from_last_cmd: 单帧目标 (1,D), 让 server 从 _*_current(上一条cmd) 平滑;
    #   否则:          (N,D) 从 measured current 插到 target。
    if from_last_cmd:
        chunk_fps = TARGET_FPS / N
        def _grp(current, tgt, D):
            return np.asarray(tgt, dtype=np.float64).reshape(1, D).astype(np.float32)
    else:
        chunk_fps = TARGET_FPS
        _alphas = np.linspace(0.0, 1.0, N).reshape(N, 1)
        def _grp(current, tgt, D):
            c = np.asarray(current, dtype=np.float64).reshape(1, D)
            t = np.asarray(tgt, dtype=np.float64).reshape(1, D)
            return ((1.0 - _alphas) * c + _alphas * t).astype(np.float32)

    # ---- 手臂 (14D) ----
    arm_current = np.asarray(js["arm"]["position"], dtype=np.float64).reshape(-1)
    if arm_current.size < 14:
        arm_current = np.concatenate([arm_current, np.zeros(14 - arm_current.size)])
    arm_current = arm_current[:14]
    chunk: dict = {"arm": _grp(arm_current, target, 14)}

    # ---- 腰 (4D [yaw,roll,pitch,height]) ----
    waist_target = None
    if target_waist is not None:
        wt = np.asarray(target_waist, dtype=np.float64).reshape(-1)
        if wt.size == 3:
            wt = np.concatenate([wt, [0.0]])          # 默认 height=0
        if wt.size != 4:
            print(f"[{label}] target_waist 需为 3D/4D [yaw,roll,pitch,(height)], "
                  f"收到 {wt.size}D — 跳过腰部复位")
        else:
            waist_target = wt
            wpos = (js.get("waist") or {}).get("position") or []
            wc = np.asarray(wpos, dtype=np.float64).reshape(-1)
            # state 只有 yaw/roll/pitch;height 读不到,从 0 起 (仅 from state 分支用)。
            if wc.size >= 3:
                waist_current = np.array([wc[0], wc[1], wc[2], 0.0], dtype=np.float64)
            else:
                waist_current = wt.copy()
            chunk["waist"] = _grp(waist_current, wt, 4)

    # ---- 手/夹爪 (hand_dim: 灵巧手 20D / 夹爪 2D) ----
    # target_hand comes from the dataset in CHECKPOINT state units, i.e.
    # radians for the 20D dex hand. get_joint_states() reports raw actuator
    # counts (0-4096), so the interpolation start must be converted to radians
    # too — mixing the two and tagging the chunk "rad" sends a ~4096-radian
    # first frame, which clamps the hand shut.
    hand_desc = ""
    if target_hand is not None:
        hand_target = np.asarray(target_hand, dtype=np.float64).reshape(-1)
        hdim = int(getattr(robot, "hand_dim", hand_target.size))
        if hand_target.size == hdim and hdim > 0:
            hpos = (js.get("hand") or {}).get("position") or []
            hc = np.asarray(hpos, dtype=np.float64).reshape(-1)
            if hdim == 20 and hc.size == 20:
                try:
                    from config import OMNIHAND_LEFT, OMNIHAND_RIGHT
                    hc = np.asarray(
                        list(OMNIHAND_LEFT.actuator_to_radians(list(hc[:10])))
                        + list(OMNIHAND_RIGHT.actuator_to_radians(list(hc[10:]))),
                        dtype=np.float64,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[{label}] hand actuator->rad 转换失败, 从目标位起插: {e}")
                    hc = hand_target.copy()
            hand_current = hc if hc.size == hdim else hand_target.copy()
            chunk["hand"] = _grp(hand_current, hand_target, hdim)
            chunk["hand_value"] = "rad" if hdim == 20 else "raw"
            hand_desc = f" hand=dataset_frame0({hdim}D, rad)"
        else:
            print(f"[{label}] dataset target_hand 维度={hand_target.size} "
                  f"与 hand_dim={hdim} 不符 — 跳过手部复位")
    elif reset_hand:
        try:
            from config import VLA_HAND_INIT_POS_2  # 灵巧手 open 的 20D fallback
        except Exception:
            VLA_HAND_INIT_POS_2 = None
        try:
            hand_target, hand_label = robot._pick_reset_hand_pose(hand_pose, VLA_HAND_INIT_POS_2)
            hand_target = np.asarray(hand_target, dtype=np.float64).reshape(-1)
            hdim = int(getattr(robot, "hand_dim", hand_target.size))
            if hand_target.size == hdim and hdim > 0:
                hpos = (js.get("hand") or {}).get("position") or []
                hc = np.asarray(hpos, dtype=np.float64).reshape(-1)
                hand_current = hc if hc.size == hdim else hand_target.copy()
                chunk["hand"] = _grp(hand_current, hand_target, hdim)
                chunk["hand_value"] = "raw"
                hand_desc = f" hand={hand_label}({hdim}D)"
            else:
                print(f"[{label}] hand pose 维度={hand_target.size} 与 hand_dim={hdim} "
                      f"不符 — 跳过手部复位")
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] 取 hand 复位位失败(跳过手部): {e}")

    # ---- 一次性把 arm(+hand+waist) 复位轨迹装进 server, wait=True 阻塞到执行完 ----
    win_ms = N / TARGET_FPS * 1000.0
    print(
        f"[{label}] 复位 (start={'last_cmd' if from_last_cmd else 'state'}, "
        f"{N} ticks≈{win_ms/1000.0:.2f}s) arm->{target.tolist()}"
        + (f" waist->{waist_target.tolist()}" if waist_target is not None else "")
        + hand_desc
    )
    try:
        resp = robot.step_chunk(
            chunk, chunk_fps=float(chunk_fps), wait=True, mode="upper_body",
            settle_ms=settle_ms, timeout_ms=max(3000.0, win_ms + 2000.0),
        )
        ok = (not isinstance(resp, dict)) or resp.get("ok", True)
        if not ok:
            print(f"[{label}] 复位 chunk 未被 server 确认: {resp}")
            return False
    except Exception as e:  # noqa: BLE001 — 复位失败不该 crash 主流程
        print(f"[{label}] 复位 chunk 下发异常(忽略): {e}")
        return False
    print(f"[{label}] 复位完成")
    return True


class AsyncRTCUpperBodyRunner:
    """Train-time RTC over a ZMQ PolicyClient, CHUNK send to the three A3 mc
    topics — mirrors A2/infer_a2_rtc.py::AsyncRTCChunkTrainRunner.

    The client virtual-ticks ``self.t`` at ``chunk_fps`` (it does NOT call
    robot.step()); a3_server plays the whole arm/hand chunk on its 150Hz
    interpolator (queued via robot.step_chunk). When ``self.t`` crosses
    ``s_min`` the inference thread re-infers with the previous chunk's
    reanchored tail as the model prefix, swaps ``self.A_cur`` and sends the new
    chunk.

    a3_server provides SERVER-ATOMIC swap on the upper-body chunk path
    (interp_publisher.swap_upper_chunk_atomic, reached via
    step_chunk(mode="upper_body")): the client sends the full chunk +
    ``s_used_local`` (its obs-time position in the current server chunk), the
    server reads its live arm played index at install time, computes
    ``actual_delay`` = frames elapsed during inference, and slices arm/hand/waist
    by it. The virtual tick resyncs to ``self.t = actual_delay`` after the swap
    (same protocol as A2's AsyncRTCChunkTrainRunner). If the a3_server predates
    the upper_body route it returns actual_delay=0 → this degrades to
    soft-continuous (full chunk from frame 0, RTC prefix provides continuity).

    Gated by KeyStateMachine: IDLE cancels the server chunk (robot holds); the
    IDLE->RUNNING edge cold-starts a fresh chunk (no prefix).

    PORTED VERBATIM from gr00t/eval/real_robot/A3/infer_a3_rtc_zmq.py.
    The only edits are: (a) initial_pose/next_mode so the operator can reset
    and switch modes from the key handler, (b) get_chunk_progress read
    separately because this transport's get_observation does not bundle it.
    Everything else — the cancel-generation guards, in-flight send tracking,
    cold-start IDLE checks — is the reference logic and must stay that way:
    they are what stops a chunk that was already in flight from un-pausing the
    robot after 'p'.
    """

    def __init__(self, policy_client, robot, obs_builder, decoder, reanchor,
                 task_holder, key_sm, args, metadata, initial_pose=None):
        self.policy = policy_client
        self.robot = robot
        self.obs_builder = obs_builder
        self.decoder = decoder
        self.reanchor = reanchor
        self.task_holder = task_holder
        self.key_sm = key_sm
        self.args = args
        self.metadata = metadata
        self.initial_pose = initial_pose
        self.next_mode = None   # set to "standard" when the operator hits 'm'

        self.action_horizon = int(metadata["action_horizon"])
        self.max_delay = int(metadata["rtc_max_delay"])
        if self.max_delay <= 0:
            raise RuntimeError(
                "Server reports rtc_max_delay=0 — this checkpoint was not "
                "trained with train-time RTC. Use --mode standard, or pass "
                "--rtc_max_delay_override <int> for a diagnostic dry-run."
            )
        self.s_min = int(args.exec_steps)
        self.chunk_fps = float(args.fps)
        self.interval = 1.0 / self.chunk_fps

        self.action_keys = list(metadata["action_keys"])
        self.action_dims = {k: int(metadata["action_dims"][k]) for k in self.action_keys}
        self.hand_key = decoder.hand_key
        self.hand_value = decoder.hand_value
        self.use_model_waist_height = bool(getattr(args, "use_model_waist_height", False))
        self.waist_full_traj = bool(getattr(args, "waist_full_traj", False))
        self._delay_safety_margin = int(args.rtc_delay_margin)
        self.Q = deque(maxlen=4)

        # shared state guarded by the condition variable C
        self.C = threading.Condition()
        self.A_cur = None            # {key: (H, D)} absolute chunk (server is playing)
        self.A_cur_raw = None        # (1, H, D_pad) normalized (prefix source)
        self.A_cur_state = None      # {state_key: (D,)} at inference time
        self.t = 0
        # server-atomic swap bookkeeping: where (in the FULL chunk's frame space)
        # the server started playing the current chunk (= actual_delay it sliced).
        self._server_chunk_start_idx = 0
        self._chunk_seq = 0
        self.running = False
        # Generation/send guards prevent a pause from being undone by an
        # in-flight chunk POST, including the cold-start POST.
        self._cancel_gen = 0
        self._send_in_flight = False
        self._deferred_cancel_ack = False

    def _server_infer(self, obs, prefix_raw, delay):
        options = None
        if prefix_raw is not None and delay > 0:
            options = {
                "rtc_mode": "train_time",
                "rtc_delay": int(min(delay, self.max_delay - 1)),
                "action_prefix": prefix_raw,
            }
        return self.policy.get_action(obs, options=options)

    def infer_chunk(self, robot_obs, prefix_raw, delay):
        obs = self.obs_builder.build(robot_obs, self.task_holder.current())
        state = _state_dict_from_obs(obs)
        action_dict, info = self._server_infer(obs, prefix_raw, delay)
        raw = info.get("action_pred_normalized") if isinstance(info, dict) else None
        if raw is not None:
            raw = np.asarray(raw, dtype=np.float32)
            if raw.ndim == 2:
                raw = raw[None, ...]
        chunk = self.decoder.decode(
            action_dict, self.obs_builder.hand_position_rad(robot_obs)
        )
        H = chunk[self.hand_key].shape[0]
        if self.args.action_horizon is not None:
            H = min(H, int(self.args.action_horizon))
            chunk = {k: v[:H] for k, v in chunk.items()}
            if raw is not None:
                raw = raw[:, :H, :]
        return chunk, raw, state

    def _send_chunk(self, chunk, s_used_local, is_init=False):
        """Send the FULL upper-body chunk (arm/hand + server-paced waist) via
        the mode="upper_body" server-atomic path. The server reads its live arm
        played index at install time, computes actual_delay = frames elapsed
        during inference, slices arm/hand/waist by it, and returns it.

        Returns actual_delay (int), or None on send failure. Degrades to 0
        (soft-continuous) automatically on a server that predates this route.
        """
        cd = build_upper_chunk_dict(
            chunk, self.hand_key, self.hand_value,
            self.use_model_waist_height, waist_target_idx=None,  # full traj; server paces
        )
        chunk_id = self._chunk_seq
        self._chunk_seq += 1
        sul = None if is_init else s_used_local
        try:
            resp = self.robot.step_chunk(
                cd, chunk_fps=self.chunk_fps, wait=False,
                mode="upper_body", s_used_local=sul, chunk_id=chunk_id,
            )
        except Exception as e:
            print(f"  [ChunkSend] step_chunk failed: {e}")
            return None
        actual_delay = 0
        if isinstance(resp, dict):
            if not resp.get("ok", True):
                print(f"  [ChunkSend] server rejected chunk_id={chunk_id}: {resp}")
                return None
            actual_delay = int(resp.get("actual_delay", 0) or 0)
        actual_delay = max(0, min(actual_delay, max(0, self.action_horizon - 1)))
        self._server_chunk_start_idx = actual_delay
        return actual_delay

    def _post_send_pause_guard_locked(self, send_gen: int, context: str) -> bool:
        """Cancel again if IDLE raced with a chunk POST.

        Caller holds ``self.C``. The compensating cancel is ordered after the
        POST response, so a late server install cannot revive a paused run.
        """
        invalidated = (
            self._cancel_gen != send_gen
            or not self.key_sm.is_running()
            or not self.running
        )
        if not invalidated:
            return False
        print(
            f"  [PauseGuard] {context} send invalidated "
            f"(send_gen={send_gen}, cancel_gen={self._cancel_gen}, "
            f"key_running={self.key_sm.is_running()}, runner_running={self.running}); "
            "issuing post-send cancel"
        )
        self._cancel_locked(f"{context} send crossed pause")
        return True

    def _cold_start_locked(self):
        announced_wait = False
        while self._send_in_flight and self.running and self.key_sm.is_running():
            if not announced_wait:
                print("[cold-start] waiting for previous in-flight send to quiesce")
                announced_wait = True
            self.C.wait(timeout=0.1)
        if not self.running or not self.key_sm.is_running():
            return False

        self.C.release()
        try:
            if not self.key_sm.is_running() or not self.running:
                return False
            self.robot.wait_for_done(settle_ms=0, timeout_ms=1000.0)
            robot_obs = self.robot.get_observation(**self.obs_builder.fetch_kwargs())
            if robot_obs is None:
                print("[cold-start] no observation")
                return False
            t0 = time.monotonic()
            chunk, raw, state = self.infer_chunk(robot_obs, prefix_raw=None, delay=0)
            cold_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            print(f"[cold-start] failed: {e}")
            return False
        finally:
            self.C.acquire()
        if not self.key_sm.is_running() or not self.running:
            print("[cold-start] IDLE flipped mid-inference, discarding chunk")
            return False
        if raw is None:
            raise RuntimeError(
                "Server returned no action_pred_normalized on cold-start — "
                "train-time RTC needs it for the next chunk's prefix."
            )
        self.A_cur = chunk
        self.A_cur_raw = raw
        self.A_cur_state = state
        self.action_horizon = chunk[self.hand_key].shape[0]
        self.t = 0
        self._server_chunk_start_idx = 0
        self.Q.clear()
        send_gen = self._cancel_gen
        self._send_in_flight = True
        self.C.release()
        try:
            send_result = self._send_chunk(chunk, s_used_local=None, is_init=True)
        finally:
            self.C.acquire()
        try:
            invalidated = self._post_send_pause_guard_locked(send_gen, "cold-start")
        finally:
            self._send_in_flight = False
            if self._deferred_cancel_ack:
                self.key_sm.ack_cancel(True)
                self._deferred_cancel_ack = False
            self.C.notify_all()
        if invalidated:
            return False
        if send_result is None:
            # A transport failure is ambiguous: the server may have installed
            # the chunk before its response was lost. Restore the IDLE
            # invariant and clear the local cold-start state explicitly.
            self.key_sm.force_idle("cold-start chunk send failed")
            self._cancel_locked("cold-start chunk send failed")
            return False
        print(f"[cold-start] chunk sent in {cold_ms:.0f}ms "
              f"(H={self.action_horizon}, s_min={self.s_min})")
        return True

    def _cancel_locked(self, reason):
        self._cancel_gen += 1
        my_gen = self._cancel_gen
        self.C.release()
        try:
            ok = self.robot.cancel_chunk() is not False
        except Exception as e:
            ok = False
            print(f"  [Cancel] exception: {e}")
        finally:
            self.C.acquire()
        if self._send_in_flight and ok:
            self._deferred_cancel_ack = True
        else:
            self.key_sm.ack_cancel(bool(ok))
        self.A_cur = None
        self.A_cur_raw = None
        self.A_cur_state = None
        self.t = 0
        self._server_chunk_start_idx = 0
        self.Q.clear()
        self.Q.append(min(self.s_min, self.max_delay - 1))
        self.C.notify_all()
        print(f"[pause] chunk cancelled ({reason}) gen={my_gen}, server={ok}")

    def _inference_loop(self):
        with self.C:
            while self.running:
                if not self.C.wait_for(
                    lambda: (not self.running)
                    or (self.key_sm.is_running()
                        and self.A_cur_raw is not None
                        and self.t >= self.s_min),
                    timeout=0.5,
                ):
                    continue
                if not self.running:
                    break

                s = self.t
                inference_gen = self._cancel_gen
                A_prev_raw = self.A_cur_raw.copy() if self.A_cur_raw is not None else None
                S_prev = dict(self.A_cur_state) if self.A_cur_state is not None else None
                d_est = (max(self.Q) if self.Q else 1) + self._delay_safety_margin

                t_start = time.monotonic()
                s_used = s
                d_used = 0
                self.C.release()
                try:
                    self.robot.wait_for_done(settle_ms=0, timeout_ms=1000.0 * self.interval)
                    robot_obs = self.robot.get_observation(
                        **self.obs_builder.fetch_kwargs()
                    )
                    if robot_obs is not None:
                        try:
                            robot_obs = dict(robot_obs)
                            robot_obs["chunk_progress"] = \
                                self.robot.get_chunk_progress(timeout=0.05)
                        except Exception:   # noqa: BLE001
                            robot_obs["chunk_progress"] = None
                    if robot_obs is None:
                        print("  [Inference] obs failed, skipping")
                        chunk = raw = state = None
                    else:
                        # obs-time position in the FULL chunk (ticks advanced
                        # during wait+obs). Correct with the server's live arm
                        # played index when available (counters long wait_for_done).
                        s_used = min(int(self.t), self.action_horizon - 1)
                        prog = robot_obs.get("chunk_progress")
                        if prog is not None and prog.get("arm") is not None:
                            srv = float(prog["arm"])  # played in current server chunk
                            if srv >= 0:
                                s_full = (int(self._server_chunk_start_idx)
                                          + int(np.floor(srv)))
                                s_used = max(
                                    s_used, min(s_full, self.action_horizon - 1)
                                )
                        prefix_raw = None
                        if A_prev_raw is not None:
                            _, H_, D_pad = A_prev_raw.shape
                            avail = max(0, H_ - s_used)
                            if avail > 0:
                                obs_for_state = self.obs_builder.build(
                                    robot_obs, self.task_holder.current()
                                )
                                S_new = _state_dict_from_obs(obs_for_state)
                                prev_slice = A_prev_raw[0, s_used:s_used + avail, :]
                                if S_prev is not None and self.reanchor.enabled:
                                    prev_slice = self.reanchor.reanchor_prefix_slice(
                                        prev_slice, s_used, S_prev, S_new,
                                    )
                                prefix_raw = np.zeros_like(A_prev_raw)
                                prefix_raw[0, :avail, :] = prev_slice
                                upper = min(self.max_delay - 1, avail, self.action_horizon - 1)
                                d_used = max(1, min(d_est, upper))
                        chunk, raw, state = self.infer_chunk(
                            robot_obs, prefix_raw=prefix_raw, delay=d_used,
                        )
                finally:
                    self.C.acquire()

                if chunk is None:
                    continue
                if (self._cancel_gen != inference_gen
                        or not self.key_sm.is_running()
                        or not self.running):
                    continue  # paused mid-inference; drop, resume cold-starts

                self.A_cur = chunk
                if raw is not None:
                    self.A_cur_raw = raw
                self.A_cur_state = state
                self.action_horizon = chunk[self.hand_key].shape[0]

                # server-atomic swap: s_used_local = obs-time position in the
                # CURRENT server chunk (full-chunk s_used minus where the server
                # started this chunk). Server reads its live played idx at install
                # time and returns actual_delay = frames elapsed during inference;
                # resync the virtual tick to it (a2 AsyncRTCChunkTrainRunner).
                if A_prev_raw is not None:
                    s_used_local = max(0, int(s_used) - int(self._server_chunk_start_idx))
                    s_used_local = min(s_used_local, self.action_horizon - 1)
                else:
                    s_used_local = None

                send_gen = self._cancel_gen
                self._send_in_flight = True
                self.C.release()
                try:
                    actual_delay = self._send_chunk(
                        chunk, s_used_local=s_used_local, is_init=False,
                    )
                finally:
                    self.C.acquire()

                try:
                    invalidated = self._post_send_pause_guard_locked(send_gen, "RTC")
                finally:
                    self._send_in_flight = False
                    if self._deferred_cancel_ack:
                        self.key_sm.ack_cancel(True)
                        self._deferred_cancel_ack = False
                    self.C.notify_all()
                if invalidated:
                    continue

                if actual_delay is None:
                    self.key_sm.force_idle("chunk send failed")
                    self._cancel_locked("chunk send failed")
                    continue
                self.t = int(actual_delay)
                self.Q.append(int(actual_delay))
                t_ms = (time.monotonic() - t_start) * 1000
                print(f"  [Inference] {t_ms:.0f}ms, s_virt={s}, "
                      f"s_used={s_used if A_prev_raw is not None else '-'}, "
                      f"d_used={d_used}, actual_delay={actual_delay}, "
                      f"H={self.action_horizon}")

    def _execution_loop(self):
        was_running = False
        next_tick = time.monotonic()
        while self.running:
            is_run = self.key_sm.is_running()
            if is_run and not was_running:
                with self.C:
                    ok = self._cold_start_locked()
                    self.C.notify_all()
                if not ok:
                    self.key_sm.force_idle("cold-start failed")
                    was_running = False
                    time.sleep(0.05)
                    continue
            elif not is_run and was_running:
                with self.C:
                    self._cancel_locked("user pressed p")
                    self.C.notify_all()
            was_running = is_run

            if not is_run:
                if self.key_sm.take_reset_request():
                    if self.initial_pose is None:
                        print("  [RTC] 无 dataset 初始位姿, 跳过复位")
                    else:
                        reset_upper_body_arm(
                            self.robot,
                            self.initial_pose["arm"],
                            self.initial_pose["waist"],
                            target_hand=self.initial_pose["hand"],
                            reset_hand=False,
                            from_last_cmd=True,
                            label="reset-key-r",
                        )
                if self.key_sm.take_mode_switch_request():
                    self.next_mode = "standard"
                    self.running = False
                    with self.C:
                        self.C.notify_all()
                    break
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue

            with self.C:
                if self.A_cur is not None and self.t < self.action_horizon:
                    self.t += 1
                self.C.notify_all()

            next_tick += self.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def run(self):
        self.running = True
        print(
            f"[RTC-Chunk/upper] H={self.action_horizon}, max_delay={self.max_delay}, "
            f"s_min={self.s_min}, chunk_fps={self.chunk_fps}, "
            f"hand={self.hand_key}({self.hand_value}), "
            f"waist=server_paced(full_traj), "
            f"waist_h={'model' if self.use_model_waist_height else '0'}, "
            f"delta_reanchor={self.reanchor.enabled} "
            f"(server-atomic swap via mode=upper_body/s_used_local; "
            f"actual_delay=0 → soft-continuous fallback on an older a3_server)"
        )
        _print_start_prompt()
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()
        try:
            self._execution_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            with self.C:
                self.C.notify_all()
            infer_thread.join(timeout=5.0)
            try:
                self.robot.cancel_chunk()
            except Exception:
                pass


def run_standard_upper_body(policy_client, robot, obs_builder, decoder,
                            task_holder, key_sm, args, initial_pose=None):
    """Sync CHUNK mode: infer one chunk -> send the whole chunk via step_chunk
    (blocking until played) -> re-infer. No RTC. Gated on key_sm.is_running().
    Full waist trajectory is safe here (chunks don't overlap)."""
    hand_key = decoder.hand_key
    hand_value = decoder.hand_value
    use_model_waist_height = bool(getattr(args, "use_model_waist_height", False))
    chunk_fps = float(args.fps)
    was_running = False
    _print_start_prompt()
    try:
        while True:
            if not key_sm.is_running():
                if was_running:
                    # 'p': drop the chunk still queued in the 150Hz
                    # interpolator, else it keeps playing for up to H/fps more
                    # seconds. cancel_chunk holds the arm where it is.
                    try:
                        robot.cancel_chunk()
                        print("  [Standard] paused — chunk cancelled")
                    except Exception as exc:            # noqa: BLE001
                        print(f"  [Standard] cancel_chunk failed: {exc}")
                    was_running = False
                if key_sm.take_reset_request():
                    if initial_pose is None:
                        print("  [Standard] 无 dataset 初始位姿, 跳过复位")
                    else:
                        reset_upper_body_arm(
                            robot,
                            initial_pose["arm"],
                            initial_pose["waist"],
                            target_hand=initial_pose["hand"],
                            reset_hand=False,
                            from_last_cmd=True,
                            label="reset-key-r",
                        )
                if key_sm.take_mode_switch_request():
                    return "rtc_chunk"   # main() swaps in the RTC runner
                time.sleep(0.1)
                continue
            was_running = True
            t_obs = time.time()
            robot_obs = robot.get_observation(**obs_builder.fetch_kwargs())
            if robot_obs is None:
                print("  [Standard] obs failed, skipping")
                continue
            obs_ms = (time.time() - t_obs) * 1000
            t_build = time.time()
            obs = obs_builder.build(robot_obs, task_holder.current())
            build_ms = (time.time() - t_build) * 1000
            t0 = time.time()
            action_dict, _ = policy_client.get_action(obs)
            infer_ms = (time.time() - t0) * 1000
            chunk = decoder.decode(
                action_dict, obs_builder.hand_position_rad(robot_obs)
            )
            H = chunk[hand_key].shape[0]
            if args.action_horizon is not None:
                H = min(H, int(args.action_horizon))
                chunk = {k: v[:H] for k, v in chunk.items()}
            # Full waist trajectory (server-paced); chunks don't overlap in
            # standard mode. s_used_local=None → server installs the whole chunk
            # (no atomic slice); wait=True blocks until arm/hand/waist finish.
            cd = build_upper_chunk_dict(chunk, hand_key, hand_value,
                                        use_model_waist_height, waist_target_idx=None)
            t_exec = time.time()
            robot.step_chunk(
                cd, chunk_fps=chunk_fps, wait=True, mode="upper_body",
                timeout_ms=max(2000.0, (H / max(chunk_fps, 1.0)) * 1000.0 + 2000.0),
            )
            send_ms = (time.time() - t_exec) * 1000
            print(f"  [Standard] obs={obs_ms:.0f} build={build_ms:.0f} "
                  f"infer={infer_ms:.0f} | chunk={H} exec={send_ms:.0f}ms "
                  f"(gap={obs_ms + build_ms + infer_ms:.0f}ms)")
            if _WAIST_DEBUG and "waist" in chunk:
                st_w = np.asarray(
                    obs["state"].get("waist", np.zeros((1, 1, 3)))
                ).reshape(-1)[:3]
                w = np.asarray(chunk["waist"], dtype=np.float32)
                print(f"  [waist] state={np.round(st_w, 4).tolist()} "
                      f"act[0]={np.round(w[0], 4).tolist()} "
                      f"act[0]-state={np.round(w[0] - st_w, 4).tolist()}",
                      flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")


def detect_embodiment_kind(metadata: dict) -> str:
    """Auto-select the pipeline from the checkpoint's action layout (hot-swap).

    - ``whole_body``: sonic_a3 (``body`` + ``pelvis_quat6d``) or a3_config
      (``leg``/``leg_pos`` + ``waist``/``waist_pos`` + ``arm``/``arm_pos`` +
      ``pelvis_quat[6d]``) -> whole-body obs + ``step_chunk(mode=whole_body)``
      server-atomic RTC.
    - ``upper_body``: (``hand``|``gripper``) + ``arm`` + ``waist``
      [+ ``waist_height``] -> CHUNK send via ``robot.step_chunk()`` over the
      three mc per-part topics (arm / hand / waist); virtual-tick RTC with
      client-side soft-continuous swap.
    """
    akeys = set(metadata.get("action_keys") or [])
    has_pelvis = "pelvis_quat" in akeys or "pelvis_quat6d" in akeys
    split_joint_keys = (
        {"leg", "waist", "arm"} <= akeys
        or {"leg_pos", "waist_pos", "arm_pos"} <= akeys
    )
    if {"body", "pelvis_quat6d"} <= akeys or (split_joint_keys and has_pelvis):
        return "whole_body"
    if ("hand" in akeys or "gripper" in akeys) and {"arm", "waist"} <= akeys:
        return "upper_body"
    raise RuntimeError(
        f"Cannot auto-detect embodiment from action_keys={sorted(akeys)}. "
        f"Expected whole-body (body / leg[_pos]+waist[_pos]+arm[_pos]+pelvis) or upper-body "
        f"(hand|gripper + arm + waist)."
    )



def _probe_hand_kind_and_reachable(
    host: str,
    port: int,
    requested: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
    require_hand: bool = False,
    require_scattered_body: bool = True,
) -> str:
    """Stage A of server readiness — connectivity + hand_kind probe.

    Polls :port/get_joint_states until it returns a hand joint array; infers
    hand_kind from its length (20→hand, 2→gripper). Proves TCP:port is up
    and the low-level mc daemon (which publishes hand_joint_state) is
    running.

    Camera / IMU readiness is a separate stage (see _wait_for_sensors_ready)
    because it depends on the checkpoint's modality schema, which we don't
    have until Gr00tPolicy has been loaded.

    Args:
        requested: 'auto' | 'hand' | 'gripper'. Non-'auto' still probes and
            RAISES on mismatch — schema disagreement between client & server
            would silently misroute hand-related state / action reads.

    Returns: resolved 'hand' or 'gripper'.
    """
    import json as _json
    import urllib.error as _ue
    import urllib.request as _ur

    url = f"http://{host}:{port}/get_joint_states"
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    detected: str | None = None
    hand_len_seen: int | None = None

    print(
        f"[ws] Stage A: waiting for a3_server at {url} "
        f"(timeout={timeout_s}s, probing hand_kind ...)"
    )
    reachable = False
    while time.monotonic() < deadline:
        try:
            with _ur.urlopen(url, timeout=poll_interval_s * 2) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except (_ue.URLError, _ue.HTTPError, TimeoutError, OSError) as e:
            last_err = str(e)
            time.sleep(poll_interval_s)
            continue
        except Exception as e:  # bad JSON etc
            last_err = f"unexpected: {e}"
            time.sleep(poll_interval_s)
            continue

        if not isinstance(data, dict):
            last_err = "server up but joint_states payload is not an object"
            time.sleep(poll_interval_s)
            continue
        # Whole-body inference reads leg/waist/arm atomically from the WBC
        # protobuf chain, so scattered body joints are not a Stage-A dependency.
        if require_scattered_body and not (data.get("arm") or data.get("leg")):
            last_err = "server up but scattered joint state has no arm/leg"
            time.sleep(poll_interval_s)
            continue
        reachable = True

        hand = (data.get("hand") or {}) if isinstance(data, dict) else {}
        pos = hand.get("position") if isinstance(hand, dict) else None
        if isinstance(pos, list) and len(pos) in (2, 20):
            hand_len_seen = len(pos)
            detected = "gripper" if hand_len_seen == 2 else "hand"
            break

        if require_hand:
            last_err = "server up but required 2D/20D hand_joint_state is absent"
            time.sleep(poll_interval_s)
            continue

        # Checkpoints without hand state/action do not need to block here.
        detected = requested if requested != "auto" else "hand"
        print(
            f"[ws] Stage A: a3_server ready but hand.position absent "
            f"(/motion/control/hand_joint_state 无数据);checkpoint 不使用 hand,"
            f"hand_kind={detected}(requested={requested})。若本机确有灵巧手/夹爪需要正确 "
            f"schema,请显式传 --hand_kind hand|gripper。"
        )
        break

    if not reachable:
        raise RuntimeError(
            f"a3_server at {url} did not become ready within {timeout_s}s "
            f"(last error: {last_err}). Bring the server up first "
            f"(RoboInterface/scripts/start_robot_a3.sh)."
        )
    if require_hand and hand_len_seen is None:
        raise RuntimeError(
            f"checkpoint requires hand/gripper state or action, but {url} did "
            f"not provide a 2D/20D hand.position within {timeout_s}s "
            f"(last error: {last_err})."
        )

    # 只有"真检测到 hand 长度"且与显式请求冲突时才报错(hand 缺失不算冲突)。
    if requested != "auto" and hand_len_seen is not None and requested != detected:
        raise RuntimeError(
            f"--hand_kind={requested} disagrees with a3_server "
            f"(hand.position length={hand_len_seen} → {detected}). "
            f"A schema disagreement will silently mis-route hand-related state "
            f"and action values. Fix one side: pass --hand_kind={detected} or "
            f"restart the server with A3_HAND_KIND={requested}."
        )
    print(f"[ws] Stage A: a3_server ready; hand_kind={detected} (hand.position len={hand_len_seen})")
    return detected


def _wait_for_sensors_ready(
    host: str,
    port: int,
    cameras_needed: list[str],
    need_imu: bool,
    timeout_s: float = 60.0,
    poll_interval_s: float = 1.0,
    state_source: Optional[str] = None,
) -> None:
    """Stage B of server readiness — cameras + IMU probe.

    Verifies that each required camera returns a real JPEG body (a 200 with
    an empty response body would still deserialize; we require content_length
    > 0 AND at least the JPEG SOI marker) and that the pelvis IMU is available
    when the checkpoint asks for pelvis_gravity / pelvis_orient6d.

    ``state_source`` selects where the IMU probe reads from:
      - None → scattered chain, /get_imu.
      - "whole_body_state" → WBC chain, /get_whole_body_state (imu.pelvis);
        this is the same source obs will read, so it also confirms the
        /wbc/whole_body_state subscription (created by set_state_source) is
        actually receiving frames.

    Why:
        _preprocess_image() replaces None with a 256×256 zero image — if we
        start inference before the camera-driver has published any frame,
        the policy sees black inputs and can emit an unsafe first chunk. IMU
        gets a default (0,0,-1) gravity + identity quat, same failure mode.

        Rather than fail late deep inside build(), fail loud at startup.

    cameras_needed and need_imu should come from A3ObsBuilder after the
    checkpoint's modality schema is resolved (only probes what the model
    actually reads).
    """
    import urllib.error as _ue
    import urllib.request as _ur

    if not cameras_needed and not need_imu and state_source != "whole_body_state":
        print("[ws] Stage B: checkpoint needs no cameras and no IMU — skipping")
        return

    _imu_via_wb = state_source == "whole_body_state"

    def _cam_ok(name: str) -> tuple[bool, str]:
        try:
            u = f"http://{host}:{port}/get_camera/{name}"
            with _ur.urlopen(u, timeout=poll_interval_s * 2) as resp:
                if resp.status != 200:
                    return False, f"HTTP {resp.status}"
                body = resp.read()
                # JPEG SOI marker
                if len(body) < 4 or body[:2] != b"\xff\xd8":
                    return False, f"not JPEG (len={len(body)})"
                return True, f"jpeg {len(body)}B"
        except (_ue.URLError, _ue.HTTPError, TimeoutError, OSError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"unexpected: {e}"

    def _imu_ok() -> tuple[bool, str]:
        import json as _json
        # whole_body: pelvis IMU lives under /get_whole_body_state["imu"];
        # otherwise it's the scattered /get_imu endpoint.
        url = (f"http://{host}:{port}/get_whole_body_state" if _imu_via_wb
               else f"http://{host}:{port}/get_imu")
        try:
            with _ur.urlopen(url, timeout=poll_interval_s * 2) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            if _imu_via_wb:
                imu = data.get("imu") if isinstance(data, dict) else None
                pelvis = imu.get("pelvis") if isinstance(imu, dict) else None
            else:
                pelvis = data.get("pelvis") if isinstance(data, dict) else None
            if not pelvis:
                return False, "pelvis null"
            # gravity_dir + orientation_xyzw both required by builders.
            gv = pelvis.get("gravity_dir")
            q = pelvis.get("orientation_xyzw")
            if not (isinstance(gv, list) and len(gv) == 3):
                return False, f"gravity_dir bad ({gv})"
            if not (isinstance(q, list) and len(q) == 4):
                return False, f"orientation_xyzw bad ({q})"
            return True, f"gravity+quat ok"
        except (_ue.URLError, _ue.HTTPError, TimeoutError, OSError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"unexpected: {e}"

    def _wb_joints_ok() -> tuple[bool, str]:
        """whole_body only: confirm /wbc/whole_body_state joints are flowing
        (leg+waist+arm non-empty). Guards against a dead wbc topic — the obs
        builder silently zero-fills missing joints (_joint_slice), so without
        this a stalled /wbc/whole_body_state would produce a zero-state first
        chunk. Runs regardless of need_imu because whole_body always reads body
        joint state."""
        import json as _json
        url = f"http://{host}:{port}/get_whole_body_state"
        try:
            with _ur.urlopen(url, timeout=poll_interval_s * 2) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            joints = data.get("joints") if isinstance(data, dict) else None
            if not isinstance(joints, dict):
                return False, "joints null"
            missing = [k for k in ("leg", "waist", "arm")
                       if not ((joints.get(k) or {}).get("position"))]
            if missing:
                return False, f"empty {missing}"
            return True, "leg+waist+arm ok"
        except (_ue.URLError, _ue.HTTPError, TimeoutError, OSError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"unexpected: {e}"

    print(
        f"[ws] Stage B: probing cameras={cameras_needed} need_imu={need_imu} "
        f"wb_state={_imu_via_wb} (timeout={timeout_s}s)"
    )
    deadline = time.monotonic() + timeout_s
    pending_cams: set[str] = set(cameras_needed)
    pending_imu = need_imu
    pending_wb_joints = _imu_via_wb   # whole_body: verify wbc joint state flows
    last_errs: dict[str, str] = {}
    while time.monotonic() < deadline and (pending_cams or pending_imu
                                           or pending_wb_joints):
        for name in list(pending_cams):
            ok, info = _cam_ok(name)
            if ok:
                print(f"  [Stage B] camera {name}: {info}")
                pending_cams.discard(name)
                last_errs.pop(name, None)
            else:
                last_errs[name] = info
        if pending_wb_joints:
            ok, info = _wb_joints_ok()
            if ok:
                print(f"  [Stage B] wbc joints: {info}")
                pending_wb_joints = False
                last_errs.pop("wb_joints", None)
            else:
                last_errs["wb_joints"] = info
        if pending_imu:
            ok, info = _imu_ok()
            if ok:
                print(f"  [Stage B] imu: {info}")
                pending_imu = False
                last_errs.pop("imu", None)
            else:
                last_errs["imu"] = info
        if pending_cams or pending_imu or pending_wb_joints:
            time.sleep(poll_interval_s)

    if pending_cams or pending_imu or pending_wb_joints:
        details = "; ".join(f"{k}={v}" for k, v in last_errs.items())
        missing = (list(pending_cams)
                   + (["imu"] if pending_imu else [])
                   + (["wb_joints"] if pending_wb_joints else []))
        imu_hint = (
            "IMU/state: check /wbc/whole_body_state is publishing on ADU (feeds "
            "both wbc joints and pelvis_imu) and that set_state_source(whole_body) ran."
            if _imu_via_wb else
            "IMU: check /body_drive/pelvis_imu/data on ADU."
        )
        raise RuntimeError(
            f"a3_server sensors not ready within {timeout_s}s. "
            f"Missing: {missing}. Details: {details}. "
            f"Cameras: check the driver on ADU is publishing (ros2 topic hz). "
            f"{imu_hint}"
        )
    print("[ws] Stage B: all cameras + IMU + wbc state (as required) ready")


# ============================================================================
# Human-in-the-loop HTTP control (--human_in_loop)
# ============================================================================


def _str2bool(v) -> bool:
    """argparse type for an explicit true/false flag (``--human_in_loop true``).
    Accepts 1/true/yes/y/on (case-insensitive) as True; everything else False."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def start_human_in_loop_server(key_sm: "KeyStateMachine", host: str, port: int):
    """Start a tiny HTTP control server for human-in-the-loop operation.

    Two endpoints flip the shared KeyStateMachine, reusing the runner's
    existing RUNNING/IDLE edge handling verbatim (cold-start on the
    IDLE→RUNNING edge, ``robot.cancel_chunk()`` on RUNNING→IDLE — see
    AsyncRTCChunkTrainRunner._execution_loop / _cold_start_locked /
    _cancel_and_reset_locked). No new control path into the runner is added;
    HTTP just presses 's' / 'p' remotely, so all the send/cancel race guards
    already in place still hold.

      POST|GET /start  → RUNNING. The execution loop cold-starts a FRESH chunk
                         with NO prefix (rtc 模式下 prefix 为空的第一次推理);
                         the inference loop then continues with the RTC prefix
                         from the 2nd chunk on. After a /stop the robot pose has
                         moved (teleop), so each /start re-anchors from scratch.
      POST|GET /stop   → IDLE. The execution loop calls robot.cancel_chunk():
                         the model stops sending and a3_server stops executing
                         the current chunk (interpolator holds the last frame),
                         handing control to the teleoperator (摇操).
      GET  /status     → {"ok": true, "state": "RUNNING"|"IDLE"} (convenience).

    Runs serve_forever() on a daemon thread; returns the ThreadingHTTPServer so
    the caller can shut it down on exit.
    """
    import json as _json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, payload: dict) -> None:
            body = _json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            path = (self.path.split("?", 1)[0]).rstrip("/") or "/"
            if path == "/start":
                ok = key_sm.set_running("http /start (human-in-loop)")
                self._reply(200 if ok else 409, {"ok": ok, **key_sm.status_snapshot()})
            elif path == "/stop":
                key_sm.force_idle("http /stop (human-in-loop → teleop)")
                self._reply(200, {"ok": True, **key_sm.status_snapshot()})
            elif path == "/grasp/arm":
                ok = key_sm.arm_grasp()
                self._reply(200 if ok else 409, {"ok": ok, **key_sm.status_snapshot()})
            elif path == "/grasp/heartbeat":
                ok = key_sm.heartbeat_grasp()
                self._reply(200 if ok else 409, {"ok": ok, **key_sm.status_snapshot()})
            elif path in ("/grasp/complete", "/grasp/fault"):
                result = "GRASP_CONFIRMED" if path.endswith("complete") else "SENSOR_UNAVAILABLE"
                ok = key_sm.latch_grasp(result)
                self._reply(200 if ok else 409, {"ok": ok, **key_sm.status_snapshot()})
            elif path in ("/status", "/"):
                self._reply(200, {"ok": True, **key_sm.status_snapshot()})
            else:
                self._reply(404, {
                    "ok": False,
                    "error": f"unknown path {path!r}",
                    "endpoints": ["/start", "/stop", "/status"],
                })

        # Both verbs do the same thing so callers can `curl -X POST` or just GET.
        do_GET = _handle
        do_POST = _handle

        def log_message(self, fmt, *a):  # noqa: A003 — silence default access log
            pass

    httpd = ThreadingHTTPServer((host, port), _Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    print(
        f"[ws] human-in-loop HTTP control on http://{host}:{port} "
        f"(POST/GET /start → cold-start+RTC, /stop → cancel+teleop, /status)"
    )
    return httpd


def _check_metadata(meta: dict) -> None:
    """Sanity checks — surface the exact error early if the policy config is
    incompatible instead of failing deep inside a decode."""
    if not meta or "action_keys" not in meta:
        raise RuntimeError(
            "Empty rtc_metadata from Gr00tPolicy. Check gr00t/policy/gr00t_policy.py "
            "— get_rtc_metadata should return the checkpoint schema."
        )
    if "video_keys" not in meta or "state_keys" not in meta:
        raise RuntimeError(
            "get_rtc_metadata missing video_keys / state_keys. Update "
            "gr00t/policy/gr00t_policy.py::get_rtc_metadata to surface them "
            "(the client can no longer hard-code the profile)."
        )


def main():
    p = argparse.ArgumentParser(
        description="A3 humanoid config-driven GR00T RTC inference "
                    "(arXiv:2512.05964, REMOTE ZMQ PolicyClient)"
    )
    # Robot / model-server
    p.add_argument("--transport", choices=["subproc", "inproc", "http"],
                   default="subproc",
                   help="robot transport. subproc (默认) = A3ServerNode 跑在独立 "
                        "ROS 子进程，whole-body 原始相机、WBC state/IMU/progress "
                        "经 POSIX shared memory 回传，chunk 经本地 pipe 进入同一 "
                        "server-atomic swap；无需另起 HTTP a3_server。inproc = "
                        "同进程 import+spin，仅保留给 upper-body standard。http = "
                        "兼容旧链路，需另起 a3_server，并执行 JPEG/base64/JSON。")
    p.add_argument("--worker-python", type=str, default="/usr/bin/python3",
                   help="subproc: ROS 子进程用的解释器。必须是 ROS jazzy 原生环境 "
                        "(系统 python3), 它的 numpy/cv_bridge 才匹配。")
    p.add_argument("--robot_ip", type=str, default="192.168.100.100",
                   help="inproc: A3ServerNode 连接机器人底层的 IP；"
                        "http: a3_server HTTP host (本机 a3_server 用 127.0.0.1)")
    p.add_argument("--robot_port", type=int, default=5050)
    p.add_argument("--engine-dir", type=str,
                   default=os.environ.get("A3_ENGINE_DIR", "/agibot/models/a3_60000/engines"),
                   help="端侧: 7 个 TRT engine 目录")
    p.add_argument(
        "--trt-mode",
        type=str,
        default=os.environ.get("A3_TRT_MODE", "vit_llm_only"),
        choices=["n17_full_pipeline", "vit_llm_only", "action_head", "dit_only"],
        help="TensorRT patch mode. vit_llm_only (default) keeps the PyTorch "
             "action head and therefore uses gr00t/dev's exact per-token train-time "
             "RTC. n17_full_pipeline is faster but uses a compatibility prefix guard.",
    )
    p.add_argument("--model-path", type=str,
                   default=os.environ.get("A3_MODEL_PATH", "/agibot/models/a3_60000/ckpt"),
                   help="端侧: checkpoint 目录(构建 Gr00tPolicy)")
    p.add_argument("--embodiment-tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--dataset", type=str,
                   default=os.environ.get("A3_DATASET"),
                   help="Optional LeRobot dataset used for the episode-0 frame-0 "
                        "upper-body reset pose and, when --task is omitted, the prompt. "
                        "Whole-body --no-reset runs do not require it.")
    p.add_argument("--dataset-episode", type=int, default=0,
                   help="Episode index used for the initial pose (default: 0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load policy/metadata and any configured dataset pose, then "
                        "exit without connecting or sending robot commands.")
    p.add_argument("--sensor-check-only", action="store_true",
                   help="Load policy, start the selected robot transport, verify "
                        "checkpoint-required whole-body state/cameras/IMU, then "
                        "exit without installing or publishing an action chunk.")
    p.add_argument("--model_host", type=str, default="localhost",
                   help="Gr00tPolicyServer ZMQ host (run_gr00t_server.py --host).")
    p.add_argument("--model_port", type=int, default=5555,
                   help="Gr00tPolicyServer ZMQ port (run_gr00t_server.py --port).")
    p.add_argument("--no-hand-spread", dest="hand_spread",
                   action="store_false", default=True,
                   help="Only move the index finger for hand_opening models. "
                        "By default the single index-bend action drives all "
                        "five fingers (each scaled to its own joint limits).")
    p.add_argument("--hand_kind", type=str, default="hand",
                   choices=["auto", "hand", "gripper"],
                   help="'auto' probes a3_server /get_joint_states and infers "
                        "from hand joint length (20→hand, 2→gripper). "
                        "RTC data flow doesn't move hand, but the interface's "
                        "hand-related paths depend on this being consistent "
                        "with the server-side setting.")
    p.add_argument(
        "--state-source",
        type=str,
        default="whole_body_state",
        choices=["whole_body_state", "scattered"],
        help="Whole-body observation source. whole_body_state reads the WBC "
        "protobuf channel (real robot default); scattered reads "
        "/get_joint_states + /get_imu (dataset-driven simulation).",
    )

    # Task / prompt
    p.add_argument("--task", type=str, default=None,
                   help="Boot task (language prompt). Defaults to the --dataset's "
                        "own meta/tasks.jsonl entry, which is what the checkpoint "
                        "was trained on — a mismatched prompt produces garbage "
                        "actions. Pass explicitly to override.")
    p.add_argument("--prompt-yaml", type=str,
                   default=str(REPO / "scripts_crp/prompt.yaml"),
                   help="prompt.yaml with tasks[] — single source of truth for "
                        "the 1..9 preset list. Missing file → silent skip.")

    # Head camera routing (checkpoint-dependent)
    p.add_argument("--head-cam-source", type=str, default="head_stereo_left",
                   choices=["head_stereo_left", "head_stereo_right"],
                   help="Physical camera to route into the sonic 'head_front' "
                        "video key. sonic_a3 checkpoints may be trained on "
                        "either eye; check the training pipeline you baked.")

    # FPS: policy(30) → wire(50)
    p.add_argument("--policy-output-fps", type=float, default=30.0,
                   help="Original chunk fps of the VLA (matches training "
                        "dataset). 30 is A3 default.")
    p.add_argument("--wire-fps", type=float, default=50.0,
                   help="Wire fps (a3_server REF_WINDOW_PUBLISH_HZ). Upsample "
                        "target. Also the virtual-tick rate.")

    # RTC / mode
    p.add_argument("--mode", type=str, default="rtc_chunk",
                   choices=["rtc_chunk", "standard"])
    p.add_argument("--exec_steps", type=int, default=15,
                   help="Trigger next inference after this many policy (pre-upsample) "
                        "frames of the current chunk (paper's s_min).")
    p.add_argument("--rtc_delay_margin", type=int, default=2)
    p.add_argument("--no_delta_reanchor", action="store_true", default=False)
    p.add_argument("--no_rtc_prefix", action="store_true", default=False,
                   help="关掉 RTC train-time prefix pin:每次推理纯 sample,不喂上一 "
                        "chunk 的 reanchored 尾巴。chunk 原子替换照常,边界 jump 交给 "
                        "server adaptive_transition 平滑(建议配 --server_transition "
                        "adaptive)。sim_inference 用;真机默认关(保留 pin)。")
    p.add_argument("--server_transition", type=str, default="fixed",
                   choices=["adaptive", "fixed"])
    p.add_argument("--action_horizon", type=int, default=None,
                   help="Cap chunk length (standard mode only; wire frames).")
    p.add_argument("--rtc_max_delay_override", type=int, default=None,
                   help="Force rtc_max_delay client-side. Diagnostic only.")
    p.add_argument("--auto-run", action="store_true", default=False,
                   help="Skip the KeyStateMachine 'press s' gate. Only sensible "
                        "in headless setups (systemd/cron) where the robot is "
                        "not attached to a physical operator — otherwise the "
                        "cold-start chunk begins pushing to a3_server "
                        "immediately.")

    # ---- Human-in-the-loop HTTP control ----
    p.add_argument("--human_in_loop", type=_str2bool, default=False,
                   help="true/false. false(默认)=普通正常推理(键盘 s/p 门控)。"
                        "true=开启两个 HTTP 接口远程门控:/start 从零 cold-start "
                        "(prefix 为空)开启推理,之后 chunk 用 RTC prefix;/stop 立马"
                        "停止模型发送并让 a3_server 取消当前 chunk(cancel_chunk,"
                        "保持最后一帧),交给摇操接管。语义与按 s/p 完全一致,只是"
                        "走 HTTP。prefix 语义仅在 --mode rtc_chunk 下成立。")
    p.add_argument("--human_in_loop_host", type=str, default="0.0.0.0",
                   help="Bind host for the --human_in_loop HTTP control server "
                        "(default 0.0.0.0 so an external teleop station can "
                        "reach it).")
    p.add_argument("--human_in_loop_port", type=int, default=5100,
                   help="Bind port for the --human_in_loop HTTP control server.")
    p.add_argument("--grasp-stop-enabled", action="store_true",
                   help="Require local pressure monitor arm before start; latch on grasp/fault.")
    p.add_argument("--server-ready-timeout-s", type=float, default=120.0,
                   help="How long to poll a3_server /get_joint_states before "
                        "giving up. Ballpark: rsync + colcon build on the ADU "
                        "takes 60-90s from a cold state, and the wait is a "
                        "no-op once the server is already up.")
    p.add_argument("--sensor-ready-timeout-s", type=float, default=60.0,
                   help="How long to wait for each required camera and (if "
                        "the ckpt uses it) the IMU to publish a real frame. "
                        "Fails LOUD instead of silently feeding the model "
                        "black images / zero-IMU defaults.")
    p.add_argument("--record-chunks-dir", type=str, default=None,
                   help="记录每次 send 的完整 chunk + server actual_delay + 时间戳到 "
                        "该目录的 chunks_<RUN_ID>.npz(默认开启)。默认取 $A3_LOG_DIR"
                        "(部署时自动带),都没有则用 CWD;传 'off' 关闭记录。")

    # ---- Upper-body (mc-topic) extras — ignored by the whole-body path ----
    p.add_argument("--fps", type=float, default=30,
                   help="Upper-body control/send rate. Defaults to "
                        "--policy-output-fps when unset.")
    p.add_argument("--arm_interp", action="store_true", default=False,
                   help="Upper-body: route the arm through the pnc_arm "
                        "interpolation channel (robot.send_arm) instead of "
                        "the mc passthrough topic. Needs ADU trajectory mode.")
    p.add_argument("--set_arm_mode", type=str, default="none",
                   choices=["none", "passthrough", "interp"],
                   help="Upper-body: switch ADU pnc_arm mode at startup via "
                        "set_mode() (needs sshpass). 'none' leaves it as-is.")
    p.add_argument("--use_model_waist_height", action="store_true", default=False,
                   help="Upper-body: send the model waist_height as the waist "
                        "4th dim. Default pads it with 0 (configs did not train it).")
    p.add_argument("--waist_full_traj", action="store_true", default=False,
                   help="Upper-body: pace the FULL per-frame waist trajectory "
                        "(a3_server cannot chunk waist, so the interface threads "
                        "it per row). Default sends ONE waist target per chunk to "
                        "avoid overlapping waist threads across RTC chunks.")
    p.add_argument("--no-reset", action="store_true", default=False,
                   help="Upper-body: 跳过 run 前后的复位。默认在开始和结束各把"
                        "手臂+手+腰复位到 UPPER_BODY_RESET_ARM_POS / _WAIST + open手 "
                        "(镜像 A2 的 robot.reset())。whole-body 忽略此项。")
    p.add_argument("--reset-hand-pose", type=str, default="open",
                   choices=["open", "fist", "none"],
                   help="Upper-body 复位时手/夹爪的目标姿势:open=张开(默认), "
                        "fist=握拳/夹爪闭合, none=复位时不动手。维度按 hand_kind "
                        "自动 (灵巧手 20D / 夹爪 2D)。")

    args = p.parse_args()
    if args.grasp_stop_enabled:
        if not args.human_in_loop or args.human_in_loop_host not in ("127.0.0.1", "localhost", "::1"):
            p.error("--grasp-stop-enabled requires --human_in_loop true and localhost binding")
        if args.mode != "rtc_chunk" or args.auto_run:
            p.error("--grasp-stop-enabled requires --mode rtc_chunk without --auto-run")
        if args.task != "抓瓶子":
            p.error("--grasp-stop-enabled requires --task 抓瓶子")
    if getattr(args, "fps", None) is None:
        args.fps = args.policy_output_fps

    # subproc supports both modes; inproc still only drives the standard loop.
    if args.transport == "inproc" and args.mode != "standard":
        raise ValueError(
            "--transport inproc supports --mode standard only; use "
            "--transport subproc (default) for RTC"
        )

    # 1) Load the local TensorRT policy. In-process ROS is initialized only
    #    after dry-run has returned, so --dry-run remains side-effect free.
    print(f"[edge] 本地 engine 推理: model={args.model_path} engines={args.engine_dir}")
    t0 = time.time()
    policy_client = LocalEnginePolicy(
        model_path=args.model_path,
        engine_dir=args.engine_dir,
        embodiment_tag=args.embodiment_tag,
        trt_mode=args.trt_mode,
    )
    print(f"[edge] LocalEnginePolicy 就绪 in {time.time() - t0:.1f}s")

    # 2) Metadata + schema (fetched from the server; no local checkpoint load).
    metadata = policy_client.get_rtc_metadata()
    _check_metadata(metadata)
    embodiment_kind = detect_embodiment_kind(metadata)
    requires_hand = bool(
        {"hand", "hand_opening", "gripper", "gripper_opening"}
        & (
            set(metadata.get("state_keys") or [])
            | set(metadata.get("action_keys") or [])
            | set(metadata.get("reference_only_keys") or [])
        )
    )
    initial_pose = None
    if args.dataset:
        initial_pose = load_dataset_initial_pose(args.dataset, args.dataset_episode)
    # The prompt must match training, or the policy emits unrelated motion.
    if args.task is None:
        if not args.dataset:
            raise RuntimeError(
                "--task is required when --dataset is not provided. Use the exact "
                "training prompt from checkpoint/experiment_cfg/launch/prompts.json."
            )
        args.task = load_dataset_task(args.dataset)
        if args.task is None:
            raise RuntimeError(
                f"{args.dataset}/meta/tasks.jsonl has no task; pass --task "
                f"with the exact prompt this checkpoint was trained on."
            )
        print(f"[edge] task from dataset: {args.task!r}", flush=True)
    else:
        ds_task = load_dataset_task(args.dataset) if args.dataset else None
        if ds_task and ds_task != args.task:
            print(
                f"[edge] WARNING: --task {args.task!r} differs from the dataset's "
                f"training prompt {ds_task!r}; expect degraded actions.",
                flush=True,
            )
    if args.dry_run:
        print("[edge] dry-run complete: no robot connection and no commands sent", flush=True)
        return
    print(f"[ws] embodiment layout (hot-swap): {embodiment_kind}")
    if embodiment_kind == "upper_body" and not args.no_reset and initial_pose is None:
        raise RuntimeError(
            "upper-body reset is enabled but --dataset was not provided; pass the "
            "matching LeRobot dataset or use --no-reset."
        )
    if args.transport == "inproc" and embodiment_kind != "upper_body":
        raise ValueError(
            "--transport inproc currently supports the upper_body A3 layout only; "
            "use --transport subproc for whole-body shared-memory RTC"
        )
    if embodiment_kind == "whole_body":
        if args.state_source != "whole_body_state":
            raise ValueError(
                "whole-body A3 inference requires --state-source whole_body_state; "
                "the scattered endpoints are not an atomic WBC observation"
            )
        if args.transport == "http":
            _require_wholebody_robointerface_protocol()

    # 3) Create either the legacy HTTP client or the documented in-process
    #    A3ServerNode adapter. The latter owns ROS callbacks and never performs
    #    JPEG/base64/HTTP round trips for observations or actions.
    inproc_robot = None
    if args.transport in ("inproc", "subproc"):
        resolved_hand_kind = args.hand_kind
        if resolved_hand_kind == "auto":
            resolved_hand_kind = "hand" if any(
                key in {"hand", "hand_action", "hand_opening"}
                for key in (metadata.get("state_keys") or []) + (metadata.get("action_keys") or [])
            ) else "gripper"
        if args.transport == "subproc":
            # The worker subscribes at startup, so resolve the camera list from
            # the checkpoint's video keys here (obs_builder is built later and
            # derives the same mapping).
            _cam_map = dict(VIDEO_CAMERA_MAP_DEFAULT)
            _cam_map["head_front"] = args.head_cam_source
            worker_cams = [_cam_map[k] for k in (metadata.get("video_keys") or [])
                           if k in _cam_map]
            robot = SubprocA3Robot(
                hand_kind=resolved_hand_kind,
                cameras=worker_cams,
                embodiment_kind=embodiment_kind,
                robot_ip=args.robot_ip,
                worker_python=args.worker_python,
            )
        else:
            robot = InProcessA3Robot(
                hand_kind=resolved_hand_kind,
                robot_ip=args.robot_ip,
            )
        inproc_robot = robot
        deadline = time.monotonic() + float(args.sensor_ready_timeout_s)
        while True:
            ready, missing = robot.ready(
                cameras=[],
                require_hand=requires_hand,
                require_imu=False,
            )
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"A3ServerNode ({args.transport}) sensors not ready: {missing}")
            time.sleep(0.1)
        print(f"[{args.transport}] joints/callbacks ready "
              f"(hand_kind={resolved_hand_kind})", flush=True)
    else:
        resolved_hand_kind = _probe_hand_kind_and_reachable(
            args.robot_ip, args.robot_port, args.hand_kind,
            timeout_s=float(args.server_ready_timeout_s),
            require_scattered_body=(embodiment_kind != "whole_body"),
            require_hand=requires_hand,
        )
        print(f"[ws] connecting robot: {args.robot_ip}:{args.robot_port} "
              f"(hand_kind={resolved_hand_kind})")
        robot = A3RobotInterface(args.robot_ip, args.robot_port, hand_kind=resolved_hand_kind)

    # HTTP-only state-source and motor-mode controls are skipped in inproc mode.
    _wb_state = embodiment_kind == "whole_body" and args.state_source == "whole_body_state"
    server_state_source = "whole_body" if _wb_state else "upper_body"
    if args.transport == "http":
        if robot.set_state_source(server_state_source):
            print(f"[ws] state_source={args.state_source} (server={server_state_source})")
        elif _wb_state:
            print("[ws] WARNING: /set_state_source not acked by a3_server")
    if embodiment_kind == "upper_body":
        robot.set_speed(hz=float(args.fps))
        if args.transport == "http" and args.set_arm_mode != "none":
            want_interp = args.set_arm_mode == "interp"
            print(f"[ws] set_mode(interp={want_interp}) ...")
            if not robot.set_mode(interp=want_interp):
                print("[ws] WARNING: set_mode failed; using current ADU mode")

    # 4) Video map + obs/decoder.
    video_camera_map = dict(VIDEO_CAMERA_MAP_DEFAULT)
    video_camera_map["head_front"] = args.head_cam_source

    if embodiment_kind == "upper_body":
        obs_builder = A3UpperBodyObsBuilder(
            metadata, video_camera_map, hand_kind=resolved_hand_kind
        )
        decoder = A3UpperBodyActionDecoder(metadata)
    else:
        obs_builder = A3ObsBuilder(
            metadata,
            video_camera_map,
            state_source="whole_body_state" if _wb_state else None,
            hand_kind=resolved_hand_kind,
        )
        decoder = A3ActionDecoder(metadata, hand_spread=args.hand_spread)

    # 4.5) Readiness Stage B. HTTP mode probes endpoints; inproc mode checks
    #      the same caches directly, so no HTTP request is made.
    if args.transport == "http":
        _wait_for_sensors_ready(
            args.robot_ip, args.robot_port,
            cameras_needed=obs_builder.cameras_needed,
            need_imu=obs_builder.needs_imu,
            timeout_s=float(args.sensor_ready_timeout_s),
            state_source="whole_body_state" if _wb_state else None,
        )
    else:
        robot.keep_only_cameras(list(obs_builder.cameras_needed))
        deadline = time.monotonic() + float(args.sensor_ready_timeout_s)
        while True:
            ready, missing = robot.ready(
                cameras=list(obs_builder.cameras_needed),
                require_hand=requires_hand,
                require_imu=obs_builder.needs_imu,
            )
            if ready:
                print(
                    f"[{args.transport}] sensors ready "
                    f"cameras={list(obs_builder.cameras_needed)}",
                    flush=True,
                )
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"A3ServerNode ({args.transport}) sensors not ready: {missing}"
                )
            time.sleep(0.1)
    print(
        f"Config from policy server:\n"
        f"  action_keys : {metadata['action_keys']} "
        f"(H={metadata['action_horizon']}, rtc_max_delay={metadata['rtc_max_delay']})\n"
        f"  state_keys  : {metadata['state_keys']}\n"
        f"  ref_only    : {metadata.get('reference_only_keys')}\n"
        f"  video_keys  : {metadata.get('video_keys')} "
        f"→ cameras {obs_builder.cameras_needed} (head_front={args.head_cam_source})\n"
        f"  action_reps : {metadata.get('action_reps')}\n"
        f"  state_key   : {metadata.get('action_state_key')}\n"
        f"  needs_imu   : {obs_builder.needs_imu}\n"
        f"  decoder     : {decoder.kind}"
    )

    if args.sensor_check_only:
        print(
            f"[edge] sensor check passed via {args.transport}; "
            "no action chunk was installed or published",
            flush=True,
        )
        robot.close()
        return

    if args.rtc_max_delay_override is not None:
        srv = int(metadata["rtc_max_delay"])
        ovr = int(args.rtc_max_delay_override)
        print(
            f"  ⚠ Overriding rtc_max_delay: policy={srv} → client={ovr}. "
            f"Only meaningful if the weights were actually trained with RTC."
        )
        metadata = dict(metadata)
        metadata["rtc_max_delay"] = ovr

    reanchor = DeltaFrameReanchor(metadata, use_it=not args.no_delta_reanchor)

    # 5) Task list + key state machine.
    tasks = _load_task_list(args.prompt_yaml, args.task)
    task_holder = TaskHolder(tasks)
    key_sm = KeyStateMachine(
        task_holder=task_holder, auto_run=args.auto_run,
        grasp_stop_enabled=args.grasp_stop_enabled,
    )

    # 5.5) Human-in-the-loop: expose start/stop over HTTP (drives the same
    #      KeyStateMachine as the keyboard). Keyboard s/p still work alongside.
    hil_server = None
    if getattr(args, "human_in_loop", False):
        if args.mode != "rtc_chunk":
            print(
                "[ws] NOTE: --human_in_loop 的 prefix 语义(第一 chunk 无 prefix、"
                f"后续用 RTC prefix)只在 --mode rtc_chunk 下成立;当前 mode="
                f"{args.mode!r},HTTP /start /stop 仅做启停门控。"
            )
        hil_server = start_human_in_loop_server(
            key_sm, args.human_in_loop_host, args.human_in_loop_port,
        )

    # 6) Go.
    #    Upper-body: 在 run 前后各把 手臂+手+腰 复位到固定位姿 (镜像 A2 infer_a2_rtc.py
    #    的 robot.reset())。whole-body 不复位 —— 腿/骨盆在环, 单独 reset 上肢不安全。
    do_reset = (embodiment_kind == "upper_body"
                and not getattr(args, "no_reset", False)
                and not args.grasp_stop_enabled)
    reset_hand = getattr(args, "reset_hand_pose", "open") != "none"
    hand_pose = getattr(args, "reset_hand_pose", "open")
    if do_reset:
        # reset-start: 还没发过 cmd, 从 measured state 过渡。
        reset_upper_body_arm(
            robot,
            initial_pose["arm"],
            initial_pose["waist"],
            target_hand=initial_pose["hand"],
            reset_hand=False,
            from_last_cmd=False,
            label="reset-dataset-frame0",
        )
    # 'm' (while paused) swaps between standard and rtc_chunk without a
    # restart: each runner returns the mode to hand over to, and we rebuild the
    # other one. Runners are cheap — the policy and engines stay loaded.
    mode = args.mode
    try:
        while True:
            can_switch = (embodiment_kind == "upper_body"
                          and args.transport != "inproc")
            print(f"\n[ws] mode={mode}"
                  + ("  (按 p 暂停后可用 m 切换)" if can_switch else ""),
                  flush=True)
            next_mode = None
            if mode == "rtc_chunk":
                if embodiment_kind == "upper_body":
                    runner = AsyncRTCUpperBodyRunner(
                        policy_client, robot, obs_builder, decoder, reanchor,
                        task_holder, key_sm, args, metadata,
                        initial_pose=initial_pose,
                    )
                else:
                    runner = AsyncRTCChunkTrainRunner(
                        policy_client, robot, obs_builder, decoder, reanchor,
                        task_holder, key_sm, args, metadata,
                    )
                runner.run()
                next_mode = getattr(runner, "next_mode", None)
            else:
                if embodiment_kind == "upper_body":
                    next_mode = run_standard_upper_body(
                        policy_client, robot, obs_builder, decoder,
                        task_holder, key_sm, args, initial_pose=initial_pose,
                    )
                else:
                    run_standard(policy_client, robot, obs_builder, decoder,
                                 task_holder, key_sm, args)
            if not next_mode:
                break
            if not can_switch:
                print(f"[ws] 当前配置不支持切到 {next_mode}, 保持 {mode}")
                break
            if next_mode == "rtc_chunk" and int(metadata["rtc_max_delay"]) <= 0:
                print("[ws] 该 checkpoint 未用 train-time RTC 训练 "
                      "(rtc_max_delay=0), 保持 standard")
                break
            try:
                robot.cancel_chunk()
            except Exception:                             # noqa: BLE001
                pass
            mode = next_mode
            args.mode = mode
    except KeyboardInterrupt:
        pass
    finally:
        if do_reset and not (args.grasp_stop_enabled and
                             key_sm.status_snapshot()["grasp_state"] == "GRASP_CONFIRMED"):
            # reset-end: runner 退出时已 cancel_chunk, server _*_current 停在模型最后
            # 一帧 —— 从"上一条 cmd"平滑复位, 不读 measured state (避免跟踪误差先倒退)。
            reset_upper_body_arm(
                robot, UPPER_BODY_RESET_ARM_POS, UPPER_BODY_RESET_WAIST,
                reset_hand=reset_hand, hand_pose=hand_pose, from_last_cmd=True,
                label="reset-end",
            )
        key_sm.close()
        if hil_server is not None:
            try:
                hil_server.shutdown()
            except Exception:
                pass
        if inproc_robot is not None:
            try:
                inproc_robot.close()
            except Exception as exc:
                print(f"[inproc] cleanup warning: {exc}", flush=True)


if __name__ == "__main__":
    main()
