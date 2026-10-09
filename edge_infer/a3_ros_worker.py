#!/usr/bin/env python3
"""ROS-side worker for the A3 edge inference loop.

Runs A3ServerNode in its OWN process so that its Python callbacks — the camera
topics plus the 150Hz InterpolationPublisher loop — never contend for the
inference process's GIL. In-process spinning measured ~340-400ms per
get_action against ~203ms standalone; this removes that contention while
keeping observation fetch at memory speed (no HTTP, no JPEG).

Observations are published into POSIX shared memory. Commands arrive on stdin
as length-prefixed pickles, replies go out on stdout; all logging goes to
stderr so it can never corrupt the channel.

Run with the SYSTEM python (/usr/bin/python3): that is the interpreter ROS
jazzy's rclpy / cv_bridge extensions are built for. The parent stays in the
torch_build venv. Separate processes, so their numpy versions never clash.

Launched by infer_a3_edge.py --transport subproc; not meant to be run by hand.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
from multiprocessing import shared_memory

import numpy as np

# ---------------------------------------------------------------------------
# Shared-memory layout (imported by BOTH sides — keep ROS imports out of the
# module body so the parent can import this without pulling in rclpy).
# ---------------------------------------------------------------------------

# Frame buffer is sized for the largest sensor we might see (1080p BGR).
MAX_FRAME_BYTES = 1920 * 1080 * 3

# meta: int64
#   [0] heartbeat (child increments; parent uses it as a liveness check)
#   [1] chunk_remaining_steps (0 = idle) — lets the parent wait without
#       blocking the child's command loop, so 'p' can interrupt a chunk
#   [2] camera count
#   [3] arm chunk played index x1000 (-1 = no chunk playing). RTC reads this
#       every tick to correct its virtual clock, so it rides the state mirror
#       instead of a round trip.
#   [4 + 4*i ...] per camera: seq, h, w, channels
#   [4] last commanded waist height x1e6 (INT64_MIN = unknown). The waist
#       column's height is never predicted by these checkpoints, so the client
#       re-sends whatever is already commanded instead of zeroing it.
META_HEAD = 5
META_PER_CAM = 4
META_PLAYED_IDX = 3
META_WAIST_H = 4
PLAYED_SCALE = 1000
HEIGHT_SCALE = 1_000_000
HEIGHT_UNKNOWN = -(2 ** 62)

# state: float64, [0] = seq, then per joint group:
#   length + MAX position + MAX velocity + MAX effort
# followed by arm/wb progress, snapshot timestamp and pelvis/torso IMU blocks.
#
# The original upper-body worker mirrored positions only.  Whole-body models
# may consume body_vel and pelvis IMU, so the shared-memory contract must carry
# the complete policy-visible state.  One seqlock covers joints + IMU +
# progress; the whole-body ``snapshot`` command writes it from the same cache
# snapshot used for the camera buffers.
JOINT_LAYOUT = (("arm", 14), ("hand", 20), ("waist", 6), ("leg", 12), ("neck", 2))
JOINT_FIELDS = ("position", "velocity", "effort")
IMU_LAYOUT = ("pelvis", "torso")
IMU_ORIENTATION_DIM = 4
IMU_VECTOR_DIM = 3
IMU_BLOCK_SIZE = 1 + IMU_ORIENTATION_DIM + 3 * IMU_VECTOR_DIM + 1


def meta_size(ncams: int) -> int:
    return META_HEAD + META_PER_CAM * ncams


def joints_size() -> int:
    return state_offsets()["size"]


def joint_offsets() -> dict:
    """group -> (length, position, velocity, effort, max_dim)."""
    out, cursor = {}, 1
    for name, dim in JOINT_LAYOUT:
        out[name] = (
            cursor,
            cursor + 1,
            cursor + 1 + dim,
            cursor + 1 + 2 * dim,
            dim,
        )
        cursor += 1 + len(JOINT_FIELDS) * dim
    return out


def state_offsets() -> dict:
    cursor = 1 + sum(1 + len(JOINT_FIELDS) * dim for _, dim in JOINT_LAYOUT)
    out = {
        "progress_arm": cursor,
        "progress_wb": cursor + 1,
        "timestamp": cursor + 2,
    }
    cursor += 3
    for name in IMU_LAYOUT:
        out[f"imu_{name}"] = cursor
        cursor += IMU_BLOCK_SIZE
    out["size"] = cursor
    return out


def _unpack_imu(buf: np.ndarray, start: int):
    if float(buf[start]) < 0.5:
        return None
    cursor = start + 1
    orientation = buf[cursor:cursor + 4].tolist()
    cursor += 4
    angular = buf[cursor:cursor + 3].tolist()
    cursor += 3
    linear = buf[cursor:cursor + 3].tolist()
    cursor += 3
    gravity = buf[cursor:cursor + 3].tolist()
    cursor += 3
    return {
        "orientation_xyzw": orientation,
        "angular_velocity": angular,
        "linear_acceleration": linear,
        "gravity_dir": gravity,
        "timestamp": float(buf[cursor]),
    }


def unpack_state(buf: np.ndarray) -> dict:
    """Shared float64 block -> atomic joints/IMU/progress snapshot."""
    joints = {}
    for name, (len_i, pos_i, vel_i, eff_i, dim) in joint_offsets().items():
        n = max(0, min(int(buf[len_i]), dim))
        if n <= 0:
            joints[name] = None
            continue
        joints[name] = {
            "position": buf[pos_i:pos_i + n].tolist(),
            "velocity": buf[vel_i:vel_i + n].tolist(),
            "effort": buf[eff_i:eff_i + n].tolist(),
        }
    offsets = state_offsets()
    arm = float(buf[offsets["progress_arm"]])
    wb = float(buf[offsets["progress_wb"]])
    return {
        "joints": joints,
        "imu": {
            "pelvis": _unpack_imu(buf, offsets["imu_pelvis"]),
            "torso": _unpack_imu(buf, offsets["imu_torso"]),
        },
        "chunk_progress": {
            "arm": arm if np.isfinite(arm) else None,
            "wb": wb if np.isfinite(wb) else None,
            "eef": None,
            "hand": None,
        },
        "timestamp": float(buf[offsets["timestamp"]]),
    }


def unpack_joints(buf: np.ndarray) -> dict:
    """Shared float64 block -> the dict shape /get_joint_states returns."""
    return unpack_state(buf)["joints"]


# ---------------------------------------------------------------------------
# stdin/stdout framing
# ---------------------------------------------------------------------------

def send_msg(stream, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def recv_msg(stream):
    header = stream.read(8)
    if not header or len(header) < 8:
        return None
    (size,) = struct.unpack("<Q", header)
    body = stream.read(size)
    if body is None or len(body) < size:
        return None
    return pickle.loads(body)


# ---------------------------------------------------------------------------
# Child main
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[ros-worker] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    import argparse
    import threading
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--shm-meta", required=True)
    ap.add_argument("--shm-joints", required=True)
    ap.add_argument("--shm-cams", required=True, help="comma-separated shm names")
    ap.add_argument("--cameras", required=True, help="comma-separated camera names")
    ap.add_argument("--hand-kind", default="hand", choices=["hand", "gripper"])
    ap.add_argument("--embodiment", default="upper_body",
                    choices=["upper_body", "whole_body"])
    ap.add_argument("--robot-ip", default="192.168.100.100")
    ap.add_argument("--roboiface-dir", default="/agibot/edge_deploy/RoboInterface")
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c]
    cam_shm_names = [c for c in args.shm_cams.split(",") if c]
    if len(cameras) != len(cam_shm_names):
        _log("camera/shm count mismatch")
        return 2

    if args.roboiface_dir not in sys.path:
        sys.path.insert(0, args.roboiface_dir)   # for config.OMNIHAND_* below

    # The parent process carries the model venv and may also inherit another
    # aimdk namespace from the system overlay.  Force this deployment's ROS
    # packages to the front before importing a3_server.
    ros_server_dir = os.environ.get("A3_ROS_SERVER_DIR", "").strip()
    if ros_server_dir:
        local_ros_paths = [
            os.path.join(ros_server_dir, "python_deps"),
            os.path.join(ros_server_dir, "_pb_gen"),
            os.path.join(
                ros_server_dir,
                "install/a3_server/lib/python3.12/site-packages",
            ),
            os.path.join(
                ros_server_dir,
                "install/ros2_plugin_proto/lib/python3.12/site-packages",
            ),
            os.path.join(
                ros_server_dir,
                "install/joint_msgs/lib/python3.12/site-packages",
            ),
        ]
        for path in reversed(local_ros_paths):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from a3_server.joint_config import HandKind
    from a3_server.server_node import CAMERA_TOPICS, A3ServerNode

    meta_shm = shared_memory.SharedMemory(name=args.shm_meta)
    joints_shm = shared_memory.SharedMemory(name=args.shm_joints)
    cam_shms = [shared_memory.SharedMemory(name=n) for n in cam_shm_names]
    meta = np.ndarray((meta_size(len(cameras)),), dtype=np.int64, buffer=meta_shm.buf)
    joints_buf = np.ndarray((joints_size(),), dtype=np.float64, buffer=joints_shm.buf)
    cam_views = [np.ndarray((MAX_FRAME_BYTES,), dtype=np.uint8, buffer=s.buf)
                 for s in cam_shms]
    cam_index = {name: i for i, name in enumerate(cameras)}

    # A3ServerNode gates every publisher on this import; without it the node
    # half-initializes and dies later on a missing attribute. Fail here with a
    # message that names the real problem instead.
    try:
        from ros2_plugin_proto.msg import RosMsgWrapper  # noqa: F401
    except ImportError as exc:
        _log(f"FATAL: ros2_plugin_proto not importable ({exc}). The worker's "
             f"PYTHONPATH must include the colcon workspace "
             f"($A3_ROS_SERVER_DIR/install/...) — source the plugin setup "
             f"before launching.")
        return 3

    required_pb_symbols = (
        (
            "aimdk.protocol.ta.ta_whole_body_state_pb2",
            "TaWholeBodyStateChannel",
        ),
        (
            "aimdk.protocol.ta.wbc_reference_window_pb2",
            "TaWholeBodyReferenceWindow",
        ),
        (
            "aimdk.protocol.ta.ta_whole_body_command_pb2",
            "TaWholeBodyCommand",
        ),
    )
    try:
        import importlib

        loaded_from = []
        for module_name, symbol_name in required_pb_symbols:
            module = importlib.import_module(module_name)
            getattr(module, symbol_name)
            loaded_from.append(f"{module_name}={module.__file__}")
        _log("protobuf preflight ready: " + "; ".join(loaded_from))
    except (ImportError, AttributeError) as exc:
        _log(
            f"FATAL: required whole-body protobuf symbol is unavailable: {exc}; "
            f"A3_ROS_SERVER_DIR={ros_server_dir!r}; sys.path={sys.path!r}"
        )
        return 4

    # Ctrl-C is handled by the parent; keep rclpy's handler from racing us.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = A3ServerNode(robot_ip=args.robot_ip,
                        hand_kind=HandKind.HAND if args.hand_kind == "hand"
                        else HandKind.GRIPPER)
    if args.embodiment == "whole_body" and not node.enable_wb_state_source():
        _log("FATAL: whole_body_state subscription could not be created; "
             "check ros2_plugin_proto and TaWholeBodyStateChannel pb2")
        return 5

    # Drop the camera topics the checkpoint never reads: their callbacks would
    # decode frames nobody consumes.
    keep = {CAMERA_TOPICS[c].lstrip("/") for c in cameras if c in CAMERA_TOPICS}
    all_cams = {t.lstrip("/") for t in CAMERA_TOPICS.values()}
    subs = [(s, s.topic_name.lstrip("/")) for s in node.subscriptions]
    dropped = 0
    for sub, topic in subs:
        if any(topic.endswith(t) for t in all_cams) and \
           not any(topic.endswith(t) for t in keep):
            node.destroy_subscription(sub)
            dropped += 1
    _log(f"cameras kept={cameras} dropped={dropped}")

    # Frames go straight into shared memory. The subscriptions call
    # self._camera_cb(name, msg), looked up at call time, so shadowing the
    # instance attribute is enough to reroute them.
    from cv_bridge import CvBridge
    bridge = CvBridge()
    lock = threading.Lock()

    def write_frame(idx, img):
        flat = np.ascontiguousarray(img).reshape(-1)
        if flat.size > MAX_FRAME_BYTES:
            _log(f"{cameras[idx]} frame {img.shape} exceeds buffer, dropping")
            return
        base = META_HEAD + META_PER_CAM * idx
        with lock:
            # seqlock: odd while writing, even when stable.
            meta[base] += 1
            cam_views[idx][:flat.size] = flat
            meta[base + 1], meta[base + 2] = img.shape[0], img.shape[1]
            meta[base + 3] = img.shape[2] if img.ndim == 3 else 1
            meta[base] += 1

    def camera_cb(name, msg):
        idx = cam_index.get(name)
        if idx is None:
            return
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:                      # noqa: BLE001
            _log(f"decode failed for {name}: {exc}")
            return
        if args.embodiment == "whole_body":
            # Keep the callback lean.  The command loop copies this immutable
            # frame reference into the snapshot SHM only when inference asks
            # for an observation, matching the HTTP endpoint's snapshot point.
            with node._cam_lock:
                node.latest_cameras[name] = img
        else:
            write_frame(idx, img)

    node._camera_cb = camera_cb

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, name="ros-spin", daemon=True)
    spin.start()

    offsets = joint_offsets()
    state_meta = state_offsets()

    def write_imu(cache, start):
        joints_buf[start] = 1.0 if cache else 0.0
        if not cache:
            joints_buf[start + 1:start + IMU_BLOCK_SIZE] = 0.0
            return
        cursor = start + 1
        fields = (
            ("orientation_xyzw", 4),
            ("angular_velocity", 3),
            ("linear_acceleration", 3),
            ("gravity_dir", 3),
        )
        for field, dim in fields:
            values = np.asarray(cache.get(field) or [0.0] * dim,
                                dtype=np.float64).reshape(-1)
            joints_buf[cursor:cursor + dim] = 0.0
            n = min(values.size, dim)
            if n:
                joints_buf[cursor:cursor + n] = values[:n]
            cursor += dim
        joints_buf[cursor] = float(cache.get("timestamp") or 0.0)

    def write_state_snapshot(sources, pelvis_imu=None, torso_imu=None,
                             arm_played=None, wb_played=None,
                             snapshot_timestamp=None):
        """Publish one complete policy-visible state under a single seqlock."""
        seq = int(joints_buf[0])
        if seq % 2:
            seq += 1
        joints_buf[0] = seq + 1
        for name, cache in sources.items():
            len_i, pos_i, vel_i, eff_i, dim = offsets[name]
            cache = cache or {}
            pos = np.asarray(cache.get("position") or [], dtype=np.float64).reshape(-1)
            n = min(pos.size, dim)
            joints_buf[len_i] = n
            for field, start in (("position", pos_i), ("velocity", vel_i),
                                 ("effort", eff_i)):
                values = np.asarray(cache.get(field) or [],
                                    dtype=np.float64).reshape(-1)
                joints_buf[start:start + dim] = 0.0
                count = min(values.size, n)
                if count:
                    joints_buf[start:start + count] = values[:count]
        joints_buf[state_meta["progress_arm"]] = (
            float(arm_played) if arm_played is not None else np.nan
        )
        joints_buf[state_meta["progress_wb"]] = (
            float(wb_played) if wb_played is not None else np.nan
        )
        joints_buf[state_meta["timestamp"]] = float(
            snapshot_timestamp if snapshot_timestamp is not None else time.time()
        )
        write_imu(pelvis_imu, state_meta["imu_pelvis"])
        write_imu(torso_imu, state_meta["imu_torso"])
        joints_buf[0] = seq + 2

    def capture_whole_body_snapshot():
        """Mirror the strict HTTP snapshot without JPEG/base64/JSON.

        Cache objects and camera arrays are replaced, never mutated.  Taking
        their references while holding the same locks/order as
        /get_observation_with_progress defines the atomic instant; the large
        SHM memcpy happens after releasing the interpolator lock so the 50 Hz
        reference-window publisher is not stalled.
        """
        with node.interp_pub._lock:
            progress = node.interp_pub._chunk_played_idx_nolock()
            with node._cam_lock:
                frames = [node.latest_cameras.get(name) for name in cameras]
            with node._wb_state_lock:
                sources = {
                    "leg": node.latest_wb_leg_joints,
                    "waist": node.latest_wb_waist_joints,
                    "neck": node.latest_wb_neck_joints,
                    "arm": node.latest_wb_arm_joints,
                    "hand": node.latest_hand_joints,
                }
                pelvis = node.latest_wb_pelvis_imu
                torso = node.latest_wb_torso_imu
            snapshot_timestamp = time.time()

        write_state_snapshot(
            sources,
            pelvis_imu=pelvis,
            torso_imu=torso,
            arm_played=progress.get("arm"),
            wb_played=progress.get("wb"),
            snapshot_timestamp=snapshot_timestamp,
        )
        for idx, frame in enumerate(frames):
            if frame is not None:
                write_frame(idx, frame)
        played = progress.get("wb")
        meta[META_PLAYED_IDX] = (
            int(float(played) * PLAYED_SCALE) if played is not None else -1
        )
        return snapshot_timestamp

    def publish_state():
        """Mirror upper-body state or whole-body live progress at 200Hz."""
        while True:
            if args.embodiment == "upper_body":
                sources = {
                    "arm": node.latest_arm_joints,
                    "hand": node.latest_hand_joints,
                    "waist": node.latest_waist_joints,
                    "leg": node.latest_leg_joints,
                    "neck": node.latest_neck_joints,
                }
                try:
                    arm_played = node.interp_pub.arm_chunk_played_idx()
                except Exception:                      # noqa: BLE001
                    arm_played = None
                write_state_snapshot(sources, arm_played=arm_played)
                try:
                    meta[1] = int(node.interp_pub.chunk_remaining_steps())
                except Exception:                      # noqa: BLE001
                    meta[1] = 0
                meta[META_PLAYED_IDX] = (
                    int(arm_played * PLAYED_SCALE)
                    if arm_played is not None else -1
                )
            else:
                # Exact state/camera snapshots are request-driven.  Only the
                # virtual-clock progress needs a continuously fresh mirror.
                try:
                    played = node.interp_pub.wb_played_idx()
                    meta[META_PLAYED_IDX] = (
                        int(played * PLAYED_SCALE) if played is not None else -1
                    )
                except Exception:                      # noqa: BLE001
                    meta[META_PLAYED_IDX] = -1
                meta[1] = 0
            try:
                cur = node.interp_pub._waist_current
                meta[META_WAIST_H] = (int(float(cur[3]) * HEIGHT_SCALE)
                                      if cur is not None and len(cur) >= 4
                                      else HEIGHT_UNKNOWN)
            except Exception:                          # noqa: BLE001
                meta[META_WAIST_H] = HEIGHT_UNKNOWN
            meta[0] += 1
            time.sleep(0.005)

    threading.Thread(target=publish_state, name="state-pub", daemon=True).start()

    def to_actuator(rows: np.ndarray) -> np.ndarray:
        """20D radians -> raw actuator counts, per 10D half (dex hand only)."""
        from config import OMNIHAND_LEFT, OMNIHAND_RIGHT
        out = np.empty_like(rows, dtype=np.float64)
        for i, row in enumerate(rows):
            out[i] = np.asarray(
                list(OMNIHAND_LEFT.radians_to_actuator(list(row[:10])))
                + list(OMNIHAND_RIGHT.radians_to_actuator(list(row[10:]))),
                dtype=np.float64,
            )
        return out

    def rows(value, width: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        return arr.reshape(1, width) if arr.ndim == 1 else arr

    def set_wb_emit_mode(emit_mode: str, ta_cmd_hz=None) -> None:
        """Mirror a3_server /send_chunk's wire-path selection."""
        emit_mode = str(emit_mode or node._wb_emit_mode or "ta_cmd")
        if emit_mode not in ("ta_cmd", "reference_window"):
            emit_mode = "ta_cmd"
        if emit_mode == "ta_cmd":
            if ta_cmd_hz is not None:
                node.interp_pub.set_ta_cmd_emit(
                    float(ta_cmd_hz), node._publish_ta_whole_body_command)
                node._ta_cmd_hz = float(ta_cmd_hz)
            if node._ref_window_timer is not None:
                node._ref_window_timer.cancel()
                node._ref_window_timer = None
        else:
            node.interp_pub.set_ta_cmd_emit(0, None)
            if (node._ref_window_timer is None
                    and node._gr00t_ref_window_pub is not None):
                node._ref_window_timer = node.create_timer(
                    1.0 / node.REF_WINDOW_PUBLISH_HZ,
                    node._publish_gr00t_reference_window,
                )

    def install_whole_body_chunk(chunk, chunk_fps: float, options: dict) -> dict:
        """Validate and atomically install the exact HTTP whole-body payload."""
        arrays = {}
        for name, width in (("arm", 14), ("leg", 12), ("waist", 3),
                            ("pelvis_quat_wxyz", 4)):
            if chunk.get(name) is None:
                raise ValueError(
                    "whole-body chunk requires arm+leg+waist+pelvis_quat_wxyz"
                )
            arr = np.asarray(chunk[name], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != width:
                raise ValueError(f"{name} expected (H, {width}), got {arr.shape}")
            arrays[name] = arr
        horizon = arrays["arm"].shape[0]
        if horizon <= 0 or any(arr.shape[0] != horizon for arr in arrays.values()):
            raise ValueError("whole-body chunk horizons must match and be non-empty")

        hand = None
        if chunk.get("hand") is not None:
            hand = np.asarray(chunk["hand"], dtype=np.float64)
            if hand.ndim != 2 or hand.shape != (horizon, hand_dim):
                raise ValueError(
                    f"hand expected ({horizon}, {hand_dim}), got {hand.shape}"
                )
            hand_value = str(chunk.get("hand_value", "raw")).lower()
            if hand_value == "rad" and hand_dim == 20:
                hand = to_actuator(hand)
            elif hand_value != "raw":
                raise ValueError(
                    "whole-body hand must use raw actuator values or 20D rad"
                )

        # Preserve the HTTP route's D2 rule: neck is sampled once at chunk
        # load time and held over the horizon.
        neck_joints = node.latest_neck_joints
        if neck_joints is not None and len(neck_joints.get("position", [])) >= 2:
            neck_hold = np.asarray(neck_joints["position"][:2], dtype=np.float64)
        else:
            neck_hold = np.zeros(2, dtype=np.float64)

        chunk_fps = float(chunk_fps)
        source_fps = float(options.get("source_fps", chunk_fps))
        if source_fps <= 0:
            source_fps = chunk_fps
        s_used_local = options.get("s_used_local")
        s_used_local = int(s_used_local) if s_used_local is not None else None
        s_used_wire = options.get("s_used_local_wire")
        if s_used_wire is not None:
            s_used_wire = int(s_used_wire)
        elif s_used_local is not None:
            s_used_wire = int(np.floor(s_used_local * chunk_fps / source_fps))

        set_wb_emit_mode(options.get("emit_mode"), options.get("ta_cmd_hz"))
        common = dict(
            leg=arrays["leg"],
            waist=arrays["waist"],
            arm=arrays["arm"],
            pelvis_quat=arrays["pelvis_quat_wxyz"],
            neck_hold=neck_hold,
            hand=hand,
            hand_effort=chunk.get("hand_effort"),
            chunk_fps=chunk_fps,
            chunk_id=int(options.get("chunk_id", -1) or -1),
        )
        if s_used_local is None:
            return node.interp_pub.set_whole_body_chunk(**common)
        return node.interp_pub.swap_whole_body_chunk_atomic(
            **common,
            s_used_local=s_used_wire,
            adaptive_transition=bool(options.get("adaptive_transition", False)),
            source_fps=source_fps,
        )

    hand_dim = 2 if args.hand_kind == "gripper" else 20
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    reply_lock = threading.Lock()   # the wait thread also writes replies
    send_msg(stdout, ("ready", cameras))
    _log("ready")

    while True:
        msg = recv_msg(stdin)
        if msg is None:
            break
        kind = msg[0]
        try:
            if kind == "snapshot":
                if args.embodiment != "whole_body":
                    raise RuntimeError("snapshot command requires whole_body worker")
                timestamp = capture_whole_body_snapshot()
                with reply_lock:
                    send_msg(stdout, ("ok", timestamp))
            elif kind == "wb_chunk":
                if args.embodiment != "whole_body":
                    raise RuntimeError("wb_chunk command requires whole_body worker")
                _, chunk, chunk_fps, options = msg
                resp = install_whole_body_chunk(chunk, chunk_fps, options)
                with reply_lock:
                    send_msg(stdout, ("ok", dict(resp or {})))
            elif kind == "wait":
                # Mirrors a3_server's /send_chunk?wait=true: the side owning
                # interp_pub blocks, like the reference client's server-side
                # wait. Polling a 5ms-stale shm mirror from the parent returned
                # early and stacked the next chunk onto one still playing (exec
                # alternating 3ms / 1337ms).
                #
                # Run it off the command loop: over HTTP a cancel arrives on its
                # own connection, but here every command shares one pipe, so a
                # blocking wait would delay the 'p' cancel until the chunk ends.
                _, settle_ms, timeout_ms = msg

                def _wait_then_reply(settle_ms=settle_ms, timeout_ms=timeout_ms):
                    try:
                        node.interp_pub.wait_chunk_done(
                            settle_sec=max(0.0, float(settle_ms)) / 1000.0,
                            timeout_sec=max(0.1, float(timeout_ms)) / 1000.0,
                        )
                    except Exception as exc:            # noqa: BLE001
                        _log(f"wait failed: {exc}")
                    with reply_lock:
                        send_msg(stdout, ("ok",))

                threading.Thread(target=_wait_then_reply,
                                 name="chunk-wait", daemon=True).start()
                continue    # reply comes from that thread
            elif kind == "chunk":
                # Same route the reference client uses (a3_server /send_chunk
                # mode=upper_body -> swap_upper_chunk_atomic): arm, hand and
                # waist are installed inside ONE lock. Calling the three
                # set_*_chunk helpers separately installs them at different
                # instants, so waist ends up anchored to a different moment
                # than arm. With s_used_local=None the swap installs without
                # slicing, which is exactly the non-RTC semantics.
                _, chunk, chunk_fps = msg
                hand = chunk.get("hand")
                if hand is not None:
                    hand = rows(hand, hand_dim)
                    if chunk.get("hand_value") == "rad" and hand_dim == 20:
                        hand = to_actuator(hand)
                node.interp_pub.swap_upper_chunk_atomic(
                    arm=(rows(chunk["arm"], 14)
                         if chunk.get("arm") is not None else None),
                    hand=hand,
                    waist=(rows(chunk["waist"], 4)
                           if chunk.get("waist") is not None else None),
                    chunk_fps=float(chunk_fps),
                    s_used_local=None,
                )
                # Report remaining steps with the ack. The parent's
                # wait_for_done polls a 5ms-old mirror, so without this it can
                # sample a stale 0 and return before the chunk starts — the
                # next chunk then lands on top of one still playing.
                try:
                    remaining = int(node.interp_pub.chunk_remaining_steps())
                except Exception:                       # noqa: BLE001
                    remaining = 0
                meta[1] = remaining
                with reply_lock:
                    send_msg(stdout, ("ok", remaining))
            elif kind == "swap":
                # RTC path: one lock critical section in interp_pub reads the
                # old chunk's played index and slices arm/hand/waist by the
                # same actual_delay. Doing it here (not parent-side) is what
                # keeps the swap atomic — no IPC gap between read and install.
                _, chunk, chunk_fps, s_used_local = msg
                hand = chunk.get("hand")
                if hand is not None:
                    hand = rows(hand, hand_dim)
                    if chunk.get("hand_value") == "rad" and hand_dim == 20:
                        hand = to_actuator(hand)
                resp = node.interp_pub.swap_upper_chunk_atomic(
                    arm=(rows(chunk["arm"], 14)
                         if chunk.get("arm") is not None else None),
                    hand=hand,
                    waist=(rows(chunk["waist"], 4)
                           if chunk.get("waist") is not None else None),
                    chunk_fps=float(chunk_fps),
                    s_used_local=s_used_local,
                )
                with reply_lock:
                    send_msg(stdout, ("ok", int((resp or {}).get("actual_delay", 0))))
            elif kind == "cancel":
                node.interp_pub.cancel_chunk()
                with reply_lock:
                    send_msg(stdout, ("ok",))
            elif kind == "ping":
                with reply_lock:
                    send_msg(stdout, ("ok",))
            elif kind == "set_speed":
                node.interp_pub.set_send_fps(float(msg[1]))
                with reply_lock:
                    send_msg(stdout, ("ok",))
            elif kind == "shutdown":
                with reply_lock:
                    send_msg(stdout, ("ok",))
                break
            else:
                with reply_lock:
                    send_msg(stdout, ("error", f"unknown command {kind!r}"))
        except Exception as exc:                        # noqa: BLE001
            _log(f"command {kind!r} failed: {exc}")
            with reply_lock:
                send_msg(stdout, ("error", str(exc)))

    _log("shutting down")
    try:
        node.interp_pub.cancel_chunk()
    except Exception:                                   # noqa: BLE001
        pass
    executor.shutdown(timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()
    for shm in (*cam_shms, joints_shm, meta_shm):
        shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
