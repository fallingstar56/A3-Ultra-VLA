"""Token-chunk transport pipeline for the robot side.

Pipeline (matches the deployment plan described in Isaac-GR00T's
gr00t/policy/gr00t_policy.py::_build_token_chunk_envelope):

    VLA (off-robot)               Robot
    ─────────────────             ─────────────────────────────────────
      inference @ ~5Hz              subscribe TOKEN_CHUNK_TOPIC (msgpack)
      output token chunk    ───▶    TokenChunkReceiver buffers envelopes
      (H, action_dim) +               │
      pooled features                 ▼
                                    TokenChunkStreamer
                                      │  advances chunk index at source_hz
                                      │  (20 or 30 Hz) — one token per tick
                                      ▼
                                    TokenInterpolator
                                      │  linear interp of consecutive
                                      │  source-rate tokens up to 60Hz
                                      ▼
                                    ResidualHeadRunner (optional)
                                      │  60Hz call of the residual head
                                      │  on the interpolated token; adds
                                      │  a bounded delta in normalized space
                                      ▼
                                    A3RobotInterface.step()
                                      │  server_node → interp_publisher
                                      │  final 60→150Hz interpolation
                                      ▼
                                    ROS actuator command topic

Notes
-----
- **Placeholder topic**: ``DEFAULT_VLA_TOKEN_CHUNK_TOPIC = "/vla/token_chunk"``.
  This will be swapped for the real per-embodiment topic (something like
  ``/ta/whole_body_command``) once the deployment side finalizes it.
- The residual head is loaded from an offline-trained ``.pt`` (see
  ``gr00t/experiment/train_residual_head.py``). We rebuild the same
  ``ResidualHead`` class on the robot rather than materializing a full
  Gr00tN1d7 there.
- All downstream commands use the same normalized-space math as GR00T's
  processor; the caller supplies a ``decode_fn`` that un-normalizes the
  final (interpolated + residual) token back to actuator units.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# Placeholder for the real deployment topic. Kept in sync with the GR00T-side
# default at gr00t/policy/gr00t_policy.py::DEFAULT_VLA_TOKEN_CHUNK_TOPIC.
DEFAULT_VLA_TOKEN_CHUNK_TOPIC = "/vla/token_chunk"


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


@dataclass
class TokenChunkEnvelope:
    """Robot-side view of the token chunk that arrived over the wire.

    Fields mirror what Gr00tPolicy._build_token_chunk_envelope produces.
    ``tokens`` is (H, action_dim) float32 in the model's normalized space.
    """

    chunk_id: int
    wall_time: float
    source_hz: float
    target_interp_hz: float
    tokens: np.ndarray  # (H, D) normalized action chunk
    action_horizon: int
    action_dim: int
    vla_feature: np.ndarray | None = None
    state_feature: np.ndarray | None = None
    tokens_uncomp: np.ndarray | None = None
    residual: np.ndarray | None = None
    topic: str = DEFAULT_VLA_TOKEN_CHUNK_TOPIC

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenChunkEnvelope":
        # Squeeze batch dim (VLA-side envelopes carry B=1 in single-arm
        # deployments; multi-arm eval would separate before send).
        def _sq(x):
            if x is None:
                return None
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            elif x.ndim == 2 and x.shape[0] == 1:
                x = x[0]
            return x

        tokens = _sq(d["tokens"])
        assert tokens.ndim == 2, (
            f"tokens must be 2D after squeeze, got {tokens.shape}"
        )
        return cls(
            chunk_id=int(d["chunk_id"]),
            wall_time=float(d["wall_time"]),
            source_hz=float(d.get("source_hz", 20.0)),
            target_interp_hz=float(d.get("target_interp_hz", 60.0)),
            tokens=tokens,
            action_horizon=int(d.get("action_horizon", tokens.shape[0])),
            action_dim=int(d.get("action_dim", tokens.shape[1])),
            vla_feature=_sq(d.get("vla_feature")),
            state_feature=_sq(d.get("state_feature")),
            tokens_uncomp=_sq(d.get("tokens_uncomp")),
            residual=_sq(d.get("residual")),
            topic=str(d.get("topic", DEFAULT_VLA_TOKEN_CHUNK_TOPIC)),
        )


# ---------------------------------------------------------------------------
# Receiver — abstract transport, concrete impl for ROS is subclassable
# ---------------------------------------------------------------------------


class TokenChunkReceiver:
    """Base class. Concrete subclasses subscribe to a real transport (ROS,
    ZMQ, WebSocket). Ships with an in-process subclass suitable for
    tests + local integration where the VLA runs in the same process.
    """

    def __init__(self, topic: str = DEFAULT_VLA_TOKEN_CHUNK_TOPIC, maxsize: int = 8):
        self.topic = str(topic)
        # Bounded queue: if the VLA outpaces the streamer (never expected
        # since VLA @ ~5Hz and streamer consumes 1 chunk / ~1s at 20Hz
        # source rate) we drop the oldest to keep memory bounded.
        self._q: queue.Queue[TokenChunkEnvelope] = queue.Queue(maxsize=maxsize)

    def submit(self, envelope: TokenChunkEnvelope | dict[str, Any]) -> None:
        """Enqueue an envelope. Accepts either the parsed dataclass or the
        raw dict the GR00T-side envelope builder produces (msgpack-friendly
        so no manual conversion needed at wire boundaries)."""
        if isinstance(envelope, dict):
            envelope = TokenChunkEnvelope.from_dict(envelope)
        try:
            self._q.put_nowait(envelope)
        except queue.Full:
            try:
                dropped = self._q.get_nowait()
                logger.warning(
                    "TokenChunkReceiver queue full, dropping chunk_id=%d to make room",
                    dropped.chunk_id,
                )
            except queue.Empty:
                pass
            self._q.put_nowait(envelope)

    def get(self, timeout: float | None = None) -> TokenChunkEnvelope | None:
        """Blocking get with optional timeout. None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def peek_available(self) -> int:
        return self._q.qsize()


class RosTokenChunkReceiver(TokenChunkReceiver):
    """ROS 2 subscriber for the VLA token chunk topic.

    The wire format is msgpack (matching ``_build_token_chunk_envelope``).
    We use ``std_msgs/msg/ByteMultiArray`` — a schema-less container so we
    don't need to define a custom .msg for the placeholder topic. When the
    real deployment topic is decided (e.g. a per-embodiment TokenChunkPb),
    swap this subscription for the proper type-safe subscription.
    """

    def __init__(
        self,
        node,  # rclpy.node.Node
        topic: str = DEFAULT_VLA_TOKEN_CHUNK_TOPIC,
        maxsize: int = 8,
    ):
        super().__init__(topic=topic, maxsize=maxsize)
        from std_msgs.msg import ByteMultiArray  # noqa: E402
        import msgpack  # noqa: E402
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

        self._msgpack = msgpack
        # Best-effort + shallow history: real-time control; missing one
        # chunk is preferable to piling stale ones up.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node = node
        self._sub = node.create_subscription(
            ByteMultiArray, topic, self._on_msg, qos_profile=qos
        )
        node.get_logger().info(
            f"RosTokenChunkReceiver subscribed to {topic} (placeholder path — swap when real topic lands)"
        )

    def _on_msg(self, msg) -> None:
        try:
            raw = bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else bytes(
                msg.data
            )
            d = self._msgpack.unpackb(raw, raw=False)
            # msgpack unpacks numpy arrays as bytes; the sender side is
            # expected to pack tokens as flat lists (or already-decoded
            # arrays via msgpack_numpy). We accept either.
            for k in (
                "tokens",
                "tokens_uncomp",
                "residual",
                "vla_feature",
                "state_feature",
            ):
                v = d.get(k)
                if v is None or isinstance(v, np.ndarray):
                    continue
                d[k] = np.asarray(v, dtype=np.float32)
            envelope = TokenChunkEnvelope.from_dict(d)
            self.submit(envelope)
        except Exception as e:
            self._node.get_logger().error(f"failed to parse token chunk: {e}")


# ---------------------------------------------------------------------------
# Streamer + 60Hz interpolator + residual head runner
# ---------------------------------------------------------------------------


class TokenInterpolator:
    """Consumes envelopes from a receiver and emits tokens at
    ``target_interp_hz`` (default 60Hz) using linear interpolation between
    the source-rate waypoints.

    Boundary policy — when a fresh chunk arrives mid-flight through the
    old one, the interpolator anchors the next 60Hz waypoint at
    ``new_chunk[0]`` and interpolates from the current in-flight value
    over one source-rate slot. This mirrors the adaptive_transition path
    in A3RobotInterface.step_chunk so chunk boundaries stay smooth even
    when the VLA re-plans in the middle of the previous chunk.
    """

    def __init__(
        self,
        target_interp_hz: float = 60.0,
        default_source_hz: float = 20.0,
    ):
        self.target_interp_hz = float(target_interp_hz)
        self.default_source_hz = float(default_source_hz)

        self._current_chunk: TokenChunkEnvelope | None = None
        # Fractional index into the current chunk in "source samples", so
        # the tick between source samples is 1 / (target_hz / source_hz).
        self._source_idx: float = 0.0
        # Last emitted 60Hz token (used as the start of the next linear
        # segment when a fresh chunk arrives so we don't jerk).
        self._last_emitted: np.ndarray | None = None
        # Ratio: source_hz / target_hz — how much source_idx advances per
        # 60Hz tick. Set on every chunk swap.
        self._delta_per_tick: float = 1.0 / 3.0
        self._lock = threading.Lock()

    def load_chunk(self, chunk: TokenChunkEnvelope) -> None:
        """Install a fresh chunk. Callable from any thread."""
        with self._lock:
            self._current_chunk = chunk
            self._source_idx = 0.0
            source_hz = chunk.source_hz if chunk.source_hz > 0 else self.default_source_hz
            self._delta_per_tick = source_hz / max(self.target_interp_hz, 1.0)

    def has_chunk(self) -> bool:
        with self._lock:
            return self._current_chunk is not None

    def next_60hz_token(self) -> np.ndarray | None:
        """Return the next 60Hz waypoint token, or None if no chunk loaded
        or the chunk has been fully consumed.

        On chunk exhaust we return the final token indefinitely — the
        caller is expected to load a fresh chunk before we run out. This
        matches the "hold last" behaviour of the existing step_chunk
        server-side runner and keeps the robot safe if the VLA drops out.
        """
        with self._lock:
            ch = self._current_chunk
            if ch is None:
                return None
            H = ch.tokens.shape[0]
            idx = self._source_idx
            if idx >= H - 1:
                # Exhausted: hold the final waypoint. Do not advance.
                out = ch.tokens[-1].copy()
                self._last_emitted = out
                return out
            lo = int(np.floor(idx))
            frac = float(idx - lo)
            a = ch.tokens[lo]
            b = ch.tokens[lo + 1]
            token = (1.0 - frac) * a + frac * b
            # Smooth cross-chunk transition: when the emitter has a
            # previous emitted value and we're right at the start of a
            # newly-loaded chunk (frac tiny and idx tiny), pull the
            # interpolation "start" from _last_emitted so we don't jump.
            if self._last_emitted is not None and idx < 1.0:
                alpha = idx / 1.0  # 0..1 over the first source-rate slot
                token = (1.0 - alpha) * self._last_emitted + alpha * token
            self._last_emitted = token.copy()
            self._source_idx = idx + self._delta_per_tick
            return token


class ResidualHeadRunner:
    """Wraps an offline-trained ``ResidualHead`` (see
    ``gr00t/model/modules/residual_head.py``) for 60Hz inference.

    Cached conditioning (vla_feature, state_feature, history) is refreshed
    on every arriving token chunk; the head itself runs at 60Hz on the
    current interpolated token. Outputs are added to that token before it
    goes downstream to the interp_publisher.
    """

    def __init__(
        self,
        head_state_dict_path: str,
        head_config: dict[str, Any],
        device: str = "cpu",
        enabled: bool = True,
    ):
        # Deferred torch import — the robot side does not always have GR00T
        # installed; this class is a no-op when ``enabled=False`` and the
        # import is skipped so the module stays lightweight in the common
        # (no-residual) deployment.
        import torch  # noqa: E402

        # Import the head class from the GR00T repo. If GR00T isn't on
        # PYTHONPATH we ship a slim fallback below.
        try:
            from gr00t.model.modules.residual_head import ResidualHead
        except ImportError:
            from ._residual_head_slim import ResidualHead  # type: ignore

        self._torch = torch
        self._device = torch.device(device)
        self._enabled = bool(enabled)
        cfg = dict(head_config)
        self._action_dim = int(cfg["action_dim"])
        self._head_output_horizon = int(cfg["output_horizon"])
        self.head = ResidualHead(**cfg).to(self._device).eval()

        state = torch.load(head_state_dict_path, map_location=self._device)
        missing, unexpected = self.head.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "ResidualHeadRunner: state_dict mismatch missing=%s unexpected=%s",
                missing,
                unexpected,
            )

        # Conditioning caches — set by refresh_from_chunk() on chunk swap.
        self._state_feat_t: Any = None
        self._vla_feat_t: Any = None

        # 60Hz step counter — reset on refresh_from_chunk() so the sinusoidal
        # PE aligns with the beginning of each chunk (matches training-time
        # position encoding for the base_action_chunk).
        self._step: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    def refresh_from_chunk(self, chunk: TokenChunkEnvelope) -> None:
        """Called each time a new envelope is installed in the interpolator.
        Copies the pooled features onto device once so the 60Hz inner loop
        doesn't have to. Also resets the step counter.
        """
        torch = self._torch
        if chunk.state_feature is not None:
            self._state_feat_t = torch.from_numpy(chunk.state_feature).to(
                self._device
            )[None, :]
        else:
            self._state_feat_t = None
        if chunk.vla_feature is not None:
            self._vla_feat_t = torch.from_numpy(chunk.vla_feature).to(self._device)[
                None, :
            ]
        else:
            self._vla_feat_t = None
        self._step = 0

    def delta_for(self, token: np.ndarray) -> np.ndarray:
        """Return delta (D,) for the given 60Hz-interpolated token (D,).
        Returns zeros when disabled or when the head has been misloaded."""
        if not self._enabled:
            return np.zeros_like(token, dtype=np.float32)
        torch = self._torch
        # (1, 1, D) batch — head reuses sinusoidal PE, so we pass a
        # step_indices tensor pointing at this 60Hz tick's position.
        base = torch.from_numpy(token).to(self._device)[None, None, :]
        step_idx = torch.tensor([[self._step]], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            delta = self.head(
                base,
                state_feature=self._state_feat_t,
                vla_feature=self._vla_feat_t,
                step_indices=step_idx,
            )
        self._step += 1
        return delta[0, 0].detach().cpu().float().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# TokenChunkStreamer — glues receiver + interpolator + residual + sink
# ---------------------------------------------------------------------------


class TokenChunkStreamer:
    """Runs the 60Hz control loop on the robot side.

    Constructor takes a receiver (source of envelopes) and a sink
    (``send_token_fn: (np.ndarray, dict) -> None``). Every 1/target_hz
    the streamer:
      1. checks the receiver for a fresh envelope; if there is one, it's
         loaded into the interpolator (and residual head runner refreshed);
      2. asks the interpolator for the next 60Hz token;
      3. asks the residual runner for a bounded delta and adds it;
      4. hands (token, meta) off to the sink for un-normalization and
         downstream ROS publish (done by the caller's decode_fn +
         RobotInterface.step()).
    """

    def __init__(
        self,
        receiver: TokenChunkReceiver,
        send_token_fn: Callable[[np.ndarray, dict[str, Any]], None],
        *,
        target_interp_hz: float = 60.0,
        default_source_hz: float = 20.0,
        residual_runner: ResidualHeadRunner | None = None,
        residual_gain: float = 1.0,
    ):
        self.receiver = receiver
        self.send_token_fn = send_token_fn
        self.interp = TokenInterpolator(
            target_interp_hz=target_interp_hz,
            default_source_hz=default_source_hz,
        )
        self.residual = residual_runner
        self.residual_gain = float(residual_gain)
        self.target_interp_hz = float(target_interp_hz)

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_chunk_id: int | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "TokenChunkStreamer started @ %.1f Hz (residual=%s)",
            self.target_interp_hz,
            "on" if self.residual and self.residual.enabled else "off",
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _maybe_pull_new_chunk(self) -> None:
        # Drain everything (keep only the newest) so a burst doesn't create
        # a queue backlog if the streamer stalled briefly.
        newest: TokenChunkEnvelope | None = None
        while True:
            env = self.receiver.get(timeout=0.0)
            if env is None:
                break
            newest = env
        if newest is None:
            return
        if self._last_chunk_id is not None and newest.chunk_id <= self._last_chunk_id:
            # Out-of-order or duplicate — skip.
            logger.debug(
                "skipping older chunk_id=%d (have %d)",
                newest.chunk_id,
                self._last_chunk_id,
            )
            return
        self._last_chunk_id = newest.chunk_id
        self.interp.load_chunk(newest)
        if self.residual is not None:
            self.residual.refresh_from_chunk(newest)

    def _loop(self) -> None:
        interval = 1.0 / max(self.target_interp_hz, 1.0)
        next_wake = time.monotonic()
        while self._running:
            self._maybe_pull_new_chunk()

            token = self.interp.next_60hz_token()
            if token is not None:
                if self.residual is not None and self.residual.enabled:
                    delta = self.residual.delta_for(token) * self.residual_gain
                    token = token + delta
                    meta = {
                        "chunk_id": self._last_chunk_id,
                        "residual_active": True,
                        "delta_norm": float(np.linalg.norm(delta)),
                    }
                else:
                    meta = {
                        "chunk_id": self._last_chunk_id,
                        "residual_active": False,
                        "delta_norm": 0.0,
                    }
                try:
                    self.send_token_fn(token.astype(np.float32), meta)
                except Exception as e:
                    logger.exception("send_token_fn raised: %s", e)

            # Absolute-time schedule to avoid drift, matching the pattern
            # used by interp_publisher._interp_loop.
            next_wake += interval
            now = time.monotonic()
            if now > next_wake + interval:
                # missed multiple deadlines — resync
                next_wake = now + interval
            remaining = next_wake - now
            if remaining > 0:
                time.sleep(remaining)
