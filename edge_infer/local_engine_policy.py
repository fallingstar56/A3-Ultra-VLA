"""端侧本地 engine policy —— 替换 infer_a3_rtc_zmq.py 里的远程 ZMQ PolicyClient。

设计（基于对 trt_model_forward.py + infer_a3_rtc_zmq.py 的分析）：
  - 本地构建 Gr00tPolicy（gr00t 模型定义，提供 embed_tokens/rope/vlln/processor 等）
  - 用 setup_tensorrt_engines() 按 trt_mode 替换 backbone/action head（GPU 加速）。
    RTC 默认用 vit_llm_only，保留 gr00t/dev 的原生 PyTorch action head；
    normalize/前处理/decode 继续复用 Gr00tPolicy。
  - 暴露和 PolicyClient 一样的接口：ping() / get_rtc_metadata() / get_action(obs, options)
  - RTC 支持：get_action 的 options 里带 action_prefix + rtc_delay 时，把 prefix 注入
    到 action_head 的去噪循环（init_actions 挂点 + 逐步位置固定），并回填
    info["action_pred_normalized"]（下一轮当 prefix）。options=None 则纯采样（standard）。

依赖：Thor 上 GPU 可用的 torch（engine I/O 是 CUDA tensor）+ gr00t 模型定义 + tensorrt。

用法（在端侧启动脚本里）：
    policy = LocalEnginePolicy(
        model_path="/agibot/models/a3_60000/ckpt",
        engine_dir="/agibot/models/a3_60000/engines",
        embodiment_tag="NEW_EMBODIMENT",
    )
    metadata = policy.get_rtc_metadata()
    action_dict, info = policy.get_action(obs, options=...)
"""
from __future__ import annotations

from contextlib import nullcontext
import os
import sys
import inspect
from typing import Any, Optional

import numpy as np


def _make_cached_cuda_graph_module(torch, model, capture_warmup: int = 3):
    """Wrap a deterministic CUDA module with one graph per input signature.

    A3 cold-start DiT uses scalar timestep ``[B]`` while train-time RTC uses
    per-token timestep ``[B, 1 + horizon]``.  Cache both signatures rather
    than assuming one static graph can serve both paths.
    """

    class CachedCudaGraphModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.capture_warmup = capture_warmup
            self._graphs = {}

        @staticmethod
        def _value_signature(value):
            if torch.is_tensor(value):
                return (
                    "tensor",
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                )
            return ("value", type(value).__name__, repr(value))

        def _signature(self, args, kwargs):
            return (
                tuple(self._value_signature(value) for value in args),
                tuple(
                    (key, self._value_signature(value))
                    for key, value in sorted(kwargs.items())
                ),
            )

        @staticmethod
        def _static_copy(value):
            return value.clone() if torch.is_tensor(value) and value.is_cuda else value

        @staticmethod
        def _copy_into(static_value, value):
            if torch.is_tensor(static_value) and static_value.is_cuda:
                static_value.copy_(value)
            elif static_value != value:
                raise RuntimeError("DiT CUDA Graph non-tensor input changed")

        def _capture(self, signature, args, kwargs):
            static_args = tuple(self._static_copy(value) for value in args)
            static_kwargs = {
                key: self._static_copy(value) for key, value in kwargs.items()
            }

            def call_static():
                return self.model(*static_args, **static_kwargs)

            torch.cuda.synchronize()
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(self.capture_warmup):
                    call_static()
            warmup_stream.synchronize()
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = call_static()
            # Capture records the work but the captured output buffer is not
            # guaranteed to contain the current invocation's result.  Replay
            # once before returning the first call for this signature.
            graph.replay()
            entry = {
                "args": static_args,
                "kwargs": static_kwargs,
                "graph": graph,
                "output": static_output,
            }
            self._graphs[signature] = entry
            print(
                "[local-policy] captured DiT CUDA graph "
                f"signature={len(self._graphs)}",
                flush=True,
            )
            return entry

        def forward(self, *args, **kwargs):
            signature = self._signature(args, kwargs)
            entry = self._graphs.get(signature)
            if entry is None:
                entry = self._capture(signature, args, kwargs)
            else:
                for static_value, value in zip(entry["args"], args):
                    self._copy_into(static_value, value)
                for key, static_value in entry["kwargs"].items():
                    self._copy_into(static_value, kwargs[key])
                entry["graph"].replay()
            return entry["output"]

    return CachedCudaGraphModule()


class LocalEnginePolicy:
    """本地 TRT engine 推理，接口对齐 gr00t.policy.server_client.PolicyClient。"""

    def __init__(
        self,
        model_path: str,
        engine_dir: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        deployment_scripts_dir: Optional[str] = None,
        device: str = "cuda",
        trt_mode: str = "vit_llm_only",
    ):
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LocalEnginePolicy 需要 GPU 可用的 torch（engine I/O 是 CUDA tensor）。"
                "当前 torch.cuda.is_available()=False —— 检查 Thor 的 torch 是否与 CUDA 驱动匹配。"
            )

        # Keep torch/torchvision/transformers from the torch build venv.  The
        # deployment venv contributes TensorRT 10.13 only; putting it first
        # would shadow torchvision with its CUDA 13 build.
        trt_candidates = (
            os.environ.get("A3_TRT_SITE"),
            "/agibot/edge_deploy/trt_venv/lib/python3.12/site-packages",
            "/agibot/torch_build/venv/lib/python3.12/site-packages",
        )
        trt_site = next(
            (p for p in trt_candidates if p and os.path.isdir(p)), None
        )
        if trt_site and trt_site not in sys.path:
            sys.path.append(trt_site)

        # 让 trt_model_forward / trt_torch 可 import（它们不是包，靠 sys.path）
        if deployment_scripts_dir is None:
            # 默认在训练仓的 scripts/deployment/ 下
            gr00t_roots = [
                os.path.abspath(p)
                for p in sys.path
                if p and os.path.isfile(os.path.join(p, "gr00t", "__init__.py"))
            ]
            for cand in (
                os.environ.get("A3_DEPLOYMENT_SCRIPTS_DIR"),
                os.path.join(os.path.dirname(__file__), "scripts", "deployment"),
                *(os.path.join(root, "scripts", "deployment") for root in gr00t_roots),
                "/agibot/gr00t/scripts/deployment",
                "/agibot/edge_deploy/gr00t_code/scripts/deployment",
                "/agibot/gr00t_code/scripts/deployment",
                "/data/gr00t_17_deploy/scripts/deployment",
            ):
                if cand and os.path.isdir(cand):
                    deployment_scripts_dir = cand
                    break
        if not deployment_scripts_dir:
            raise FileNotFoundError(
                "cannot find scripts/deployment/trt_model_forward.py; set "
                "A3_DEPLOYMENT_SCRIPTS_DIR or A3_GR00T_DIR"
            )
        if deployment_scripts_dir and deployment_scripts_dir not in sys.path:
            sys.path.insert(0, deployment_scripts_dir)

        emb = EmbodimentTag.resolve(embodiment_tag) if hasattr(EmbodimentTag, "resolve") \
            else getattr(EmbodimentTag, embodiment_tag)

        print(f"[local-policy] 构建 Gr00tPolicy: {model_path} (device={device})")
        self.policy = Gr00tPolicy(model_path=model_path, embodiment_tag=emb, device=device)

        print(f"[local-policy] 加载 TRT engine + 打补丁: {engine_dir} (mode={trt_mode})")
        from trt_model_forward import setup_tensorrt_engines
        setup_tensorrt_engines(self.policy, trt_engine_path=engine_dir, mode=trt_mode)

        self._torch = torch
        self._action_head = self.policy.model.action_head
        self._trt_mode = trt_mode
        self._sdpa_backend = os.environ.get("A3_SDPA_BACKEND", "default").strip().lower()
        if self._sdpa_backend == "default":
            self._sdpa_context = nullcontext
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend_names = {
                "math": "MATH",
                "flash": "FLASH_ATTENTION",
                "efficient": "EFFICIENT_ATTENTION",
                "cudnn": "CUDNN_ATTENTION",
            }
            enum_name = backend_names.get(self._sdpa_backend)
            if enum_name is None or not hasattr(SDPBackend, enum_name):
                raise ValueError(
                    "A3_SDPA_BACKEND must be one of "
                    "default, math, flash, efficient, cudnn; got "
                    f"{self._sdpa_backend!r}"
                )
            backend = getattr(SDPBackend, enum_name)
            self._sdpa_context = lambda: sdpa_kernel([backend])
        print(f"[local-policy] SDPA backend: {self._sdpa_backend}")
        self._dit_cuda_graph_enabled = os.environ.get(
            "A3_DIT_CUDA_GRAPH", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._dit_cuda_graph_enabled:
            if trt_mode != "vit_llm_only":
                raise RuntimeError(
                    "A3_DIT_CUDA_GRAPH is only supported with "
                    "A3_TRT_MODE=vit_llm_only"
                )
            self._action_head.model = _make_cached_cuda_graph_module(
                torch, self._action_head.model
            )
            print(
                "[local-policy] DiT CUDA Graph enabled "
                "(cold/RTC signatures captured lazily)",
                flush=True,
            )
        if trt_mode == "vit_llm_only":
            native_sources = []
            try:
                # gr00t/dev keeps the model-level get_action wrapper separate
                # from the action-head flow-matching implementation.  Current
                # N1.5/N1D7 puts the train-time RTC branch in
                # get_action_with_features(), not get_action().
                for method_name in ("get_action", "get_action_with_features"):
                    method = getattr(self._action_head, method_name, None)
                    if method is not None:
                        native_sources.append(inspect.getsource(method))
                policy_source = inspect.getsource(self.policy._get_action)
            except (OSError, TypeError):
                policy_source = ""
            native_source = "\n".join(native_sources)
            has_native_rtc = (
                callable(getattr(self._action_head, "_sample_actions_with_prefix", None))
                and "_sample_actions_with_prefix" in native_source
                and "rtc_mode" in native_source
            )
            has_policy_prefix_injection = (
                "action_prefix" in policy_source
                and 'collated_inputs["inputs"]["action"]' in policy_source
            )
            if not has_native_rtc or not has_policy_prefix_injection:
                raise RuntimeError(
                    "the loaded gr00t checkout does not contain the complete "
                    "train-time RTC path (policy prefix injection + per-token action "
                    "head sampling). Deploy gr00t/dev before running A3 whole-body RTC."
                )
        self._rtc_guard_installed = self._install_trt_rtc_guard()
        print("[local-policy] 就绪")

    def _install_trt_rtc_guard(self) -> bool:
        """Make the TRT action-head path preserve the train-time RTC prefix.

        The PyTorch action head in current ``gr00t/dev`` natively implements
        per-token train-time RTC.  The exported full-pipeline TRT forward uses
        one scalar diffusion timestep, so it cannot reproduce that conditioning
        exactly, but it must still obey the two non-negotiable wire semantics:

        1. sample the postfix from noise rather than denoising the entire old
           action chunk; and
        2. return the first ``rtc_delay`` normalized actions byte-for-byte from
           the supplied prefix.

        Without this guard the old deployment code initializes *all* actions
        from the previous chunk and denoises every position.  In RTC mode this
        can make successive chunks barely update (the observed "whole body does
        not move" failure) and also breaks the prefix continuity contract.
        """
        if self._trt_mode not in ("n17_full_pipeline", "action_head"):
            # The action head remains PyTorch and already has the exact
            # _sample_actions_with_prefix implementation.
            return False

        torch = self._torch
        action_head = self._action_head
        base_get_action = action_head.get_action

        def _field(obj, name):
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        def rtc_guarded_get_action(backbone_output, action_input, options=None):
            opts = options or {}
            delay = int(opts.get("rtc_delay", 0) or 0)
            if opts.get("rtc_mode") != "train_time" or delay <= 0:
                return base_get_action(backbone_output, action_input, options=options)

            prefix = _field(action_input, "action")
            if prefix is None:
                prefix = getattr(action_head, "_local_rtc_prefix", None)
            if prefix is None:
                raise RuntimeError(
                    "train-time RTC requested but the normalized action prefix "
                    "was not injected into the TRT action head"
                )

            prefix = torch.as_tensor(prefix, device=_field(action_input, "state").device)
            horizon = int(action_head.config.action_horizon)
            action_dim = int(action_head.action_dim)
            delay = min(delay, horizon, int(prefix.shape[1]))
            width = min(action_dim, int(prefix.shape[-1]))
            batch_size = int(prefix.shape[0])
            engine_dtype = torch.bfloat16

            init_actions = torch.randn(
                (batch_size, horizon, action_dim),
                dtype=engine_dtype,
                device=prefix.device,
            )
            init_actions[:, :delay, :width] = prefix[:, :delay, :width].to(engine_dtype)

            sentinel = object()
            previous_init = getattr(action_head, "init_actions", sentinel)
            action_head.init_actions = init_actions
            try:
                result = base_get_action(
                    backbone_output, action_input, options=options
                )
            finally:
                if previous_init is sentinel:
                    try:
                        delattr(action_head, "init_actions")
                    except AttributeError:
                        pass
                else:
                    action_head.init_actions = previous_init

            pred = result["action_pred"]
            pred_pinned = pred.clone()
            pred_pinned[:, :delay, :width] = prefix[:, :delay, :width].to(pred.dtype)
            result["action_pred"] = pred_pinned
            return result

        action_head.get_action = rtc_guarded_get_action
        print(
            "[local-policy] installed TRT train-time RTC prefix guard "
            "(random postfix + exact normalized prefix pin)"
        )
        return True

    # ---- 接口对齐 PolicyClient ----
    def ping(self) -> bool:
        return True

    def get_rtc_metadata(self) -> dict[str, Any]:
        # Gr00tPolicy 自带 get_rtc_metadata（server 端也是调它）
        return self.policy.get_rtc_metadata()

    def get_action(self, observation: dict, options: Optional[dict] = None):
        """返回 (action_dict, info)。
        options 带 action_prefix + rtc_delay 时启用 RTC prefix 注入；None 则纯采样。
        info 必含 action_pred_normalized（下一轮 prefix）。
        """
        torch = self._torch
        ah = self._action_head

        # 1) Keep a fallback copy for deployment checkouts whose Gr00tPolicy
        # predates the action_input["action"] injection in current gr00t/dev.
        prefix = None
        rtc_delay = 0
        if options:
            prefix = options.get("action_prefix", None)
            rtc_delay = int(options.get("rtc_delay", 0) or 0)

        if prefix is not None and rtc_delay > 0:
            init = torch.as_tensor(np.asarray(prefix, dtype=np.float32), device="cuda")
            ah._local_rtc_prefix = init
            ah._local_rtc_delay = rtc_delay
        else:
            # standard / cold-start：清除注入，纯高斯噪声起点。
            for attr in ("_local_rtc_prefix", "_local_rtc_delay", "init_actions"):
                if hasattr(ah, attr):
                    try:
                        delattr(ah, attr)
                    except Exception:
                        setattr(ah, attr, None)

        # 2) 跑推理（policy.get_action 内部：前处理→backbone(engine)→action_head(engine)→反归一化）
        #    Gr00tPolicy.get_action 会把 options 传给 action_head.get_action。
        try:
            with self._sdpa_context():
                action_dict, info = self.policy.get_action(observation, options=options)
        finally:
            for attr in ("_local_rtc_prefix", "_local_rtc_delay"):
                if hasattr(ah, attr):
                    try:
                        delattr(ah, attr)
                    except Exception:
                        setattr(ah, attr, None)

        # 3) 确保 info 里有 action_pred_normalized（RTC 下一轮 prefix 需要）
        if "action_pred_normalized" not in info:
            # 兜底：若 policy 未回填，尝试从 action_head 最近一次归一化输出取
            npred = getattr(ah, "_last_action_pred_normalized", None)
            if npred is not None:
                info["action_pred_normalized"] = np.asarray(
                    npred.detach().float().cpu().numpy(), dtype=np.float32
                )
        if prefix is not None and rtc_delay > 0:
            normalized = info.get("action_pred_normalized")
            if normalized is None:
                raise RuntimeError(
                    "RTC inference returned no info['action_pred_normalized']; "
                    "the next chunk cannot be built safely"
                )
            normalized = np.asarray(normalized, dtype=np.float32)
            expected = np.asarray(prefix, dtype=np.float32)
            delay = min(rtc_delay, normalized.shape[1], expected.shape[1])
            width = min(normalized.shape[2], expected.shape[2])
            max_err = float(
                np.max(np.abs(normalized[:, :delay, :width] - expected[:, :delay, :width]))
            ) if delay > 0 and width > 0 else 0.0
            if max_err > 1e-2:
                raise RuntimeError(
                    f"TRT RTC prefix pin verification failed: max_abs_err={max_err:.6f}. "
                    "Use --trt-mode vit_llm_only for the exact PyTorch action-head path."
                )
            info["rtc_prefix_max_abs_err"] = max_err
        return action_dict, info
