"""A3 pnc_arm 控制模式切换 (replay 跑在 ADU 上, 直接调本地 S_*ControlMode.py)。

跟 ``RoboInterface/scripts/utils/change_pnc_arm_mode_a3.sh`` 等价, 但跳过 ssh ──
本模块只在 ADU 本机使用 (a3_server.replay 的 entry_point), 工具脚本就在
``/agibot/software/v0/config/pnc_arm/tools/``, 直接 subprocess 调即可。

如果将来 replay 也要在主机端跑 (现在没这需求), 用 sh 脚本走 ssh 就行。
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional


_TOOLS_DIR = Path("/agibot/software/v0/config/pnc_arm/tools")
_GET_SCRIPT = _TOOLS_DIR / "S_GetControlMode.py"
_SET_SCRIPT = _TOOLS_DIR / "S_SetControlMode.py"

_ALIAS_TO_FULL = {
    "passive": "PncArmControlMode_PASSIVE",
    "trajectory": "PncArmControlMode_ONLINE_TRAJECTORY",
    "traj": "PncArmControlMode_ONLINE_TRAJECTORY",
    "online_planning": "PncArmControlMode_ONLINE_PLANNING",
    "planning": "PncArmControlMode_ONLINE_PLANNING",
    "offline_planning": "PncArmControlMode_OFFLINE_PLANNING",
    "ik_servo": "PncArmControlMode_IK_SERVO",
    "collision_escape": "PncArmControlMode_COLLISION_ESCAPE",
}


def _resolve_target(target: str) -> str:
    if target.startswith("PncArmControlMode_"):
        return target
    key = target.lower()
    if key in _ALIAS_TO_FULL:
        return _ALIAS_TO_FULL[key]
    raise ValueError(
        f"未知 pnc_arm 模式 '{target}'; 支持: "
        f"{sorted(_ALIAS_TO_FULL)} 或 PncArmControlMode_<NAME>"
    )


def _run(cmd: str, timeout: float = 15.0, stdin: Optional[str] = None) -> str:
    """在 ADU 上 cd 到 tools 目录并执行 cmd, 返回 stdout (合并 stderr)."""
    proc = subprocess.run(
        ["bash", "-c", f"cd {_TOOLS_DIR} && {cmd}"],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def get_pnc_arm_mode(timeout: float = 5.0) -> Optional[str]:
    """读 S_GetControlMode.py 输出里的 .info.current_mode, 失败返回 None."""
    if not _GET_SCRIPT.is_file():
        print(f"[pnc_arm_mode] 找不到 {_GET_SCRIPT} (replay 必须跑在 ADU 上)")
        return None
    try:
        out = _run(f"timeout {int(timeout)} ./S_GetControlMode.py", timeout=timeout + 2)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r'"current_mode"\s*:\s*"([A-Za-z0-9_]+)"', out)
    return m.group(1) if m else None


def _resolve_menu_index(target_full: str, timeout: float = 8.0) -> Optional[int]:
    """跑 S_SetControlMode.py 一次拿菜单 (q 退出), 找 target_full 对应的编号."""
    try:
        out = _run(
            f"printf q | timeout {int(timeout)} ./S_SetControlMode.py",
            timeout=timeout + 2, stdin=None,
        )
    except subprocess.TimeoutExpired:
        return None
    pattern = re.compile(rf"^\s*(\d+):\s+{re.escape(target_full)}\s*$", re.MULTILINE)
    m = pattern.search(out)
    if not m:
        return None
    return int(m.group(1))


def set_pnc_arm_mode(target: str, timeout: float = 30.0,
                      verify: bool = True) -> bool:
    """切 pnc_arm 控制模式; 已是目标则幂等返回 True.

    Args:
        target: 'passive' / 'trajectory' / 'PncArmControlMode_*' (大小写不敏感)
        timeout: 整个切换流程的总秒数预算
        verify:  切换后回读 current_mode 验证 (默认 True)

    Returns: True = 成功或已是目标; False = 失败
    """
    try:
        target_full = _resolve_target(target)
    except ValueError as e:
        print(f"[set_pnc_arm_mode] {e}")
        return False

    if not _SET_SCRIPT.is_file():
        print(f"[set_pnc_arm_mode] 找不到 {_SET_SCRIPT} (replay 必须跑在 ADU 上)")
        return False

    cur = get_pnc_arm_mode()
    if cur == target_full:
        print(f"[set_pnc_arm_mode] 当前已是 {target_full}, 无需切换")
        return True
    print(f"[set_pnc_arm_mode] 当前: {cur}, 目标: {target_full}")

    idx = _resolve_menu_index(target_full)
    if idx is None:
        print(f"[set_pnc_arm_mode] S_SetControlMode 菜单中找不到 {target_full}")
        return False
    print(f"[set_pnc_arm_mode] 菜单编号 {idx}, 发送切换指令...")

    try:
        out = _run(
            f"echo {idx} | timeout 15 ./S_SetControlMode.py",
            timeout=timeout, stdin=None,
        )
    except subprocess.TimeoutExpired:
        print(f"[set_pnc_arm_mode] 切换超时 ({timeout}s)")
        return False

    if not re.search(r'"code"\s*:\s*"?0"?', out):
        print(f"[set_pnc_arm_mode] 切换响应可能失败 (没看到 code:0); 输出尾部:")
        print(out[-500:])

    if not verify:
        return True

    # 等生效再回读
    import time
    time.sleep(2.0)
    new_mode = get_pnc_arm_mode()
    if new_mode == target_full:
        print(f"[set_pnc_arm_mode] 已切换到 {new_mode}")
        return True
    print(f"[set_pnc_arm_mode] 验证失败: 期望 {target_full}, 实际 {new_mode}")
    return False
