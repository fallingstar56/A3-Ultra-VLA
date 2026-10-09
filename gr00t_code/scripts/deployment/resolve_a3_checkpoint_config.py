#!/usr/bin/env python3
"""Resolve A3 deployment settings from checkpoint metadata.

The normal path is intentionally zero-input beyond ``--checkpoint``:

* embodiment comes from the saved training config;
* hand kind comes from the selected embodiment's state/action schema;
* task comes from the saved training prompt catalog.

Explicit overrides exist only for older or ambiguous checkpoints.  Shell
output is safely quoted and can be saved as ``deploy.env`` and sourced by the
ADU launch terminals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def normalize_tag(value: str) -> str:
    return value.strip().strip("'\"").upper()


def saved_embodiment_candidates(root: Path) -> list[str]:
    candidates: list[str] = []

    # Saved Hydra/YAML files contain Python object tags, so parsing the single
    # scalar field as text is safer than requiring PyYAML unsafe loaders.
    pattern = re.compile(r"^\s*embodiment_tag:\s*([^#\s]+)", re.MULTILINE)
    for relative in ("experiment_cfg/config.yaml", "experiment_cfg/conf.yaml"):
        path = root / relative
        if path.is_file():
            candidates.extend(match.group(1) for match in pattern.finditer(path.read_text()))

    if candidates:
        return candidates

    # Older checkpoints may omit the resolved YAML but keep the per-training
    # dataset statistics.  A single key is an unambiguous embodiment.
    stats = load_json(root / "experiment_cfg/dataset_statistics.json")
    if len(stats) == 1:
        return list(stats)

    processor = load_json(root / "processor_config.json")
    configs = processor.get("processor_kwargs", {}).get("modality_configs", {})
    if isinstance(configs, dict) and len(configs) == 1:
        return list(configs)
    return []


def resolve_embodiment(root: Path, override: str | None) -> tuple[str, str]:
    if override:
        return normalize_tag(override), "CLI override"

    raw = saved_embodiment_candidates(root)
    normalized = list(dict.fromkeys(normalize_tag(value) for value in raw))
    if len(normalized) == 1:
        return normalized[0], "checkpoint training config"
    if not normalized:
        raise SystemExit(
            "cannot resolve embodiment from checkpoint; provide "
            "--embodiment-tag after checking the training config"
        )
    raise SystemExit(
        "checkpoint contains conflicting embodiment tags: "
        f"{normalized}; provide --embodiment-tag"
    )


def selected_modality_config(root: Path, embodiment: str) -> dict[str, Any]:
    processor = load_json(root / "processor_config.json")
    configs = processor.get("processor_kwargs", {}).get("modality_configs", {})
    if not isinstance(configs, dict):
        return {}
    wanted = embodiment.lower()
    for key, value in configs.items():
        if str(key).lower() == wanted and isinstance(value, dict):
            return value
    return {}


def modality_keys(config: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for section_name in ("state", "action"):
        section = config.get(section_name, {})
        if isinstance(section, dict):
            values = section.get("modality_keys", []) or []
            keys.extend(str(value) for value in values)
            refs = section.get("reference_only_keys", []) or []
            keys.extend(str(value) for value in refs)
    return list(dict.fromkeys(keys))


def resolve_hand_kind(
    root: Path, embodiment: str, override: str | None
) -> tuple[str, str, list[str]]:
    if override:
        return override, "CLI override", []

    keys = modality_keys(selected_modality_config(root, embodiment))
    lower = [key.lower() for key in keys]
    has_gripper = any("gripper" in key for key in lower)
    has_hand = any("hand" in key for key in lower)
    if has_gripper and not has_hand:
        return "gripper", "checkpoint action/state schema", keys
    if has_hand and not has_gripper:
        return "hand", "checkpoint action/state schema", keys
    if has_gripper and has_hand:
        raise SystemExit(
            "checkpoint schema contains both hand and gripper keys; verify the "
            "physical end effector and pass --hand-kind"
        )
    raise SystemExit(
        "cannot resolve hand kind from checkpoint schema; verify the physical "
        "end effector and pass --hand-kind hand|gripper"
    )


def saved_prompts(root: Path) -> list[str]:
    data = load_json(root / "experiment_cfg/launch/prompts.json")
    prompts: list[str] = []
    for value in data.get("prompts", []) or []:
        if isinstance(value, str):
            prompts.append(value)
        elif isinstance(value, dict) and value.get("prompt"):
            prompts.append(str(value["prompt"]))
    for value in data.get("entries", []) or []:
        if isinstance(value, dict) and value.get("prompt"):
            prompts.append(str(value["prompt"]))
    return list(dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip()))


def resolve_task(root: Path, override: str | None) -> tuple[str, str, list[str]]:
    prompts = saved_prompts(root)
    if override:
        if prompts and override not in prompts:
            raise SystemExit(
                "--task is not present in the checkpoint prompt catalog; "
                f"available prompts: {prompts}"
            )
        return override, "CLI override", prompts
    if len(prompts) == 1:
        return prompts[0], "checkpoint prompt catalog", prompts
    if not prompts:
        raise SystemExit(
            "checkpoint has no experiment_cfg/launch/prompts.json; recover the "
            "exact training prompt and pass --task"
        )
    raise SystemExit(
        "checkpoint contains multiple training prompts; select one with --task. "
        f"available prompts: {prompts}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment-tag", default=None)
    parser.add_argument("--hand-kind", choices=("hand", "gripper"), default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print source-compatible export statements instead of JSON",
    )
    args = parser.parse_args()

    root = Path(args.checkpoint).expanduser().resolve()
    if not (root / "processor_config.json").is_file():
        raise SystemExit(f"not a checkpoint directory: {root}")

    embodiment, embodiment_source = resolve_embodiment(root, args.embodiment_tag)
    hand_kind, hand_source, keys = resolve_hand_kind(root, embodiment, args.hand_kind)
    task, task_source, prompts = resolve_task(root, args.task)

    result = {
        "checkpoint": str(root),
        "embodiment": embodiment,
        "hand_kind": hand_kind,
        "task": task,
        "sources": {
            "embodiment": embodiment_source,
            "hand_kind": hand_source,
            "task": task_source,
        },
        "schema_keys": keys,
        "available_prompts": prompts,
    }

    if args.shell:
        values = {
            "EMBODIMENT": embodiment,
            "HAND_KIND": hand_kind,
            "TASK": task,
            "A3_HAND_KIND": hand_kind,
            "A3_TASK": task,
        }
        for name, value in values.items():
            print(f"export {name}={shlex.quote(value)}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
