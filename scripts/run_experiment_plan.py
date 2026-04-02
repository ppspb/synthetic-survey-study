from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir, write_json
from src.lmstudio_rest import LMStudioRestClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequence of benchmark/robustness experiments from a YAML plan.")
    parser.add_argument("--base-config", default="config.yaml", help="Base config YAML to start from")
    parser.add_argument("--plan", required=True, help="Experiment plan YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of experiment names to run")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--plan-name", default=None, help="Optional label for temp config directory")
    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def estimate_requests(script_kind: str, cfg: dict[str, Any]) -> int | None:
    questions = len(cfg.get("benchmark", {}).get("questions", []))
    groups = len(cfg.get("benchmark", {}).get("groups", []))
    participants = int(cfg.get("benchmark", {}).get("participants_per_group_question", 0) or 0)
    modes = len(cfg.get("benchmark", {}).get("prompt_modes", []))
    if not all([questions, groups, participants, modes]):
        return None
    base = questions * groups * participants * modes
    if script_kind == "benchmark":
        return base
    perturbations = len(cfg.get("robustness", {}).get("perturbations", []))
    return base * perturbations if perturbations else None


def maybe_load_model(cfg: dict[str, Any]) -> tuple[LMStudioRestClient | None, str | None]:
    mm = cfg.get("model_management", {}) or {}
    if not (mm.get("enabled") and mm.get("use_lmstudio_rest") and mm.get("load_before_run")):
        return None, None
    base_url = cfg.get("api", {}).get("base_url", "")
    if str(base_url).startswith("mock://"):
        return None, None
    client = LMStudioRestClient(
        base_url_v1=base_url,
        api_key=cfg.get("api", {}).get("api_key", "lm-studio"),
        timeout_seconds=int(cfg.get("api", {}).get("timeout_seconds", 120) or 120),
    )
    if not client.ping():
        raise RuntimeError(f"LM Studio REST API is not reachable at {base_url}")
    instance_id = client.load_model(
        model=cfg.get("api", {}).get("model"),
        context_length=mm.get("context_length"),
        eval_batch_size=mm.get("eval_batch_size"),
        flash_attention=mm.get("flash_attention"),
        num_experts=mm.get("num_experts"),
        offload_kv_cache_to_gpu=mm.get("offload_kv_cache_to_gpu"),
    )
    return client, instance_id


def maybe_unload_model(cfg: dict[str, Any], client: LMStudioRestClient | None, instance_id: str | None) -> None:
    mm = cfg.get("model_management", {}) or {}
    if not (client and instance_id and mm.get("enabled") and mm.get("use_lmstudio_rest") and mm.get("unload_after_run")):
        if client:
            client.close()
        return
    try:
        client.unload_model(instance_id)
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    base_config_path = ROOT / args.base_config if not Path(args.base_config).is_absolute() else Path(args.base_config)
    plan_path = ROOT / args.plan if not Path(args.plan).is_absolute() else Path(args.plan)

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    defaults = expand_env(plan.get("defaults", {}))
    experiments = plan.get("experiments", [])
    if args.only:
        only = set(args.only)
        experiments = [e for e in experiments if e.get("name") in only]
    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]

    plan_label = args.plan_name or plan.get("name") or plan_path.stem
    temp_root = ROOT / "outputs" / "plan_configs" / plan_label
    ensure_dir(temp_root)

    results: list[dict[str, Any]] = []

    for idx, exp in enumerate(experiments, start=1):
        name = exp["name"]
        script_kind = exp.get("script", "benchmark")
        merged_cfg = deep_merge(base_cfg, defaults)
        merged_cfg = deep_merge(merged_cfg, expand_env(exp))

        script = "scripts/run_benchmark.py" if script_kind == "benchmark" else "scripts/run_robustness.py"
        tmp_cfg_path = temp_root / f"{idx:02d}_{name}.yaml"
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)

        cmd = [sys.executable, script, "--config", str(tmp_cfg_path), "--run-name", name]
        if merged_cfg.get("judge", {}).get("use_llm") is False:
            cmd.append("--disable-llm-judge")
        if exp.get("mock"):
            cmd.append("--mock")
        if exp.get("test_run"):
            cmd.append("--test-run")
        if exp.get("resume_run_dir"):
            cmd.extend(["--resume-run-dir", str(exp["resume_run_dir"])])

        estimated = estimate_requests(script_kind, merged_cfg)
        result_row = {
            "name": name,
            "script": script_kind,
            "model": merged_cfg.get("api", {}).get("model"),
            "api_mode": merged_cfg.get("api", {}).get("api_mode", "chat_completions"),
            "estimated_requests": estimated,
            "config_path": str(tmp_cfg_path),
            "command": cmd,
        }
        results.append(result_row)

        print(f"[{idx}/{len(experiments)}] {name} | script={script_kind} | model={result_row['model']} | api_mode={result_row['api_mode']} | estimated_requests={estimated}")
        print("    "+" ".join(cmd))

        if args.dry_run:
            continue

        rest_client = None
        instance_id = None
        try:
            rest_client, instance_id = maybe_load_model(merged_cfg)
            if instance_id:
                result_row["instance_id"] = instance_id
            completed = subprocess.run(cmd, cwd=str(ROOT))
            result_row["returncode"] = completed.returncode
        finally:
            try:
                maybe_unload_model(merged_cfg, rest_client, instance_id)
            except Exception as unload_err:
                result_row["unload_error"] = repr(unload_err)

        if result_row.get("returncode", 1) != 0 and args.stop_on_error:
            break

    write_json(temp_root / "plan_execution_summary.json", {"plan": str(plan_path), "results": results, "dry_run": bool(args.dry_run)})


if __name__ == "__main__":
    main()
