from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import AppConfig
from .io_utils import ensure_dir, write_json


def slugify(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    return text.strip("-").lower()


def create_run_dir(base_outputs_dir: Path, script_name: str, run_name: str | None = None) -> tuple[str, Path]:
    ensure_dir(base_outputs_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = slugify(run_name)
    base_run_id = f"{script_name}_{ts}" + (f"_{suffix}" if suffix else "")
    run_id = base_run_id
    run_dir = base_outputs_dir / "runs" / run_id

    counter = 1
    while run_dir.exists():
        run_id = f"{base_run_id}_{counter:02d}"
        run_dir = base_outputs_dir / "runs" / run_id
        counter += 1

    ensure_dir(run_dir)
    return run_id, run_dir


def write_latest_pointer(base_outputs_dir: Path, script_name: str, run_dir: Path) -> None:
    ensure_dir(base_outputs_dir)
    latest_path = base_outputs_dir / f"latest_{script_name}.txt"
    latest_path.write_text(str(run_dir), encoding="utf-8")


def apply_test_overrides(config: AppConfig, *, max_questions: int | None = None, max_groups: int | None = None, participants_per_group_question: int | None = None, modes: Iterable[str] | None = None, perturbations: Iterable[str] | None = None, disable_llm_judge: bool = False, mock: bool = False) -> AppConfig:
    cfg = deepcopy(config)
    if max_questions is not None:
        cfg.benchmark.questions = cfg.benchmark.questions[:max_questions]
    if max_groups is not None:
        cfg.benchmark.groups = cfg.benchmark.groups[:max_groups]
    if participants_per_group_question is not None:
        cfg.benchmark.participants_per_group_question = participants_per_group_question
    if modes is not None:
        cfg.benchmark.prompt_modes = list(modes)
    if perturbations is not None:
        cfg.robustness.perturbations = list(perturbations)
    if disable_llm_judge:
        cfg.judge.use_llm = False
    if mock:
        cfg.api.base_url = "mock://local"
        cfg.api.model = "mock-model"
    return cfg


def write_run_manifest(run_dir: Path, payload: dict) -> None:
    write_json(run_dir / "run_manifest.json", payload)


def apply_runtime_overrides(config: AppConfig, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None, outputs_dir: str | None = None) -> AppConfig:
    cfg = deepcopy(config)
    if model:
        cfg.api.model = model
    if base_url:
        cfg.api.base_url = base_url
    if api_key:
        cfg.api.api_key = api_key
    if outputs_dir:
        cfg.paths.outputs_dir = Path(outputs_dir)
    return cfg
