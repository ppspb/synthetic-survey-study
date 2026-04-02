from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from .config_resolver import expand_env_tree, make_config_env, resolve_repo_path


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_retries: int
    timeout_seconds: int
    api_mode: str = "chat_completions"  # chat_completions | responses
    use_structured_output: bool = True
    structured_output_strict: bool = True
    max_completion_tokens: int | None = None
    request_delay_seconds: float = 0.0


@dataclass
class PathsConfig:
    data_dir: Path
    samples_file: str
    question_metadata_file: str
    outputs_dir: Path
    full_value_qa_file: str | None = None


@dataclass
class BenchmarkConfig:
    prompt_modes: list[str]
    questions: list[str]
    groups: list[dict[str, str]]
    participants_per_group_question: int
    explanation_max_words: int
    random_seed: int
    continue_on_error: bool = True


@dataclass
class JudgeConfig:
    enabled: bool = True
    use_llm: bool = False
    llm_model: str | None = None
    max_retries: int = 2
    fail_on_judge_error: bool = False


@dataclass
class RobustnessConfig:
    perturbations: list[str]
    compare_to: str = "original"
    continue_on_error: bool = True


@dataclass
class ModelManagementConfig:
    enabled: bool = False
    use_lmstudio_rest: bool = True
    load_before_run: bool = True
    unload_after_run: bool = True
    context_length: int | None = None
    eval_batch_size: int | None = None
    flash_attention: bool | None = None
    num_experts: int | None = None
    offload_kv_cache_to_gpu: bool | None = None


@dataclass
class AppConfig:
    api: APIConfig
    paths: PathsConfig
    benchmark: BenchmarkConfig
    judge: JudgeConfig
    robustness: RobustnessConfig
    model_management: ModelManagementConfig


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).resolve()
    env = make_config_env(path)
    project_root = Path(env["PROJECT_ROOT"])
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = expand_env_tree(yaml.safe_load(f), env)

    data_dir = resolve_repo_path(project_root, raw["paths"]["data_dir"])
    outputs_dir = resolve_repo_path(project_root, raw["paths"]["outputs_dir"])

    return AppConfig(
        api=APIConfig(**raw["api"]),
        paths=PathsConfig(
            data_dir=data_dir,
            samples_file=str(raw["paths"]["samples_file"]),
            question_metadata_file=str(raw["paths"]["question_metadata_file"]),
            outputs_dir=outputs_dir,
            full_value_qa_file=str(raw["paths"].get("full_value_qa_file")) if raw["paths"].get("full_value_qa_file") is not None else None,
        ),
        benchmark=BenchmarkConfig(**raw["benchmark"]),
        judge=JudgeConfig(**raw.get("judge", {})),
        robustness=RobustnessConfig(**raw.get("robustness", {"perturbations": ["original"]})),
        model_management=ModelManagementConfig(**raw.get("model_management", {})),
    )
