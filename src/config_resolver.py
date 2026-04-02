from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def guess_project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for candidate in [path.parent, *path.parents]:
        if (candidate / "src").exists() and (candidate / "scripts").exists():
            return candidate
    # fallback for configs/<group>/<file>.yaml
    if len(path.parents) >= 3:
        return path.parents[2]
    return path.parent


def _expand_string(value: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        return env.get(key, default if default is not None else match.group(0))
    prev = value
    for _ in range(5):
        new = _ENV_RE.sub(repl, prev)
        if new == prev:
            return new
        prev = new
    return prev


def expand_env_tree(value: Any, env: dict[str, str] | None = None) -> Any:
    env = dict(os.environ) if env is None else dict(env)
    if isinstance(value, str):
        return _expand_string(value, env)
    if isinstance(value, list):
        return [expand_env_tree(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_env_tree(v, env) for k, v in value.items()}
    return value


def make_config_env(config_path: str | Path) -> dict[str, str]:
    config_path = Path(config_path).resolve()
    project_root = guess_project_root(config_path)
    return {
        **os.environ,
        'CONFIG_PATH': str(config_path),
        'CONFIG_DIR': str(config_path.parent),
        'PROJECT_ROOT': str(project_root),
        'OUTPUTS_DIR': os.environ.get('OUTPUTS_DIR', str(project_root / 'local_outputs')),
        'OPENAI_BASE_URL': os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'lm-studio'),
        'CEREBRAS_BASE_URL': os.environ.get('CEREBRAS_BASE_URL', 'https://api.cerebras.ai/v1'),
        'CEREBRAS_API_KEY_ENV': os.environ.get('CEREBRAS_API_KEY_ENV', 'CEREBRAS_API_KEY'),
        'DATA_DIR': os.environ.get('DATA_DIR', str(project_root / 'data' / 'private')),
        'SAMPLES_FILE': os.environ.get('SAMPLES_FILE', 'samples.tsv'),
        'QUESTION_METADATA_FILE': os.environ.get('QUESTION_METADATA_FILE', '../benchmark_spec/question_metadata.json'),
        'FULL_VALUE_QA_FILE': os.environ.get('FULL_VALUE_QA_FILE', str(project_root / 'data' / 'private' / 'wvs_wave7_source.zip')),
    }


def resolve_repo_path(project_root: Path, path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()
