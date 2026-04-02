from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def infer_kind(config_path: Path, raw: dict) -> str:
    parts = {p.lower() for p in config_path.parts}
    name = config_path.name.lower()
    benchmark = raw.get('benchmark', {})
    has_perturb = bool(benchmark.get('perturbations')) or 'invariance' in name or 'robustness' in parts
    mode = benchmark.get('prompt_mode') or (benchmark.get('prompt_modes') or [None])[0]
    is_distribution = isinstance(mode, str) and 'distribution' in mode
    if 'api' in parts or 'cerebras' in name:
        return 'cerebras'
    if has_perturb and is_distribution:
        return 'invariance'
    if is_distribution:
        return 'distribution'
    return 'benchmark'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run any canonical YAML config with a single command.')
    p.add_argument('--config', required=True, help='Path to YAML config')
    p.add_argument('--kind', choices=['auto','benchmark','distribution','invariance','cerebras'], default='auto')
    p.add_argument('--run-name', default=None)
    p.add_argument('--test-run', action='store_true')
    p.add_argument('--mock', action='store_true')
    p.add_argument('--fail-on-error', action='store_true')
    p.add_argument('--model', default=None)
    p.add_argument('--base-url', default=None)
    p.add_argument('--api-key', default=None)
    p.add_argument('--api-key-env', default=None)
    p.add_argument('--outputs-dir', default=None)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    kind = infer_kind(cfg_path, raw) if args.kind == 'auto' else args.kind
    script_map = {
        'benchmark': ROOT / 'scripts' / 'run_benchmark.py',
        'distribution': ROOT / 'scripts' / 'run_distribution_benchmark.py',
        'invariance': ROOT / 'scripts' / 'run_distribution_invariance.py',
        'cerebras': ROOT / 'scripts' / 'cerebras_distribution_runner.py',
    }
    script = script_map[kind]
    cmd = [sys.executable, str(script), '--config', str(cfg_path)]
    if args.run_name:
        cmd += ['--run-name', args.run_name]
    if args.test_run:
        cmd.append('--test-run')
    if args.mock and kind in {'benchmark','distribution','invariance'}:
        cmd.append('--mock')
    if args.fail_on_error and kind in {'distribution','invariance','cerebras'}:
        cmd.append('--fail-on-error')
    if args.model:
        cmd += ['--model', args.model]
    if args.base_url:
        cmd += ['--base-url', args.base_url]
    if args.api_key and kind in {'benchmark','distribution','invariance'}:
        cmd += ['--api-key', args.api_key]
    if args.api_key_env and kind == 'cerebras':
        cmd += ['--api-key-env', args.api_key_env]
    if args.outputs_dir:
        cmd += ['--outputs-dir', args.outputs_dir]
    print('Resolved kind:', kind)
    print('Command:', ' '.join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == '__main__':
    main()
