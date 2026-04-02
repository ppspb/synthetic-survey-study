# Scripts

Use `run_yaml.py` as the main entrypoint.

## Common commands

Smoke test without private data:

```bash
python scripts/run_yaml.py --config configs/quickstart/mock_smoke.yaml --test-run --mock
```

Run a local model benchmark:

```bash
python scripts/run_yaml.py --config configs/reproduce/single_direct_persona.yaml --model qwen3.5-4b
```

Run a distribution benchmark:

```bash
python scripts/run_yaml.py --config configs/reproduce/distribution_qwen35_direct_persona.yaml --model qwen3.5-4b
```

Run invariance:

```bash
python scripts/run_yaml.py --config configs/robustness/qwen_targeted_invariance.yaml --model qwen3.5-4b
```

Run an API-based benchmark:

```bash
python scripts/run_yaml.py --config configs/api/qwen235b_distribution_probe36_cerebras.yaml --model qwen-3-235b-a22b-instruct-2507
```

Note: rectification scripts live in `../rectification/scripts/`.
