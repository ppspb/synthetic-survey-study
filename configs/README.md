# Configs

These YAML files are organized by *use*, not by model vendor.

## Folders

- `quickstart/` — smoke tests that work without private benchmark data
- `reproduce/` — canonical benchmark and distribution runs used in the write-up
- `api/` — API-based large-model runs
- `robustness/` — perturbation / invariance runs
- `followup/` — targeted follow-up runs for selected hard questions

## Design rule

A YAML file should describe the *kind of run*. It should not force you to edit the file every time you switch endpoints or models.

Use env vars or CLI overrides instead:

```bash
python scripts/run_yaml.py   --config configs/reproduce/distribution_qwen35_direct_persona.yaml   --model qwen3.5-4b   --base-url http://127.0.0.1:1234/v1   --api-key lm-studio
```
