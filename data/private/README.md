# Private benchmark inputs

Place files derived from World Values Survey / WorldValuesBench here when reproducing the benchmark locally.

This directory is intentionally empty in the public repository.

Expected local layout:

```text
data/private/
├── raw/
│   └── WVS_Cross-National_Wave_7_csv_v6_0.zip
├── samples.tsv
└── question_metadata.json
```

Use `scripts/prepare_wvs_wave7.py` to create the local derived files after downloading the official WVS Wave 7 CSV zip yourself.
