# data/

This repository splits data into three layers.

## `benchmark_spec/`
Public metadata that defines the benchmark shape:
- question metadata
- question taxonomy
- codebook-like mappings

## `mock/`
A tiny synthetic dataset that ships with the repo.
Use it to smoke-test the pipeline without touching licensed survey data.

## `private/`
Local-only files required to reproduce the real benchmark:
- the downloaded WVS Wave 7 zip
- the prepared long-format `samples.tsv`

These files are intentionally excluded from GitHub.
