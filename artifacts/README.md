# Artifacts

`artifacts/` хранит inspectable run-артефакты проекта.

## Что внутри

- `runs/` — run-папки с логами, summaries, predictions и `raw.jsonl`
- `RUN_ALIASES.json` — машинно-читаемый индекс по run-папкам
- `RUN_ALIASES.csv` — табличная версия индекса
- `RUN_ALIASES.md` — человекочитаемый индекс

## Что не лежит здесь

- `subset.tsv` и другие файлы, которые не стоит публиковать из-за ограничений исходных данных
- тяжёлые zip-снапшоты, которые удобнее публиковать через GitHub Releases

## Как читать

- Для быстрого обзора открой `../docs/runs.html`
- Для прямого чтения на GitHub начни с `RUN_ALIASES.md`
