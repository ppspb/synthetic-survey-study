# synthetic-survey-study

`synthetic-survey-study` — публичный репозиторий с двумя задачами:

1. **Показать проект**: статическая страница в `docs/`, индекс запусков в `docs/runs.html`, индекс rectification в `docs/rectification.html`, inspectable артефакты в `artifacts/` и `rectification/`.
2. **Повторить проект**: основной код в `src/`, entry points в `scripts/`, параметризованные YAML-конфиги в `configs/`.

## Структура репозитория

```text
.
├── docs/                  # GitHub Pages: главная страница и каталоги
├── artifacts/             # Опубликованные run-артефакты и индекс по ним
│   ├── runs/
│   ├── RUN_ALIASES.json
│   ├── RUN_ALIASES.csv
│   └── RUN_ALIASES.md
├── rectification/         # Отдельный блок rectification: scripts, results, analysis, index
│   ├── scripts/
│   ├── results/
│   ├── analysis/
│   ├── regenerated_big_models/
│   └── RECTIFICATION_INDEX.*
├── src/                   # Основной код benchmark / prompting / evaluation
├── scripts/               # Запуск benchmark, invariance, robustness и подготовка данных
├── configs/               # Канонические YAML-конфиги
├── data/
│   ├── benchmark_spec/    # Публичные спецификации benchmark-а
│   ├── mock/              # Synthetic smoke dataset без лицензируемых данных
│   └── private/           # Плейсхолдер для локальных WVS/WVB-derived файлов
├── requirements.txt
└── README.md
```

## Что лежит в репозитории

В дереве репозитория лежит то, что удобно читать и проверять прямо на GitHub:

- `docs/` — сайт проекта и каталоги по артефактам
- `artifacts/runs/` — run-папки с summaries, predictions, logs и `raw.jsonl`
- `rectification/` — scripts, summaries, analysis и индексы по версиям v1–v5
- `configs/` — YAML-конфиги, которые можно запускать через `scripts/run_yaml.py`

## Что лучше публиковать через GitHub Releases

В Releases удобнее держать то, что тяжело читать как дерево файлов:

- большие zip-снапшоты результатов
- исторические архивы сайта
- крупные выгрузки rectification
- downloadable bundles, которые нужны как целый архив, а не как browseable папка

## GitHub Pages

Сайт публикуется из `docs/` как project site для репозитория `synthetic-survey-study`.

Главная страница:
- `docs/index.html`

Каталоги:
- `docs/runs.html`
- `docs/rectification.html`

Если у тебя есть новая версия главной страницы, переименуй её в `index.html` и положи в `docs/`.

## Ограничения по данным

Репозиторий **не должен** содержать:

- raw WVS zip
- derived benchmark tables, которые напрямую зависят от лицензируемых данных
- `subset.tsv` внутри run-папок

WorldValuesBench опирается на World Values Survey Wave 7. Из-за лицензирования исходные данные нужно скачать и подготовить локально.

См.:
- `data/private/README.md`
- `scripts/prepare_wvs_wave7.py`
- репозиторий WorldValuesBench

## Как повторить проект

### 1. Smoke test без лицензируемых данных

```bash
python scripts/run_yaml.py --config configs/quickstart/mock_smoke.yaml --test-run --mock
```

### 2. Dry run для локальной модели

```bash
python scripts/run_yaml.py   --config configs/reproduce/distribution_direct_persona.yaml   --model qwen3.5-4b   --base-url http://127.0.0.1:1234/v1   --api-key lm-studio   --dry-run
```

### 3. Реальный запуск после подготовки приватных данных

```bash
python scripts/run_yaml.py   --config configs/reproduce/distribution_direct_persona.yaml   --model qwen3.5-4b   --base-url http://127.0.0.1:1234/v1   --api-key lm-studio
```

## Как читать артефакты

### Запуски
- `artifacts/RUN_ALIASES.md`
- `artifacts/RUN_ALIASES.csv`
- `artifacts/RUN_ALIASES.json`
- `docs/runs.html`

### Rectification
- `rectification/RECTIFICATION_INDEX.md`
- `rectification/RECTIFICATION_INDEX.csv`
- `rectification/RECTIFICATION_INDEX.json`
- `docs/rectification.html`

## Что изменено в этой публичной версии

- сохранены `raw.jsonl`, summaries и predictions
- удалены лишние технические папки и дубли кода
- `rectification/` оставлен отдельным верхнеуровневым блоком
- локальные пути и локальные endpoint-адреса в логах зачистены
