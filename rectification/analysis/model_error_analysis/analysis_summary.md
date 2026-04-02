
# Анализ ошибок: Qwen3.5-4b `direct_persona_distribution` vs Llama `interview_direct_distribution`

## Главные метрики
- Qwen: mean_w1 ~ 0.183, top_option_hit ~ 0.269, zero_variance_cell_rate ~ 0.019, mean_unique_scores_per_cell ~ 3.03
- Llama: mean_w1 ~ 0.172, top_option_hit ~ 0.282, zero_variance_cell_rate ~ 0.028, mean_unique_scores_per_cell ~ 3.05

## Предлагаемый calibration split
Калибровочные вопросы (по одному самому трудному на категорию):
Q107, Q178, Q46, Q160, Q113, Q123, Q132, Q237, Q201, Q165, Q57, Q1

Оценочные вопросы (оставшиеся 24):
Q58, Q108, Q166, Q159, Q131, Q59, Q2, Q177, Q158, Q106, Q112, Q47, Q121, Q199, Q122, Q235, Q3, Q236, Q48, Q200, Q133, Q114, Q164, Q176
