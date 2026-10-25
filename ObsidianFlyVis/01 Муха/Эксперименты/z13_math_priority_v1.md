---
aliases:
- Предварительная математика резерва действий и цены ошибок
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P8
- муха
исход: положительный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Предварительная математика резерва действий и цены ошибок

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_math_priority_v1`

## Вопрос
Какой доступный выход имеет резерв и какой точности выбора потребует полезное вмешательство?

## Гипотеза
Полная рамка имеет больший резерв, но потребует подавления вреда; три старых пика могут не достигать прежнего порога даже при идеальном выборе.

## Ожидалось (до прогона)
For video reserve O, assumed recovery r and assumed harm frequency f among other known cases: G=r*O-f*(1-O). r/f are scenarios, not estimated future skills. Report means and paired-video95CI under assumptions.

## Что сделано
2026-10-25

## Получено
runs/z13_math_priority_20261025/report.json: action_bounds={'KeepTrack_three_peaks': {'source_key': 'per_video.Dpeak3', 'mean_per_video': 0.016333290216897698, 'ci95': [0.011763925813050998, 0.021382755368597604], 'better': 22, 'n': 26, 'required_recovery_for_2pp_at_1pct_false_harm': 1.8267395424691182, 'cannot_reach_2pp_even_with_perfect_selection': True}, 'region_center': {'source_key': 'per_video.centers_eye', 'mean_per_video': 0.026993698388551486, 'ci95': [0.019500768530289193, 0.03457753994915234], 'better': 21, 'n': 26, 'required_recovery_for_2pp_at_1pct_false_harm': 1.1013704972240315, 'cannot_reach_2pp_even_with_perfect_selection': False}, 'full_box_readout': {'source_key': 'per_video.boxes_eye', 'mean_per_video': 0.0524127276987803, 'ci95': [0.04232102625676238, 0.0625369349210125], 'better': 25, 'n': 26, 'required_recovery_for_2pp_at_1pct_false_harm': 0.5623800557073112, 'cannot_reach_2pp_even_with_perfect_selection': False}, 'wide_search_full_box': {'source_key': 'per_video.boxes_frame', 'mean_per_video': 0.07252156438672289, 'ci95': [0.06000494888042196, 0.08582558827957609], 'better': 25, 'n': 26, 'required_recovery_for_2pp_at_1pct_false_harm': 0.40367006150093954, 'cannot_reach_2pp_even_with_perfect_selection': False}};box_minus_center={'mean_per_video': 0.025419029310228818, 'ci95': [0.01849229629111801, 0.032841937480310404], 'better': 24, 'n': 26}

## Интерпретация
These are saved one-step action-space ceilings using truth, not implementations or predictions of named papers. Full learned association over all real regions shares the box ceiling, not the three-peak ceiling. Broad box support is optimistic relative to strict trainable masks.

## Решение
actual full-box supervision; перед обучением проверить строгий набор. Значения r/f сценарные, не прогноз качества.

## Нельзя заявлять
Repeated train26, one-step baseline-state outcomes, perfect choice can abstain, no closed-loop feedback. Conditional math has no measured learned r/f and is not a confidence interval for a future model.

## Критерии приёмки (записаны до прогона)
- ✅ Admit a non-temporal full-box label/model pilot only when saved eye-box ceiling mean>=.02 and lower95CI>0 and increment over eye-center lower>0; real strict-support gate before training must separately satisfy>=.02 and lower>0 excluding NULL-only reserve. No HOTA forecast. No acceptance threshold changes.  
  *runs/z13_math_priority_20261025/report.json*

## Доказательства
- Условия и пределы математической проверки до расчёта: `runs/z13_math_priority_20261025/protocol.json` · 
- Парные границы и условная чувствительность: `runs/z13_math_priority_20261025/report.json` · 

