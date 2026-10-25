---
tags:
- журнал
дата: '2026-10-25'
время: 05:36
---

# 2026-10-25 05:36 — итог обучения правильным невредным решениям `z13_region_semantic_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_region_semantic_20261025/training/report.json → paired: {'D-baseline': {'mean_per_video': -5.4094990803851555e-05, 'ci95': [-0.0001622849724115547, 0.0], 'better': 0, 'n': 26}, 'D-C': {'mean_per_video': -5.4094990803851555e-05, 'ci95': [-0.0001622849724115547, 0.0], 'better': 0, 'n': 26}, 'D-R1': {'mean_per_video': 0.001180213321976607, 'ci95': [-0.0008993772312237543, 0.0034153941851931082], 'better': 8, 'n': 26}, 'D-geometry': {'mean_per_video': 0.00027256176720871447, 'ci95': [-0.0008833744504075092, 0.0019962625791980727], 'better': 3, 'n': 26}, 'Dsemantic-Dutility': {'mean_per_video': -7.50130387537185e-05, 'ci95': [-0.00023317497196494508, 6.761351254156344e-05], 'better': 1, 'n': 26}};12моделей,3сида, признаки прежние, fit84/fit20/train26.

*Вывод.* Изменено множество допустимых ответов в той же функции обучения. Правильное бездействие остаётся допустимым. Результат не доказывает общую причину отказов; это первые назначения, не HOTA.

*Решение.* Pass -> separately budgeted frozen-policy closed-loop parity/price/HOTA with original/no-fly controls. Fail -> preserve this supervised-target implementation. No further scalar loss/threshold variants under this action; choose a different internal representation/state mechanism only with a measured cheap prerequisite. Измеренный основной критерий: False
