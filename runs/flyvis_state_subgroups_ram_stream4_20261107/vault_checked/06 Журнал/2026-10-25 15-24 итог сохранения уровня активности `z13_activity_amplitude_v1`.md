---
tags:
- журнал
дата: '2026-10-25'
время: '15:24'
---

# 2026-10-25 15:24 — итог сохранения уровня активности `z13_activity_amplitude_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_activity_amplitude_20261025/training/report.json → paired: {'Daffine-baseline': {'mean_per_video': -0.003491223018656633, 'ci95': [-0.012578521010528889, 0.001819947023518558], 'better': 13, 'n': 26}, 'Daffine-D_norm': {'mean_per_video': -0.0015619521565209335, 'ci95': [-0.007228524458251436, 0.001839425119657753], 'better': 13, 'n': 26}, 'Daffine-C_affine': {'mean_per_video': -0.003785495816930639, 'ci95': [-0.012787944038714437, 0.0014824769210418944], 'better': 11, 'n': 26}, 'Daffine-R1_affine': {'mean_per_video': -0.0035257817259811966, 'ci95': [-0.012488720828683379, 0.0018193839243145441], 'better': 12, 'n': 26}, 'Daffine-geometry_affine': {'mean_per_video': -0.003491223018656633, 'ci95': [-0.012578521010528889, 0.001819947023518558], 'better': 13, 'n': 26}}

*Вывод.* Проверено одно преобразование68входов при равной ёмкости, одинаковых начальных весах и точном повторе12старых моделей. Это действие регионального считывателя на повторных26видео, не HOTA и не полное объяснение отказа.

*Решение.* Pass -> separate frozen closed-loop gate, including native/C/R1/no-FlyVis. Fail -> preserve this exact normalization ablation and do not claim input amplitude is universally useless; no epsilon/gain/normalizer sweep on26. Основной критерий: False

HOTA и процент к стенду не измерены; доля потолка не установлена.
