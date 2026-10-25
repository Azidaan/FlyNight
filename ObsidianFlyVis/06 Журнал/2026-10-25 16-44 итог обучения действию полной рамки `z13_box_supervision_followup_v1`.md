---
tags:
- журнал
дата: '2026-10-25'
время: '16:44'
---

# 2026-10-25 16:44 — итог обучения действию полной рамки `z13_box_supervision_followup_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_box_supervision_followup_20261025/training/report.json → paired: {'Dbox-baseline': {'mean_per_video': -0.0006249487512164997, 'ci95': [-0.0023736802413272997, 0.0003526666001311032], 'better': 3, 'n': 26}, 'Dbox-D_blank': {'mean_per_video': -1.3785918936725732e-05, 'ci95': [-0.0023735852493426838, 0.0024309203111224707], 'better': 11, 'n': 26}, 'Dbox-C_context': {'mean_per_video': 0.0010138165488213185, 'ci95': [-0.001390441358282744, 0.004846096344449262], 'better': 8, 'n': 26}, 'Dbox-R1_context': {'mean_per_video': -0.000562999085612896, 'ci95': [-0.0026022159308863716, 0.000919733743218069], 'better': 6, 'n': 26}, 'Dbox-geometry_context': {'mean_per_video': -0.0007133660810131399, 'ci95': [-0.002403846153846154, 0.00020341608576902696], 'better': 2, 'n': 26}, 'Dbox-Dcategorical': {'mean_per_video': 0.0013043221109191995, 'ci95': [-0.0003168299055562982, 0.0034219640718165753], 'better': 8, 'n': 26}}

*Вывод.* Одинаковое внутреннее считывание обучено фактическим действиям полной рамки с равными контролями. Изменены сопряжённые выход и метки, не доказана одна причина прежнего провала. Повторные train26, первые назначения, не HOTA.

*Решение.* Pass -> separately preregister matched runtime and closed-loop HOTA with native/C/R1/no-FlyVis controls. Fail -> preserve this exact action-supervision implementation and stop this categorical-to-box family; no new loss/guard/threshold sweep on26. A different internal mechanism needs its own measured rationale. Основной критерий: False

HOTA и относительный процент к стенду не измерены; доля потолка не установлена.
