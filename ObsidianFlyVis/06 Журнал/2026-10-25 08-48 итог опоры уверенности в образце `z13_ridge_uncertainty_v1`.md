---
tags:
- журнал
дата: '2026-10-25'
время: 08:48
---

# 2026-10-25 08:48 — итог опоры уверенности в образце `z13_ridge_uncertainty_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_ridge_uncertainty_20261025/replay/report.json → paired: {'Dposterior-baseline': {'mean_per_video': -0.0010709796452467316, 'ci95': [-0.002795695431041091, 0.00047531374455853553], 'better': 4, 'n': 26}, 'Dposterior-D_old': {'mean_per_video': -5.029843632614319e-05, 'ci95': [-0.0011901671994917406, 0.0013391760341862442], 'better': 2, 'n': 26}, 'Dposterior-D_constant': {'mean_per_video': 0.0001801432866546955, 'ci95': [-0.0007707338254450718, 0.0014681495566807438], 'better': 1, 'n': 26}, 'Dposterior-C_posterior': {'mean_per_video': 4.592099992061737e-05, 'ci95': [-0.001096454942241036, 0.0014649036219078623], 'better': 3, 'n': 26}, 'Dposterior-R1_posterior': {'mean_per_video': 0.0016375777932264184, 'ci95': [-0.0007282445277154001, 0.0039783374512443615], 'better': 11, 'n': 26}, 'Dposterior-geometry': {'mean_per_video': -0.0008581008148241064, 'ci95': [-0.0026558249715814605, 0.000906103022577801], 'better': 3, 'n': 26}}

*Вывод.* Измерены первые назначения10парных условий при прежних средних откликах/весах/времени. Постоянный штраф отделяет общее ужесточение отказа от пространственной неопределённости. Это не калиброванная вероятность полезности и не HOTA.

*Решение.* Pass -> separate bounded runtime parity/price and frozen-policy matched HOTA. Fail -> preserve this conditional-covariance confidence implementation; do not retune1.96, lambda, alpha or thresholds on26. Select a genuinely different target/state mechanism with measured prerequisite, not another scalar penalty. Primary: False
