---
tags:
- журнал
дата: '2026-10-25'
время: '12:58'
---

# 2026-10-25 12:58 — итог локальных пространственных признаков `z13_spatial_derivatives_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_spatial_derivatives_20261025/replay/report.json → paired: {'Dgradient-baseline': {'mean_per_video': -0.000894577804128659, 'ci95': [-0.0028583229024525354, 0.000872017656817422], 'better': 5, 'n': 26}, 'Dgradient-Dold': {'mean_per_video': 0.00012610340479192938, 'ci95': [0.0, 0.0003783102143757881], 'better': 1, 'n': 26}, 'Dgradient-C_gradient': {'mean_per_video': 0.0004760724392513178, 'ci95': [0.0, 0.0011457250026291497], 'better': 3, 'n': 26}, 'Dgradient-R1_gradient': {'mean_per_video': -0.0006994642740037581, 'ci95': [-0.002603229805675989, 0.0009190278331309512], 'better': 7, 'n': 26}, 'Dgradient-geometry': {'mean_per_video': -0.0006816989737060338, 'ci95': [-0.0018629765308342738, 0.00024058141451965443], 'better': 2, 'n': 26}, 'gradient_effect_D-R1': {'mean_per_video': -0.0005380174863547949, 'ci95': [-0.001994557302877814, 0.0006700098435470139], 'better': 2, 'n': 26}, 'gradient_effect_D-C': {'mean_per_video': 0.0005174974205386973, 'ci95': [0.0, 0.0012039876544427243], 'better': 3, 'n': 26}}

*Вывод.* Парное вмешательство меняет только локальные входы модели цели: прежние68каналов дополнены двумя пространственными производными. Состояние/образец/нормы/регуляризация/политика прежние. Это первые назначения на многократно использованном train26, не HOTA; структурная потеря направления не доказывает причину реальных ошибок.

*Решение.* Pass -> separate frozen-runtime parity/price and matched closed-loop gate before HOTA. Fail -> preserve this exact two-direction first-ring descriptor; no derivative scale/radius/lambda sweep on26 and no core training rescue. Distinguish representation change from proven natural error cause; choose another node only with measured prerequisite. Primary: False
