---
tags:
- журнал
дата: '2026-10-25'
время: 06:40
---

# 2026-10-25 06:40 — итог частоты искусственного времени образца `z13_history_clock_gate_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_history_clock_gate_20261025/report.json → affected_query_fraction: {'mean_per_video': 0.039541515812622004, 'ci95': [0.02845169836048589, 0.052124133460154085], 'better': 22, 'n': 26};37296запросов,26600известных; все нулевые интервалы — точные ведущие копии.

*Вывод.* Измерена только частота триггера искусственного времени. Не измерены ошибка активности, улучшение представления или HOTA. Положительный статус относится только к наличию мишени.

*Решение.* If enough exposure and exact leading-copy semantics are verified, separately register a bounded frozen-core skip-padding prototype with same-history C/D/R1 controls and saved old responses; no new core training. Otherwise preserve the specific insufficient trigger and choose another state/input node. No HOTA from counts. Admission: True
