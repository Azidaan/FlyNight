---
tags:
- журнал
дата: '2026-10-25'
время: 07:10
---

# 2026-10-25 07:10 — итог доступности пространственных частей `z13_spatial_bins_gate_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_spatial_bins_gate_20261025/report.json → spatial_oracle_exposure: {'mean_per_video': 0.022802605365315858, 'ci95': [0.01636847685657445, 0.02932090951337184], 'better': 22, 'n': 26};37296запросов,26600известных, прежние реальные области совпадают.

*Вывод.* Это только доля случаев с разрешимыми четырьмя частями рамки и оракульно достижимым полезным действием среди неверных кандидатов. Не измерены различимость признаков, точность или HOTA.

*Решение.* Pass -> separately preregister and budget four-bin feature extraction, old-response parity, equal-capacity readout controls and assignment gate. No automatic HOTA or training admission from geometric coverage. Fail/inconclusive -> preserve this exact criterion; do not relax bin occupancy or choose a favorable subset. Gate: inconclusive
