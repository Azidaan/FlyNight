---
tags:
- журнал
дата: '2026-10-18'
время: '22:20'
---

# 2026-10-18 22:20 — итог z13_directional_context_v1

↑ [[06 Журнал/Дни/2026-10-18|2026-10-18]]

*Получено.* runs/z13_directional_context_20261018/report.json: {"localization": {"D_directional-D_mean": {"mean_per_video": 0.01691126847019758, "ci95": [0.0005876933538153486, 0.03252932656492639], "better": 16, "n": 26}, "D_directional-D_shuffled": {"mean_per_video": 0.021560697896978207, "ci95": [-0.008598058227838276, 0.0564354210439271], "better": 14, "n": 26}, "D_directional-C_mean": {"mean_per_video": 0.013939190464576564, "ci95": [-0.036779777871385956, 0.05254838597433548], "better": 15, "n": 26}, "D_directional-R1_directional": {"mean_per_video": 0.12375269702966868, "ci95": [0.08522330612354408, 0.16515175999309506], "better": 22, "n": 26}}, "selection_at1percent": {"D_context_map-D_mean_map": {"mean_per_video": 0.0025102266409775665, "ci95": [-0.0006305170239596469, 0.006100552673133318], "better": 4, "n": 26}, "D_context_map-D_context_geometry": {"mean_per_video": -0.001914142675703298, "ci95": [-0.007168714409989515, 0.0032704828150894208], "better": 3, "n": 26}, "D_context_map-D_shuffled_map": {"mean_per_video": 0.0011685895874661684, "ci95": [-0.006735821352000274, 0.008218240442222576], "better": 6, "n": 26}, "D_context_map-R1_context_map": {"mean_per_video": 0.0028944956825024226, "ci95": [-0.00189909135294015, 0.008014374685449879], "better": 6, "n": 26}}, "localization_pass": false, "selection_pass": false, "proceed": false, "decision": "directional_readout_not_confirmed; do_not_tune_dev_or_apply_tracker", "dev_evaluated": false, "hota_measured": false, "limitations": "Exploratory reused train26, one fixed contrast scale and original lambda; no proof of unique failure cause, no blind evaluation or safety guarantee on unknowns."}

*Вывод.* Оба критерия конкретной пространственной замены не выполнены. Неуспех не определяет единственную причину и не закрывает FlyVis.

*Решение.* directional_readout_not_confirmed; do_not_tune_dev_or_apply_tracker. HOTA не измерялась.
