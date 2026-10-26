---
tags:
- журнал
дата: '2026-10-21'
время: 00:40
---

# 2026-10-21 00:40 — итог парной цели оценки `z13_pairwise_utility_v1`

↑ [[06 Журнал/Дни/2026-10-21|2026-10-21]]

*Получено.* runs/z13_pairwise_utility_20261021/report.json (primary): pair_map-point_map: +0,2175 п.п. [-0,2298; +0,8022]; pair_map-old_map: +0,2175 п.п. [-0,2298; +0,8022]; pair_map-pair_geometry: +0,4712 п.п. [-0,0473; +1,1387]; D_pair_map-R1_pair_map: +0,4464 п.п. [-0,2542; +1,2050]. runs/z13_pairwise_utility_20261021/report.json (arms.D.coverage.pair_map.0.01): {"selected": 84, "known": 48, "unknown": 36, "useful": 12, "harmful": 36, "useful_share_known": 0.25}. Все26видео train; воспроизведение прежних оценок и выбранных ключей подтверждено в runs/z13_pairwise_utility_20261021/reproduction.json; source_hashes_unchanged=true.

*Вывод.* Заранее заданный критерий парной цели не выполнен. Это итог одной фиксированной реализации, не диагноз представления или отказ от FlyVis.

*Решение.* fixed_pairwise_loss_gate_not_passed. Exploratory reused train26, one fixed seed; unknown outcomes excluded from fitting but retained in selection denominator. This is an application of modernized FlyVis, not new core modernization or evidence of HOTA. No dev, heldout75, GPU, core training, video decoding or downloads. Дополнительные доли2%/5% и отдельный положительный контраст не заменяют общий критерий. Известный неполезный ответ не тождествен измеренному вреду HOTA.
