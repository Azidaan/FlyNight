---
tags:
- журнал
дата: '2026-10-21'
время: 00:36
---

# 2026-10-21 00:36 — постановка парного обучения оценки `z13_pairwise_utility_v1`

↑ [[06 Журнал/Дни/2026-10-21|2026-10-21]]

*Постановка.* Add one fixed pairwise ranking term to a linear utility objective for unchanged D/C/R1 centers and saved14 features. Принцип парной логистической ошибки — RankNet, раздел3, https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf. Применимость к нашей задаче не установлена; смешанная цель — наша проверка. D_pair_map-D_point_map, D_pair_map-D_old_map, D_pair_map-D_pair_geometry, D_pair_map-R1_pair_map: all four lower95% paired video bootstrap intervals strictly above0 at fixed global1% coverage. Primary uses all26 videos.2%/5% secondary cannot rescue failure.

*Решение.* Один дешёвый полный train26 по сохранённым данным. Exploratory reused train26, one fixed seed; unknown outcomes excluded from fitting but retained in selection denominator. This is an application of modernized FlyVis, not new core modernization or evidence of HOTA. No dev, heldout75, GPU, core training, video decoding or downloads. Протокол: runs/z13_pairwise_utility_20261021/protocol.json. Бюджет600с +15с завершения; предел автора 2026-10-21T07:40:00+03:00.
