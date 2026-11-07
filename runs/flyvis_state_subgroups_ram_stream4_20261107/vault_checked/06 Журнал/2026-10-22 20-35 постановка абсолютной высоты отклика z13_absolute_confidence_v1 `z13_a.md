---
tags:
- журнал
дата: '2026-10-22'
время: '20:35'
---

# 2026-10-22 20:35 — постановка абсолютной высоты отклика z13_absolute_confidence_v1 `z13_absolute_confidence_v1`

↑ [[06 Журнал/Дни/2026-10-22|2026-10-22]]

*Постановка.* Replace query-normalized peak z=(max-mean)/std by raw max of the same observed-center ridge response. Center/model/lambda/history unchanged; no boundary veto.

One scalar confidence replacement only, no fitted coefficients or grids. The same84responses per arm are ranked by max(raw response). Reference max-z is reconstructed unchanged, then compare original,geometry6,map14 and R1 under identical1%quota.

Критерии до исходов: ['All C/D/R1 raw centers<1e-6px and original z-scores<1e-9 to saved last rows; all maps/hash unchanged.', 'D absolute-D raw, D absolute-D geometry6 and D absolute-D map14: all paired-video95%lower>0 for selected_success_fraction_positive at1%equal quota.', 'D absolute-R1 absolute paired-video95%lower>0. Controls keep the same history and maps; no anatomy claim.']

Контраргумент: Raw ridge amplitudes can vary across target models and scene contrast; removing normalization may worsen ranking. Score is not a calibrated probability. Previous geometry6/map14 controls must be beaten, not only the weak max-z baseline.

По первичному итогу предыдущего опыта D max-z и отказ от края выбрали2полезных/84;62известных отрицательных случая связаны с отсутствием цели,20неизвестны. Источник: runs/z13_boundary_readout_20261022/report.json → summary.D_raw/D_boundary. Ухудшение координат при присутствующей цели в этих выбранных строках не наблюдалось. Это ограничение отбора, не доказательство причины всей HOTA. Опора — официальный DiMP localize_advanced использует абсолютный max_score1; его модель и пороги не переносятся. CPUдо600с, без новых извлечений или обучения. Каталог новой карточки [z13_absolute_confidence_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_absolute_confidence_v1) содержит самостоятельные критерии.
