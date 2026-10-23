---
tags:
- журнал
дата: '2026-10-24'
время: 00:18
---

# 2026-10-24 00:18 — постановка описания цели коэффициентами фильтра `z13_filter_descriptor_v1`

↑ [[06 Журнал/Дни/2026-10-24|2026-10-24]]

*Постановка.* Replace mean foreground descriptor supplied to candidate ranker with68coefficients of the existing observed-center ridge target filter. Same3peaks/queryfeatures/geometry. Foreground+background affect filter fitting; no query labels enter descriptor.

Current response map already uses this filter; coefficients may add no information. Coefficients are not feature means, hence separate descriptor embedding in both matched arms. This test does not diagnose intrinsic representational limits.

Протокол `runs/z13_filter_descriptor_cache_20261023/protocol.json`. Основание DiMP [§3↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F3.%20%D0%92%D1%81%D0%B5%20%D0%B8%D0%B7%D0%BC%D0%B5%D1%80%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%87%D0%B8%D1%81%D0%BB%D0%B0)/3.1: модель цели как веса различающего фильтра. Новое здесь — подать уже существующие веса выборщику вместо среднего; сам фильтр, карта и ядро прежние. Равные отдельные кодировщики в обеих версиях,36моделей. До исходов: прирост D>=2п.п. и положительный интервал, также сильные контроли. Сначала8эпизодов только для стоимости/воспроизведения; CPU5400с+20, затем отдельное обучение1800с, до07:40. Текущую D на стенде не заменять.
