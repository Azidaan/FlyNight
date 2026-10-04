---
aliases:
- CLR_FP
- FP
tags:
- тип/метрика
---

# FP

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*false positives*

Число ложных рамок: детекций трекера, не совпавших ни с одним эталонным объектом.

- **Лучше:** lower
- **Единица:** шт.
- **Чем считается:** TrackEval (CLR_FP)

## Где используется
- [Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1)
- [Цель 6: вето рождений по q_birth в неизменённом XYWHByteTrack (парная точка по точности рождений)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_qbirth_v1)
- [Второй проход чистым IoU: возвращает FN, добавляет FP↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_iou_second)
