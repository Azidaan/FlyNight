---
aliases:
- AssA
tags:
- тип/метрика
---

# AssA

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*Association Accuracy*

Качество ассоциации: насколько правильно детекции связаны в траектории одного объекта во времени. Не зависит от числа выданных рамок.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** TrackEval 1.3.0

## Где используется
- [Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1)
- [Цель 6: вето рождений по q_birth в неизменённом XYWHByteTrack (парная точка по точности рождений)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_qbirth_v1)
- [Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_stress)
- [ByteTrack против BoT-SORT и зоопарк из одиннадцати трекеров↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ftracker_zoo)
- [XYWHByteTrack: модель рамки с независимыми шириной и высотой↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_bytetrack)
- [Второй проход чистым IoU: возвращает FN, добавляет FP↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_iou_second)
