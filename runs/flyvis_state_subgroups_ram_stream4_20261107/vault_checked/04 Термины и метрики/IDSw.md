---
aliases:
- ID switches
- IDSW
- IDSw
- idsw
tags:
- тип/метрика
---

# IDSw

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*ID switches*

Число переключений идентификатора: сколько раз одна эталонная траектория получила другой ID трекера.

- **Лучше:** lower
- **Единица:** шт.
- **Чем считается:** bdd100k / TrackEval

## Где используется
- [Отложенное рождение по q_birth на обучающих видео (130/35/35) и его совмещение с буфером IoU↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_deferred_train_v1)
- [Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1)
- [Цель 6: вето рождений по q_birth в неизменённом XYWHByteTrack (парная точка по точности рождений)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_qbirth_v1)
- [Каскад буферов IoU (C-BIoU) и буфер второй ассоциации на стенде XYWHByteTrack — HOTA на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fcbiou_cascade_v1)
- [Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_stress)
- [Итоговый трекер линии B: каскад буферов IoU + отложенное рождение по q_birth + сшивка обрывков по геометрии↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fline_b_combined_v1)
- [Офлайн-сшивка треков стенда по голове повторного опознания (геометрия + внешность + вход глаза) — HOTA/IDF1/IDSw на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_appearance_v1)
- [Вейвлетный и обучаемый предсказатель движения (рельс А)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fwavelet_motion)
- [XYWHByteTrack: модель рамки с независимыми шириной и высотой↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_bytetrack)
- [Второй проход чистым IoU: возвращает FN, добавляет FP↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_iou_second)
