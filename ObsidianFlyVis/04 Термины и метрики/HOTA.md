---
aliases:
- HOTA
- HOTA_cls_av
- mHOTA
tags:
- тип/метрика
---

# HOTA

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*Higher Order Tracking Accuracy*

Совместная оценка качества обнаружения и ассоциации: среднее геометрическое DetA и AssA, усреднённое по порогам IoU. Главная сводная метрика трекинга.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** TrackEval 1.3.0 (evaluation/hota_eval.py)
- **Оговорка:** Два усреднения — по объектам (HOTA) и по классам (HOTA_cls_av, аналог mHOTA в статьях). В одной таблице не смешивать.
- **Источник:** Luiten et al., IJCV 2021

## Где используется
- [Цель 0: заморозка воспроизводимого стенда трекера (7 видео, 30 Гц)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbench_freeze_v1)
- [Отложенное рождение по q_birth на обучающих видео (130/35/35) и его совмещение с буфером IoU↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_deferred_train_v1)
- [Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1)
- [Цель 6: вето рождений по q_birth в неизменённом XYWHByteTrack (парная точка по точности рождений)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_qbirth_v1)
- [Гейт рамок: сглаживание траектории и выбор рамки на выходе трекера (без обучения)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbox_smoothing_v1)
- [Каскад буферов IoU (C-BIoU) и буфер второй ассоциации на стенде XYWHByteTrack — HOTA на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fcbiou_cascade_v1)
- [Д1: лестница оракулов детекции — потолок HOTA для каждой стороны обнаружения (порог, ложные, рамки, невидимые, рождение)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fdetection_oracle_ladder_v1)
- [Сквозной трекер: рычаги надёжности предсказания (источник движения Калман/DIS, буфер IoU) и голова надёжности без мухи и с мухой — HOTA на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fe2e_motion_source_v1)
- [Н11: сигнатура состояния фовеального глаза FlyVis (с сетчаткой-ламиной) для повторного опознания объекта после потери трека↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_signature_reid_v1)
- [Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_stress)
- [Итоговый трекер линии B: каскад буферов IoU + отложенное рождение по q_birth + сшивка обрывков по геометрии↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fline_b_combined_v1)
- [Офлайн-сшивка треков стенда по голове повторного опознания (геометрия + внешность + вход глаза) — HOTA/IDF1/IDSw на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_appearance_v1)
- [ByteTrack против BoT-SORT и зоопарк из одиннадцати трекеров↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ftracker_zoo)
- [XYWHByteTrack: модель рамки с независимыми шириной и высотой↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_bytetrack)
- [Второй проход чистым IoU: возвращает FN, добавляет FP↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_iou_second)
