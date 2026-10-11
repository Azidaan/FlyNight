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
- [Отложенное рождение по q_birth на обучающих видео (130/35/35) и его совмещение с буфером IoU↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_deferred_train_v1)
- [Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1)
- [Цель 6: вето рождений по q_birth в неизменённом XYWHByteTrack (парная точка по точности рождений)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_qbirth_v1)
- [Гейт рамок: сглаживание траектории и выбор рамки на выходе трекера (без обучения)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbox_smoothing_v1)
- [Буфер потерянного трека 30 против 150 кадров без мухи, та же сшивка MASA (А1, шаг после аудита запаса связей)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbuffer_extend_v1)
- [Каскад буферов IoU (C-BIoU) и буфер второй ассоциации на стенде XYWHByteTrack — HOTA на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fcbiou_cascade_v1)
- [Голова «удалить итоговую цепочку» после сшивки с MASA: фильтр без мухи против того же фильтра с мухой↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_fly_v1)
- [Голова удаления цепочки после MASA: ранжирование (AP) и порог при одинаковом бюджете потерь — муха против контроля↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1)
- [Запрет возврата потерянного трека на детекцию другого класса после потери > 30 кадров (ByteTrack, без мухи)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fclass_gate_v1)
- [Д1: лестница оракулов детекции — потолок HOTA для каждой стороны обнаружения (порог, ложные, рамки, невидимые, рождение)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fdetection_oracle_ladder_v1)
- [Сквозной трекер: рычаги надёжности предсказания (источник движения Калман/DIS, буфер IoU) и голова надёжности без мухи и с мухой — HOTA на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fe2e_motion_source_v1)
- [Н11: сигнатура состояния фовеального глаза FlyVis (с сетчаткой-ламиной) для повторного опознания объекта после потери трека↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_signature_reid_v1)
- [Трекер на мухе, этап 1, v2: при «жив» по мухе продолжать трек рамкой слабой детекции, а не предсказанием↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_tracker_lifecycle_v2)
- [Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_stress)
- [Последняя дешёвая проверка ветки удаления жизней: связь с цепочкой и внешность MASA поверх соседства — H1 / B1 / B2 (+ муха), данные разработки↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_chain_appearance_v1)
- [Контекст соседних треков для различения ложных жизней (вариант (а)) — головы H0 / H1 (+ соседство) / H2 (+ соседство + муха), данные разработки↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_context_neighbors_v1)
- [HOTA головы отдельной жизни (бустинг) при бюджете потерь 1 % — фильтр с мухой против того же фильтра без мухи (dev, test, val7)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_filter_boost_hota_v1)
- [Проверка фильтра жизней на новых val-видео (вариант (а)): без фильтра / фильтр без мухи / фильтр с мухой — ночь основная↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_filter_newval_v1)
- [Оракул удаления отдельных ложных жизней внутри готовых цепочек после MASA (dev): запас сверх удаления целых цепочек↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_filter_oracle_v1)
- [Оракул удаления отдельных ложных жизней внутри готовых цепочек после MASA (dev) — исправленная идентификация жизней↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_filter_oracle_v2)
- [Запас идеального удаления полностью ложных жизней на 97 новых val-видео (диагностика на сохранённых выходах)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_only_oracle_newval_v1)
- [Оракул того же действия, что у головы жизни: удаление полностью ложных жизней из исходного выхода после MASA (dev)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flife_only_oracle_v1)
- [Итоговый трекер линии B: каскад буферов IoU + отложенное рождение по q_birth + сшивка обрывков по геометрии↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fline_b_combined_v1)
- [Аудит А1: неисправленные разрывы свыше 30 кадров на готовом выходе после MASA и запас связей (dev, без мухи)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Flost_link_audit_v1)
- [Запас фильтрации готового выхода по меткам при IoU = 0,5 (вариант А): удаление целых ложных цепочек↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Foutput_filter_oracle_v1)
- [Сшивка обрывков без утечки разметки в кандидаты: внешность DINOv3-S против MASA-R50 и вклад мухи↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_appearance_models_v1)
- [Офлайн-сшивка треков стенда по голове повторного опознания (геометрия + внешность + вход глаза) — HOTA/IDF1/IDSw на обучающих видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_appearance_v1)
- [Сшивка обрывков: глобальное назначение пар вместо жадного (замена узла «сшиватель»)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_global_assign_v1)
- [ByteTrack против BoT-SORT и зоопарк из одиннадцати трекеров↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ftracker_zoo)
- [XYWHByteTrack: модель рамки с независимыми шириной и высотой↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_bytetrack)
- [Второй проход чистым IoU: возвращает FN, добавляет FP↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fxywh_iou_second)
