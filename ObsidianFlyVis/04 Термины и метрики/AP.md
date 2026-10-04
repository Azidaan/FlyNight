---
aliases:
- AP
- AUPRC
- PR-AUC
- average_precision
tags:
- тип/метрика
---

# AP

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*average precision*

Площадь под кривой точность–полнота. Для гейтов q_match — средняя точность классификатора пар «одна траектория / разные».

- **Лучше:** higher
- **Диапазон:** [0, 1]

## Где используется
- [Контроль «тот же вход для DIS»: помогает ли сетчатка-ламина только мухе или любому оценщику движения (голова надёжности, тёмные/светлые ролики)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fdis_retina_control_v1)
- [Цель 4: головы q_det / q_match / q_birth / q_alive без FlyVis — ранжирование, калибровка, контроль монотонного преобразования↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fevent_heads_v1)
- [Нулевая модель коннектома: муха с перемешанными связями (степени и знаки сохранены) против настоящей — в повторном опознании поверх входа глаза↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_connectome_null_v1)
- [Н11, контроль того же объёма информации: схема мухи против её собственного входа (721 фасетка, то же поле зрения) и разрез по времени суток↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_matched_controls_v1)
- [Н11: сигнатура состояния фовеального глаза FlyVis (с сетчаткой-ламиной) для повторного опознания объекта после потери трека↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_signature_reid_v1)
- [Н3: считывание FlyVis как признак событий без пары — q_alive (фовеальный глаз на кадре потери) и q_birth (тайловое считывание рамки рождения)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_event_features_v1)
- [Цель 2: фовеация — глаз FlyVis на каждый трек XYWHByteTrack, то же считывание T4/T5 и all34↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_foveation_readout_v1)
- [Цель 2: считывание внутренней активности FlyVis (T4/T5, 34 типа) для q_match против декодированного потока и shuffled↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_internal_readout_v1)
- [Диагностика: нелинейная модель на признаках считывания FlyVis (tiled и фовеация)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_readout_nonlinear_diag)
- [Н9/М1: фронт-энд «фоторецептор + ламина» перед глазом FlyVis — восстанавливается ли сигнал мухи на тёмных видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_retina_frontend_v1)
- [Н9/М1, ступень B: муха с сетчаткой-ламиной на обучающих видео — убирает ли ламина ночной провал мухи в голове надёжности (разбивка тёмные/светлые)↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_retina_stageB_v1)
- [Н2: голова достоверности движения — предсказывает ли остаточный поток фовеального глаза FlyVis большую ошибку предсказания фильтра↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fmotion_reliability_v1)
- [30-герцовый псевдоразмеченный гейт q_match↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo)
- [Дешёвый гейт q_match: FlyVis поверх DIS на парах детекций↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_gate)
