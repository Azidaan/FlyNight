---
aliases:
- HOTA
- HOTA_cls_av
- mHOTA
tags:
- тип/метрика
---

# HOTA

*Higher Order Tracking Accuracy*

Совместная оценка качества обнаружения и ассоциации: среднее геометрическое DetA и AssA, усреднённое по порогам IoU. Главная сводная метрика трекинга.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** TrackEval 1.3.0 (evaluation/hota_eval.py)
- **Оговорка:** Два усреднения — по объектам (HOTA) и по классам (HOTA_cls_av, аналог mHOTA в статьях). В одной таблице не смешивать.
- **Источник:** Luiten et al., IJCV 2021

## Где используется
- [[bench_freeze_v1]]
- [[birth_deferred_train_v1]]
- [[birth_policy_deferred_v1]]
- [[birth_policy_qbirth_v1]]
- [[box_smoothing_v1]]
- [[cbiou_cascade_v1]]
- [[detection_oracle_ladder_v1]]
- [[e2e_motion_source_v1]]
- [[fly_signature_reid_v1]]
- [[flyvis_stress]]
- [[line_b_combined_v1]]
- [[stitch_appearance_v1]]
- [[tracker_zoo]]
- [[xywh_bytetrack]]
- [[xywh_iou_second]]
