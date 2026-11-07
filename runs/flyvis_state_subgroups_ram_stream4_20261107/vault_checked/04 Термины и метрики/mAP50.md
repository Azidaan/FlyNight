---
aliases:
- mAP50
- mAP@0.5
- mAP@0.50
tags:
- тип/метрика
---

# mAP50

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*mean Average Precision @ IoU 0.5*

Средняя по классам точность детектора при пороге совпадения IoU 0.5. Главная метрика качества детектора.

- **Лучше:** higher
- **Диапазон:** [0, 1]
- **Чем считается:** Ultralytics val / evaluation/detection_metrics.py
- **Оговорка:** Разброс между сидами: 0.0144 на 12 эпохах, 0.0007 на 100. Различия меньше разброса — не результат.

## Где используется
- [DRB v1 и v2: синергия ветвей LQA + FGA↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fdrb_v1_v2)
- [Фиксированный частотный фильтр (срез снизу, ноль обучаемых частот)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffixed_filter)
