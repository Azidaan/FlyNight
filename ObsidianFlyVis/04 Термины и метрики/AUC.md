---
aliases:
- AUC
- ROC-AUC
- roc_auc
tags:
- тип/метрика
---

# AUC

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*area under ROC curve*

Площадь под ROC-кривой: вероятность, что случайная верная детекция получит балл выше случайной ложной.

- **Лучше:** higher
- **Диапазон:** [0, 1]

## Где используется
- [Карта качества: три гейта ночи (сверх уверенности детектора)↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fquality_map_gates)
- [Проверка вклада FlyVis в гибридное узнавание цели на train26 — замороженная версия v2, одна оценка↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_hybrid_fly_confirm_v1)
- [Гибридное узнавание цели — базовая ветвь внешнего вида MASA и ветвь FlyVis, решение о переносе рамки↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_hybrid_identity_v1)
- [Гибридное узнавание цели v2 — точнее образец внешнего вида и обучаемое сравнение компонент, ветвь FlyVis и решение о переносе↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_hybrid_identity_v2)
- [Двухступенчатое считывание FlyVis с накопленным образом цели трека — узнавание цели и решение о переносе↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_identity_gate_bank_v1)
- [Двухступенчатое считывание FlyVis с согласованностью движения T4/T5 — узнавание цели по направлению движения и решение о переносе↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_identity_gate_motion_v1)
- [Двухступенчатое считывание FlyVis — плотное узнавание цели среди областей и решение о переносе рамки↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_identity_gate_readout_v1)
- [Двухступенчатое считывание FlyVis со сравнением области с соседями — относительное узнавание цели и решение о переносе↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_identity_gate_relative_v1)
- [Двухступенчатое считывание FlyVis с памятью о прошлой цели трека — узнавание цели и решение о переносе рамки↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_identity_gate_tagged_v1)
- [Потолок информации считывания FlyVis — различают ли сохранённые признаки полезную и вредную рамку↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_information_ceiling_v1)
