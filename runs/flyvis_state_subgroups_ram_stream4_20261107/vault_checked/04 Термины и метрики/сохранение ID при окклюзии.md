---
aliases:
- id_kept_pct
tags:
- тип/метрика
---

# сохранение ID при окклюзии

↑ [[04 Термины и метрики/Термины и метрики|Термины и метрики]]

*identity preservation across occlusion transitions*

Доля окклюзионных переходов, на которых объект сохранил идентификатор при наличии рамки на обоих концах.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** scripts/evaluate_occlusion_transitions.py

## Где используется
- [Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fflyvis_stress)
