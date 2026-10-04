---
aliases:
- id_kept_pct
tags:
- тип/метрика
---

# сохранение ID при окклюзии

*identity preservation across occlusion transitions*

Доля окклюзионных переходов, на которых объект сохранил идентификатор при наличии рамки на обоих концах.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** scripts/evaluate_occlusion_transitions.py

## Где используется
- [[flyvis_stress]]
