---
aliases:
- IDF1
tags:
- тип/метрика
---

# IDF1

*ID F1 score*

F-мера по идентификаторам: доля детекций, отнесённых к правильной траектории на всём протяжении. Чувствительна к сохранению личности объекта.

- **Лучше:** higher
- **Единица:** %
- **Диапазон:** [0, 100]
- **Чем считается:** bdd100k / TrackEval
- **Оговорка:** Два конвейера (bdd100k и TrackEval) расходятся на ~0.07–0.14 пункта; числа из разных конвейеров в одной строке не сравнивать.

## Где используется
- [[birth_deferred_train_v1]]
- [[birth_policy_deferred_v1]]
- [[birth_policy_qbirth_v1]]
- [[cbiou_cascade_v1]]
- [[e2e_motion_source_v1]]
- [[flyvis_stress]]
- [[infra_bdd_baseline]]
- [[line_b_combined_v1]]
- [[stitch_appearance_v1]]
- [[tracker_thresholds]]
- [[tracker_zoo]]
- [[wavelet_motion]]
- [[xywh_bytetrack]]
