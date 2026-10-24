---
tags:
- журнал
дата: '2026-10-25'
время: 02:53
---

# 2026-10-25 02:53 — итог мишени прямого выбора области `z13_region_choice_gate_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_region_choice_gate_20261025/recovery1/report.json → {'regions_baseline': {'mean_per_video': 0.0635804966892227, 'ci95': [0.05265237331078239, 0.07518989745701771], 'better': 26, 'n': 26}, 'regions_Dpeak3': {'mean_per_video': 0.04724720647232502, 'ci95': [0.03863916126055624, 0.055851861632864076], 'better': 26, 'n': 26}}; {'rows': 8326, 'known': 5822, 'unknown': 2504, 'supported_regions': 9178}

*Вывод.* Это оракульная достижимость внутри одной строки назначения, одинаковая для любого представления. Польза FlyVis и HOTA не измерены.

*Решение.* Pass -> register raw-activity region learner with equal geometry-only/R1/C controls, exact baseline reproduction and fit-only hyperparameters before GPU extraction. Fail -> stop this interface; choose a different internal FlyVis node, no retrospective weakening of the2pp gate. Измеренный итог критерия: True

HOTA не измерялась, процент к стенду не определён. Доля потолка для обученного узла ещё не измерена. Оракул не входит в лучшие улучшения. Нового архитектурного узла нет: измерена мишень перед реализацией.
