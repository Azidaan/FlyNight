---
aliases:
- 'З13: запас считывания выбора реальной области вместо центра'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P8
- муха
исход: положительный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 З13: запас считывания выбора реальной области вместо центра

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_region_choice_gate_v1`

Запас прямого выбора области прошёл критерий подготовки обучаемого узла.

## Вопрос
Есть ли достаточный дополнительный запас полезных назначений у прямого выбора реальной области?

## Гипотеза
Даже идеального бинарного присутствия при прежнем ранжировании недостаточно; модель цели может нуждаться в другом выходе и конкуренции реальных областей.

## Ожидалось (до прогона)
For each unchanged tracker row: original cost vector, NULL vector all1, and each visible same-class region with one-hot probability1. With the unchanged cost=1-det_conf*prob interface, chosen cost=1-det_conf, every other cost1. Solve the whole original assignment. Extreme distributions are only an optimistic diagnostic of that interface. No true identity, target box or query outcome enters action generation.

## Что сделано
2026-10-25

## Получено
runs/z13_region_choice_gate_20261025/recovery1/report.json → {'regions_baseline': {'mean_per_video': 0.0635804966892227, 'ci95': [0.05265237331078239, 0.07518989745701771], 'better': 26, 'n': 26}, 'regions_Dpeak3': {'mean_per_video': 0.04724720647232502, 'ci95': [0.03863916126055624, 0.055851861632864076], 'better': 26, 'n': 26}}; {'rows': 8326, 'known': 5822, 'unknown': 2504, 'supported_regions': 9178}

## Интерпретация
Это оракульная достижимость внутри одной строки назначения, одинаковая для любого представления. Польза FlyVis и HOTA не измерены.

## Решение
Pass -> register raw-activity region learner with equal geometry-only/R1/C controls, exact baseline reproduction and fit-only hyperparameters before GPU extraction. Fail -> stop this interface; choose a different internal FlyVis node, no retrospective weakening of the2pp gate. Измеренный итог критерия: True

## Нельзя заявлять
An oracle can exploit detections without a fly; the real readout may not distinguish them. One-row shadow choices omit feedback, simultaneous interventions and subsequent association stages. One-hot probabilities are optimistic extremes. A larger reserve is necessary for this proposed implementation, not sufficient for training success or scientific novelty.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (bf7e986c5e300c2ec89db1cc30efd97bb8842dbe) — Проверен полный оракульный запас реальных областей.
  > Оракульная проверка мишени до реализации обучаемого считывателя.

## Критерии приёмки (записаны до прогона)
- ✅ 26 original tracker outputs float32-exact; predictions, causal identities, labels and UIDs identical to saved one-step replay. First alternative equals unchanged assignment exactly. Do not alter original state or costs. First2 videos:2*mean_elapsed*24+60 must fit remaining600s budget. Completed videos never repeated.  
  *runs/z13_region_choice_gate_20261025/recovery1/independent_audit.json*
- ✅ Mean-video any-beneficial reserve>=0.02 and its lower95%paired-videoCI>0, plus categorical-reserve minus Dthreepeak-reserve lower95%paired-videoCI>0. Failing gate stops this specific output interface before learned feature extraction. Passing only warrants registration of a matched learned region head; it is not FlyVis benefit or HOTA.  
  *runs/z13_region_choice_gate_20261025/recovery1/report.json*

## Доказательства
- Действие и критерии записаны до расчёта: `runs/z13_region_choice_gate_20261025/protocol.json` · 

## Код
- `tracking/z13_region_choice.py`
- `tests/test_z13_region_choice.py`
- `scripts/z13_region_choice_gate_20261025.py`
- `scripts/z13_region_choice_gate_console.ps1`

