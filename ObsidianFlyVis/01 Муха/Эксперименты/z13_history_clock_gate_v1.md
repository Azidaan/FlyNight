---
aliases:
- 'Временное состояние FlyVis: мишень искусственных повторов образца'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P3
- метка/fly-problem-P8
- муха
исход: положительный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Временное состояние FlyVis: мишень искусственных повторов образца

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_history_clock_gate_v1`

Мишень повторного времени достаточна для отдельной пробы.

## Вопрос
How many unchanged natural queries inherit duplicate leading exemplar timestamps or missing frame intervals, and are zero-time entries exact leading copies with the same crop?

## Гипотеза
Рабочая гипотеза: повтор кадров для заполнения четырёх мест образца может добавлять состояние без нового времени видео. Наличие этого механизма и влияние на выход — разные вопросы.

## Ожидалось (до прогона)
Candidate internal timing change: do not advance recurrent state on zero-elapsed duplicate exemplar-padding entries. Preserve every genuine observed/search frame, state across genuine steps, eye geometry, learned parameters and old normalization/readout. This gate only checks whether the trigger has enough exposure before any GPU pilot.

## Что сделано
2026-10-25

## Получено
runs/z13_history_clock_gate_20261025/report.json → affected_query_fraction: {'mean_per_video': 0.039541515812622004, 'ci95': [0.02845169836048589, 0.052124133460154085], 'better': 22, 'n': 26};37296запросов,26600известных; все нулевые интервалы — точные ведущие копии.

## Интерпретация
Измерена только частота триггера искусственного времени. Не измерены ошибка активности, улучшение представления или HOTA. Положительный статус относится только к наличию мишени.

## Решение
If enough exposure and exact leading-copy semantics are verified, separately register a bounded frozen-core skip-padding prototype with same-history C/D/R1 controls and saved old responses; no new core training. Otherwise preserve the specific insufficient trigger and choose another state/input node. No HOTA from counts. Admission: True

## Нельзя заявлять
The initial fade-in may already be close to a steady state, so extra constant-input steps can have negligible influence or even help initialization. Changing timing can shift the distribution learned by D. Frequent padding alone proves neither failure causality nor a useful replacement. Thresholds and core weights are not changed by this gate.

## Критерии приёмки (записаны до прогона)
- ✅ Each query/episode/frame/crop binding valid; nondecreasing actual timestamps. Count duplicates only before last exemplar and verify identical frame,crop,hex data. Admission for a skip-padding prototype only if mean-video fraction of known evaluation queries with such duplicates>=0.02 and paired lower95%CI>0; this is an affected-subset upper-bound inventory, not evidence of improvement. Other gaps reported but not silently repaired.  
  *runs/z13_history_clock_gate_20261025/report.json*

## Доказательства
- Действие, дешёвая мишень и критерий до измерения: `runs/z13_history_clock_gate_20261025/protocol.json` · 

