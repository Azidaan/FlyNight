---
aliases:
- 'З13: обучаемое считывание реальных областей по последствиям сопоставления'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 З13: обучаемое считывание реальных областей по последствиям сопоставления

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_region_assignment_v1`

Полный обучаемый региональный узел: основной критерий не пройден.

## Вопрос
Использует ли новый внутренний региональный узел сигнал D для полезного назначения лучше C, входа глаза и геометрии?

## Гипотеза
Нужен выбор реальной области с учётом соседних кандидатов и возможности не вмешаться, а не точность одного центра или бинарное присутствие.

## Ожидалось (до прогона)
Replace the internal FlyVis target readout: learned joint categorical region choice and explicit unmatched action, using frozen target-model coefficients and raw retinal activity pooled at real high-score same-class detection boxes. Preserve the surrounding detector, tracker and MASA. No learned core parameters in this trial. Identical action sets, labels, architecture, seeds, minibatch UIDs and optimizer forD,C,R1,geometry. Geometry uses the same D tensors but zeros both activity and target-model coefficients before embedding. C is the standard fly; D remains the retained trained core. R1 is the rich retinal-input control. New full readout vs no action, thenD vs each matched control.

## Что сделано
2026-10-25

## Получено
runs/z13_region_assignment_20261025/recovery2/training/report.json → paired: {'D-baseline': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}, 'D-C': {'mean_per_video': -3.141057580732824e-05, 'ci95': [-0.0001888788712676626, 0.00011488295758832168], 'better': 2, 'n': 26}, 'D-R1': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}, 'D-geometry': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}};12моделей,трисидa,порог fit20; первичные строки и все контроли сохранены.

## Интерпретация
Полный внутренний узел оценён по точным однокадровым назначениям. Это не HOTA и не изоляция одного свойства ядра. Контроль геометрии отделяет пользу внешнего применения; C/R1 отделяют вклад D.

## Решение
Complete criteria pass -> separate matched closed-loop test with frozen thresholds and equal MASA/raw inputs. Fail -> preserve all controls and negative seed results, do not run expensive HOTA for this version. Technical failure -> preserve sources/logs, repair only verified service issue within original phase/global budgets. Основной итог: False

## Нельзя заявлять
The oracle used truth and is equally available to non-fly representations. Low-resolution activity may not discriminate real instances; geometry may explain all utility. Some targets have no high-score detection, so this action cannot help them. One-step training omits feedback and later stages. The protocol changes the complete readout output/learning target, not an isolated causal test of a single scalar. StandardC vsD core histories differ by design. Sparse labels and repeatedly reused train26 limit generalization.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (714f73a37bc57d1229df8ac8d1ce9b6ea1b9dcaa) — Проверен полный равный пилот регионального считывателя.
  > Три контролируемых этапа зарегистрированы до обучения.

## Критерии приёмки (записаны до прогона)
- ✅ All130 natural-video labels: reuse completed26 gate rows exactly, replay remaining104 with float32-exact unchanged outputs. Preserve all query UID/video/frame/track/label bindings. Same supported real choices from the pure observable API. Each D/C/R1 cache reproduces the saved original centers/scores and identical ROI masks. Every phase has its own independent controller and price gate; no completed arm/chunk/model repeated. No core training/dev/HOTA/heldout75/download.  
  *runs/z13_region_assignment_20261025/recovery2/independent_audit.json*
- ❌ Average per-video net assignment effect across all3seeds, then paired-video95%CI. D-baseline mean>=0.02 and lower>0; D-C,D-R1,D-geometry all lower>0. Report each seed and all counts, unknowns and chosen thresholds. Passing is a gate for separately budgeted closed-loop HOTA, never automatic HOTA or final novelty.  
  *runs/z13_region_assignment_20261025/recovery2/training/report.json*

## Доказательства
- Предварительные критерии, равные контроли и бюджеты: `runs/z13_region_assignment_20261025/protocol.json` · 
- Пройденная мишень действия до обучения: `runs/z13_region_choice_gate_20261025/recovery1/report.json` · 

## Код
- `tracking/z13_region_assignment.py`
- `tests/test_z13_region_assignment.py`
- `scripts/z13_region_assignment_20261025.py`
- `scripts/z13_region_assignment_console.ps1`

