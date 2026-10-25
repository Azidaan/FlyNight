---
aliases:
- Повторная постановка полного действия FlyVis после математического допуска
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🔴 Повторная постановка полного действия FlyVis после математического допуска

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_box_supervision_followup_v1`

Обучение последствиям полной рамки: критерий не пройден.

## Вопрос
Поможет ли неизменный внутренний считыватель при обучении последствиям именно полной рамки?

## Гипотеза
Рабочая гипотеза: метки прежнего категориального изменения стоимости не переносятся автоматически на другую геометрию действия. Обучение полной рамке при равных признаках может изменить полезность.

## Ожидалось (до прогона)
Replace the internal regional readout action semantics and corresponding supervision: selected real region now transfers complete XYWH and uses unchanged native get_dists, instead of a categorical cost row. Explicit unmatched still sets this row costs to1; fixed unchanged logit still means no action. No row or target identity is supplied at runtime. Reuse all D/C/R1 raw68 regional activity, observed target coefficients, original assignment context6, eight geometry inputs and query UID order. Same24heads (4representations x context/blank x3seeds),1150steps,batch128,Adam0.001,signed good-harm loss, fixed fit20 threshold grid. Train84; calibrate20; evaluate26. Only outcome labels and applied candidate-box action change together. No new feature/core computation or GPU.

## Что сделано
2026-10-25

## Получено
runs/z13_box_supervision_followup_20261025/training/report.json → paired: {'Dbox-baseline': {'mean_per_video': -0.0006249487512164997, 'ci95': [-0.0023736802413272997, 0.0003526666001311032], 'better': 3, 'n': 26}, 'Dbox-D_blank': {'mean_per_video': -1.3785918936725732e-05, 'ci95': [-0.0023735852493426838, 0.0024309203111224707], 'better': 11, 'n': 26}, 'Dbox-C_context': {'mean_per_video': 0.0010138165488213185, 'ci95': [-0.001390441358282744, 0.004846096344449262], 'better': 8, 'n': 26}, 'Dbox-R1_context': {'mean_per_video': -0.000562999085612896, 'ci95': [-0.0026022159308863716, 0.000919733743218069], 'better': 6, 'n': 26}, 'Dbox-geometry_context': {'mean_per_video': -0.0007133660810131399, 'ci95': [-0.002403846153846154, 0.00020341608576902696], 'better': 2, 'n': 26}, 'Dbox-Dcategorical': {'mean_per_video': 0.0013043221109191995, 'ci95': [-0.0003168299055562982, 0.0034219640718165753], 'better': 8, 'n': 26}}

## Интерпретация
Одинаковое внутреннее считывание обучено фактическим действиям полной рамки с равными контролями. Изменены сопряжённые выход и метки, не доказана одна причина прежнего провала. Повторные train26, первые назначения, не HOTA.

## Решение
Pass -> separately preregister matched runtime and closed-loop HOTA with native/C/R1/no-FlyVis controls. Fail -> preserve this exact action-supervision implementation and stop this categorical-to-box family; no new loss/guard/threshold sweep on26. A different internal mechanism needs its own measured rationale. Основной критерий: False

## Нельзя заявлять
The old representations may not identify absent targets even with correct action labels. Severe imbalance may yield no action; the same geometry controls may suffice. Action reserve does not imply learnability. Both20fit and26evaluation repeatedly used, so this is exploratory, not independent confirmation. No manual guards or threshold edits after evaluation. Это адаптация цели действия, не воспроизведение полной пространственной сети SiamMOT.

## Критерии приёмки (записаны до прогона)
- ✅ Same original130raw outputs bit-exact;37296queries/26600known; exact regional choices/context/feature bindings. Full-box batch distances checked against native scalar for>=128actions. Reuse4immutable completed videos. Price after2new videos:2*mean*remaining+60 within fresh cache budget3600;training firstmodel2*time*23+60 within1800. All24model predictions independently reproduced from saved weights, labels and thresholds; three seeds and controls unchanged.  
  *runs/z13_box_supervision_followup_20261025/independent_audit.json*
- ❌ Preregister D_context as primary, no choosing blank after evaluation: net baseline effect mean>=2pp and all6paired95%video lower>0 versus baseline,D_blank,C_context,R1_context,geometry_context,previous Dcontext categorical action. Report all24models and frozen fit20 thresholds, including zero-action and unfavorable outcomes.  
  *runs/z13_box_supervision_followup_20261025/training/report.json*

## Доказательства
- Новая просьба автора, отдельный бюджет и прежняя научная постановка: `runs/z13_box_supervision_followup_20261025/protocol.json` · 
- Пройденная предварительная математика, не прогноз модели: `runs/z13_math_priority_20261025/report.json` · 

