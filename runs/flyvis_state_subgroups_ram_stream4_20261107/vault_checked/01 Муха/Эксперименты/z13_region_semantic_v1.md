---
aliases:
- Правильный ответ как цель обучения регионального считывателя
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

# 🟢 Правильный ответ как цель обучения регионального считывателя

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_region_semantic_v1`

Правильный ответ как цель: основной критерий не пройден.

## Вопрос
Поможет ли обучение всем правильным невредным решениям вместо только редких улучшений, с допустимым правильным бездействием?

## Гипотеза
Рабочая гипотеза: обучение редкому улучшению может не научить различать сам правильный ответ. Наличие больше положительных меток не доказывает эту причину.

## Ожидалось (до прогона)
Change the acceptable action target, not the likelihood family or input: include every own-correct, nonharmful action instead of only strictly beneficial changes. Include unchanged in the positive marginal whenever the original own assignment was correct. If neither an action nor baseline is correct, unchanged is the conservative expressible fallback.

## Что сделано
2026-10-25

## Получено
runs/z13_region_semantic_20261025/training/report.json → paired: {'D-baseline': {'mean_per_video': -5.4094990803851555e-05, 'ci95': [-0.0001622849724115547, 0.0], 'better': 0, 'n': 26}, 'D-C': {'mean_per_video': -5.4094990803851555e-05, 'ci95': [-0.0001622849724115547, 0.0], 'better': 0, 'n': 26}, 'D-R1': {'mean_per_video': 0.001180213321976607, 'ci95': [-0.0008993772312237543, 0.0034153941851931082], 'better': 8, 'n': 26}, 'D-geometry': {'mean_per_video': 0.00027256176720871447, 'ci95': [-0.0008833744504075092, 0.0019962625791980727], 'better': 3, 'n': 26}, 'Dsemantic-Dutility': {'mean_per_video': -7.50130387537185e-05, 'ci95': [-0.00023317497196494508, 6.761351254156344e-05], 'better': 1, 'n': 26}};12моделей,3сида, признаки прежние, fit84/fit20/train26.

## Интерпретация
Изменено множество допустимых ответов в той же функции обучения. Правильное бездействие остаётся допустимым. Результат не доказывает общую причину отказов; это первые назначения, не HOTA.

## Решение
Pass -> separately budgeted frozen-policy closed-loop parity/price/HOTA with original/no-fly controls. Fail -> preserve this supervised-target implementation. No further scalar loss/threshold variants under this action; choose a different internal representation/state mechanism only with a measured cheap prerequisite. Измеренный основной критерий: False

## Нельзя заявлять
Correct neutral actions need not improve the stand; expanding the acceptable set may dilute rare useful corrections or leave an easy unchanged solution. Fixed response/mean-pooled representation may remain insufficient. One-row outcomes omit feedback; repeatedtrain26 is exploratory, not HOTA or proven causal diagnosis. HOTA не измерялась; доля потолка стенда неизвестна.

## Прежние версии (до исправлений)
- **Нельзя заявлять**, 2026-10-25 (7c8276f74847ccd59a3593d0da5658d00a0858bc) — Проверен полный пилот семантики допустимых ответов.
  > Correct neutral actions need not improve the stand; expanding the acceptable set may dilute rare useful corrections or leave an easy unchanged solution. Fixed response/mean-pooled representation may remain insufficient. One-row outcomes omit feedback; repeatedtrain26 is exploratory, not HOTA or proven causal diagnosis.

## Критерии приёмки (записаны до прогона)
- ✅ All12utility models exactly replayed; actual fit84 two-query CPU forward/backward finite. Mask/NULL/fallback gradients and utility special-case value+gradient equivalence tested. Same full caches/sources/UID batches, no completed training repeated. First full model2*seconds*11+60 fits original1800s; independent controller only own tree. Independently re-infer12new models/rows/calibration and paired intervals before final acceptance.  
  *runs/z13_region_semantic_20261025/independent_audit.json*
- ❌ Dsemantic-baseline and Dsemantic-Dutility means both>=2pp; all5paired-video95%lower>0: baseline,utilityD,semanticC,semanticR1,semanticgeometry. All26videos and3seeds. Same net-beneficial assignment metric; semantic label accuracy cannot rescue a failed primary.  
  *runs/z13_region_semantic_20261025/training/report.json*

## Доказательства
- Предварительные критерии, неизменные контроли/пакеты и семантика цели: `runs/z13_region_semantic_20261025/protocol.json` · 

