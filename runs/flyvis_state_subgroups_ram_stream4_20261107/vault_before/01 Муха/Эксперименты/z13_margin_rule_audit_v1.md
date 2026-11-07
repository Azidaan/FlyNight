---
aliases:
- Прямой порог пользы при прежних24считывателях FlyVis
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Прямой порог пользы при прежних24считывателях FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_margin_rule_audit_v1`

Все24модели воспроизведены. Замена одного порога не прошла допуск на20обучающих видео.

## Вопрос
Объясняет ли правило выбора часть провала при неизменных оценках?

## Гипотеза
Сравнение с нулём и прямой порог могут устранить влияние числа кандидатов; польза не гарантирована.

## Ожидалось (до прогона)
Exact saved predictions/rows, weights and labels of all24 models; train26 only reproduction, no new policy evaluation or selection there. Models trained on84; five fixed video folds within separate fit20. Calibrate each policy on16videos, evaluate4. Same seeds/arms/context. No best-seed selection. Both threshold grids11values fixed before calculation. No AUC gate.

## Что сделано
2026-10-25

## Получено
runs/z13_margin_rule_audit_20261025/report.json → paired: {'Dmargin-baseline': {'mean_per_video': -0.0005885676673375595, 'ci95': [-0.0016499332137694623, 0.00017618512373102107], 'better': 2, 'n': 20}, 'Dmargin-Dprobability': {'mean_per_video': 0.00020199829424558357, 'ci95': [-0.0003964646464646465, 0.0008366765930460197], 'better': 3, 'n': 20}, 'Dmargin-C_context': {'mean_per_video': -0.001109294057527968, 'ci95': [-0.0029696235431816136, 0.0007869825341146364], 'better': 5, 'n': 20}, 'Dmargin-R1_context': {'mean_per_video': -0.0012705382029377984, 'ci95': [-0.0025580377312962865, -2.520811216727924e-05], 'better': 3, 'n': 20}, 'Dmargin-geometry_context': {'mean_per_video': -0.0005885676673375595, 'ci95': [-0.0016499332137694623, 0.00017618512373102107], 'better': 2, 'n': 20}, 'Dmargin-D_blank': {'mean_per_video': -0.0008312218141451715, 'ci95': [-0.0020355154787964234, 0.00041091955938968636], 'better': 4, 'n': 20}}

## Интерпретация
Изолировано только правило выбора, оценки и действие прежние. Прямой порог не дал достаточной пользы. Интервалы условны на повторныхfit20 и фиксированном разбиении; новая функция ошибки не проверялась.

## Решение
Не продвигать только замену порога на26/HOTA. Перейти к отдельно зарегистрированной проверке нового обучения отказу наfit104; провал правила не доказывает отсутствие информации.

## Нельзя заявлять
Repeated development fit20; overlapping calibration folds, bootstrap is descriptive conditional on this partition, not an independent certification. New fit104 learning requires its own preregistered budget. Unknown outcomes excluded from net but counts retained.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (a9c823ec441a4cdf6dca998a347a70840c4dfc20) — Завершённая проверка по неизменному протоколу; первичные строки и интервалы независимо сверены.
  > 
- **Получено**, 2026-10-25 (a9c823ec441a4cdf6dca998a347a70840c4dfc20) — Завершённая проверка по неизменному протоколу; первичные строки и интервалы независимо сверены.
  > 
- **Интерпретация**, 2026-10-25 (a9c823ec441a4cdf6dca998a347a70840c4dfc20) — Завершённая проверка по неизменному протоколу; первичные строки и интервалы независимо сверены.
  > 
- **Решение**, 2026-10-25 (a9c823ec441a4cdf6dca998a347a70840c4dfc20) — Завершённая проверка по неизменному протоколу; первичные строки и интервалы независимо сверены.
  > 

## Критерии приёмки (записаны до прогона)
- ✅ Exact saved predictions/rows, weights and labels of all24 models; train26 only reproduction, no new policy evaluation or selection there.  
  *runs/z13_margin_rule_audit_20261025/reproduction.json*
- ❌ Margin-only advancement requires mean D_context net>=.02 and paired-video95%lower>0 against zero, old_probability D, C, R1, geometry_context. Otherwise do not promote margin-only to26/HOTA. Failure does not test the new loss or prove absent information.  
  *runs/z13_margin_rule_audit_20261025/report.json*

## Доказательства
- Критерии до расчёта: `runs/z13_margin_rule_audit_20261025/protocol.json` · 

