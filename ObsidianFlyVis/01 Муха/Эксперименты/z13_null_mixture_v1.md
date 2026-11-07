---
aliases:
- 'Применение считывателя: NULL как сохранение решения трекера'
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

# 🟢 Применение считывателя: NULL как сохранение решения трекера

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_null_mixture_v1`

NULL как сохранение строки: критерий не пройден.

## Вопрос
Поможет ли явное сохранение исходной строки в ветви NULL при той же карте мухи и без нового штрафа неопределённости?

## Гипотеза
Рабочая гипотеза: неопределённый ответ не должен автоматически означать повышение стоимости всех кандидатов. Это новый контракт применения, не доказанная ошибка прежнего смысла «цели нет».

## Ожидалось (до прогона)
New readout application contract: NULL probability leaves the original association row unchanged instead of raising every supported cost. Keep original mean maps, original unit-clock state, weights, normalization, null logit0, support and alpha0.25. No uncertainty penalty from the failed parent is combined.8paired conditions C/D/R1/geometry timesold/null_mix.

## Что сделано
2026-10-25

## Получено
runs/z13_null_mixture_20261025/report.json → paired: {'Dnull-baseline': {'mean_per_video': 0.00019830489785028374, 'ci95': [-0.002111283003094532, 0.0026917646209558116], 'better': 7, 'n': 26}, 'Dnull-Dold': {'mean_per_video': 0.0012189861067708726, 'ci95': [-1.883107529268459e-05, 0.0026875935350595213], 'better': 7, 'n': 26}, 'Dnull-Cnull': {'mean_per_video': 0.0005359971535842912, 'ci95': [-7.430727759469521e-05, 0.0013174310683777884], 'better': 4, 'n': 26}, 'Dnull-R1null': {'mean_per_video': 0.0004977238788049369, 'ci95': [-0.0014793798669394524, 0.0027844899995528435], 'better': 7, 'n': 26}, 'Dnull-geometrynull': {'mean_per_video': 0.0004324566833872398, 'ci95': [-0.0003052646692945973, 0.0012759048126848127], 'better': 5, 'n': 26}, 'null_effect_D-geometry': {'mean_per_video': 0.0012402590618852028, 'ci95': [6.392986411376339e-05, 0.002576045016189404], 'better': 7, 'n': 26}}

## Интерпретация
Изменён только контракт применения NULL при той же исходной карте: неиспользованная масса вероятности сохраняет исходную строку. Это8парных условий первых назначений, не HOTA и не самостоятельное доказательство модернизации ядра FlyVis.

## Решение
Pass -> distinct runtime contract parity/price and frozen matched closed-loop stage; still separate application from intrinsic FlyVis improvement. Fail -> preserve this exact NULL expectation contract and do not retune alpha/probability thresholds. Observed-negative and four-bin gates remain inconclusive and do not advance. Primary: False

## Нельзя заявлять
NULL conflates absence and uncertainty; suppressing its rejection effect can leave a false or wrong baseline match untouched and reduce beneficial rejection. Generic reduced intervention may explain everything without FlyVis, so geometry/R1/C and interaction controls are mandatory. One-step naturaltrain26 is repeatedly used; no HOTA or claim of causal explanation of all previous failures. This is a scientifically new interpretation of the readout NULL, not service repair of the previous declared no-candidate cost. The previous contract could intentionally reject all candidates when target absent. The new one treats NULL as no trustworthy intervention; it can preserve incorrect baseline matches. It tests an interface mechanism, not an intrinsic improvement of FlyVis core.

## Критерии приёмки (записаны до прогона)
- ✅ Three unit checks: fullNULLnoop, explicit expected-cost mixture, q1oldformula/alpha0exact. All26baseline outputs float32exact; everyold branch action and geometry exactly reproduce the prior mean-region report; old center<0.5px/score<0.001 unchanged. First2CPUreplays conservative old2*mean_seconds*44+120 estimate fits remaining900seconds or preservepartial/stop. Primary rows/hashes/paired intervals independently verified.  
  *runs/z13_null_mixture_20261025/independent_audit.json*
- ❌ Dnull-baseline and Dnull-Dold means>=2pp; all6paired-video95%lower>0 includingCnull,R1null,geometrynull and (Dnull-Dold)-(geometrynull-geometryold). A shared improvement without a D-specific gain is an application change, not a FlyVis result. No favorable subset/alpha/null-logit selection.  
  *runs/z13_null_mixture_20261025/report.json*

## Доказательства
- Одна замена контракта, контроли и критерии до расчёта: `runs/z13_null_mixture_20261025/protocol.json` · 

