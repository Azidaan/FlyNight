---
aliases:
- Уверенность считывателя по опоре признаков в наблюдённом образце
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🔴 Уверенность считывателя по опоре признаков в наблюдённом образце

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_ridge_uncertainty_v1`

Условная неопределённость: критерий не пройден.

## Вопрос
Улучшит ли штраф за неопределённость модели цели полезные назначения сверх прежней D и одинакового штрафа всем областям?

## Гипотеза
Рабочая гипотеза: часть высоких неверных ответов может быть экстраполяцией вне признаков образца. Это не установленная причина; уверенная ошибка идентичности может остаться.

## Ожидалось (до прогона)
Change only region readout confidence: subtract1.96times the template-only conditional posterior standard deviation from the old standardized region response. Mean target model and original unit-clock core/history unchanged. Per-query constant-penalty control uses the same full-field mean standard deviation. Old/variable/constant forC/D/R1 plusunchanged geometry,10arms; same fixedalpha0.25 and null0.

## Что сделано
2026-10-25

## Получено
runs/z13_ridge_uncertainty_20261025/replay/report.json → paired: {'Dposterior-baseline': {'mean_per_video': -0.0010709796452467316, 'ci95': [-0.002795695431041091, 0.00047531374455853553], 'better': 4, 'n': 26}, 'Dposterior-D_old': {'mean_per_video': -5.029843632614319e-05, 'ci95': [-0.0011901671994917406, 0.0013391760341862442], 'better': 2, 'n': 26}, 'Dposterior-D_constant': {'mean_per_video': 0.0001801432866546955, 'ci95': [-0.0007707338254450718, 0.0014681495566807438], 'better': 1, 'n': 26}, 'Dposterior-C_posterior': {'mean_per_video': 4.592099992061737e-05, 'ci95': [-0.001096454942241036, 0.0014649036219078623], 'better': 3, 'n': 26}, 'Dposterior-R1_posterior': {'mean_per_video': 0.0016375777932264184, 'ci95': [-0.0007282445277154001, 0.0039783374512443615], 'better': 11, 'n': 26}, 'Dposterior-geometry': {'mean_per_video': -0.0008581008148241064, 'ci95': [-0.0026558249715814605, 0.000906103022577801], 'better': 3, 'n': 26}}

## Интерпретация
Измерены первые назначения10парных условий при прежних средних откликах/весах/времени. Постоянный штраф отделяет общий сдвиг вероятностей кандидатов и стоимости сопоставления от пространственной неопределённости. Это не ужесточение отказа от вмешательства: в прежнем cost_row уменьшение вероятности кандидата повышает его стоимость, а полный NULL не возвращает исходную строку. Это не калиброванная вероятность полезности, не общая причина ошибок FlyVis и не HOTA.

## Решение
Pass -> separate bounded runtime parity/price and frozen-policy matched HOTA. Fail -> preserve this conditional-covariance confidence implementation; do not retune1.96, lambda, alpha or thresholds on26. Select a genuinely different target/state mechanism with measured prerequisite, not another scalar penalty. Primary: False

## Нельзя заявлять
The model can be confidently wrong about identity; spatially correlated721columns violate iidnoise assumptions and underestimate uncertainty. A simple global penalty may explain any benefit, so the matched constant branch is mandatory. This changes an internal confidence readout, not measured feature quality or biological dynamics. Repeatedtrain26 exploratory; noHOTAclaim.

## Прежние версии (до исправлений)
- **Интерпретация**, 2026-10-25 (a30b200f0db5c6e13110f939f14f48c90f3c70bc) — Термин «ужесточение отказа» неточен: проверена формула cost_row и контрпример полного NULL, прежняя версия сохранена.
  > Измерены первые назначения10парных условий при прежних средних откликах/весах/времени. Постоянный штраф отделяет общее ужесточение отказа от пространственной неопределённости. Это не калиброванная вероятность полезности и не HOTA.

## Критерии приёмки (записаны до прогона)
- ✅ All8326old feature-derived centers<0.5px/scores<0.001 for eachC/D/R1; all stored means/stds finite with exact label bindings and original8326rows. Posterior mean equals oldridge in direct-matrix unit checks. All26raw tracker outputs and everyold arm/geometry action exactly equal prior mean-region prototype. First2Dgroups conservative price fits original1800remaining; each phase independently bounded; completed chunks not repeated.  
  *runs/z13_ridge_uncertainty_20261025/independent_audit.json*
- ❌ Means Dposterior-baseline,Dposterior-Dold,Dposterior-Dconstant each>=2pp. All6paired-video95%lower>0 includingCposterior,R1posterior,geometry. All conditions required; no favorable arm/subset selection. Mean-score/center parity is technical only, not a positive result.  
  *runs/z13_ridge_uncertainty_20261025/replay/report.json*

## Доказательства
- Ранжирование12вариантов, формула, равные контроли и критерии до расчёта: `runs/z13_ridge_uncertainty_20261025/protocol.json` · 
- NULL прежнего контракта не означает отсутствие вмешательства: `runs/z13_null_mixture_20261025/cost_contract_check.json` · 

