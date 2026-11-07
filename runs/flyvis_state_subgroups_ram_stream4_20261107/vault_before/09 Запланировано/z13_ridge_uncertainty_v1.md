---
aliases:
- Уверенность считывателя по опоре признаков в наблюдённом образце
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-25'
---

# 🔵 Уверенность считывателя по опоре признаков в наблюдённом образце

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_ridge_uncertainty_v1`

## Вопрос
Улучшит ли штраф за неопределённость модели цели полезные назначения сверх прежней D и одинакового штрафа всем областям?

## Гипотеза
Рабочая гипотеза: часть высоких неверных ответов может быть экстраполяцией вне признаков образца. Это не установленная причина; уверенная ошибка идентичности может остаться.

## Ожидалось (до прогона)
Change only region readout confidence: subtract1.96times the template-only conditional posterior standard deviation from the old standardized region response. Mean target model and original unit-clock core/history unchanged. Per-query constant-penalty control uses the same full-field mean standard deviation. Old/variable/constant forC/D/R1 plusunchanged geometry,10arms; same fixedalpha0.25 and null0.

## Нельзя заявлять
The model can be confidently wrong about identity; spatially correlated721columns violate iidnoise assumptions and underestimate uncertainty. A simple global penalty may explain any benefit, so the matched constant branch is mandatory. This changes an internal confidence readout, not measured feature quality or biological dynamics. Repeatedtrain26 exploratory; noHOTAclaim.

## Критерии приёмки (записаны до прогона)
- ⏳ All8326old feature-derived centers<0.5px/scores<0.001 for eachC/D/R1; all stored means/stds finite with exact label bindings and original8326rows. Posterior mean equals oldridge in direct-matrix unit checks. All26raw tracker outputs and everyold arm/geometry action exactly equal prior mean-region prototype. First2Dgroups conservative price fits original1800remaining; each phase independently bounded; completed chunks not repeated.
- ⏳ Means Dposterior-baseline,Dposterior-Dold,Dposterior-Dconstant each>=2pp. All6paired-video95%lower>0 includingCposterior,R1posterior,geometry. All conditions required; no favorable arm/subset selection. Mean-score/center parity is technical only, not a positive result.

## Доказательства
- Ранжирование12вариантов, формула, равные контроли и критерии до расчёта: `runs/z13_ridge_uncertainty_20261025/protocol.json` · 

