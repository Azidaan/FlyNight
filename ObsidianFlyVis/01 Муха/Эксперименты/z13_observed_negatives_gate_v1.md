---
aliases:
- 'Модель конкретной цели: доступность наблюдённых отрицательных соседей'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Модель конкретной цели: доступность наблюдённых отрицательных соседей

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_observed_negatives_gate_v1`

Наблюдённые отрицательные соседи: inconclusive

## Вопрос
Хватает ли причинно доступных отрицательных соседей образца для отдельной проверки их веса в модели цели?

## Гипотеза
Рабочая гипотеза: одинаковые объекты рядом с целью могут требовать отдельной подгонки к конкретному треку. Обычные отрицательные колонки уже есть; их отсутствие не заявляется.

## Ожидалось (до прогона)
Candidate target-specific ridge update: weight4on separable same-class detector-negative columns in the last OBSERVED exemplar, weight1elsewhere; same Gaussian target/core/readout and matched unweightedC/D/R1 controls. This queue permits only the cheap availability prerequisite, not extraction or fitting.

## Что сделано
2026-10-25

## Получено
runs/z13_observed_negatives_gate_20261025/report.json → observed_negative_oracle_exposure: {'mean_per_video': 0.02533457497664401, 'ci95': [0.01838207128923364, 0.032686372316585104], 'better': 23, 'n': 26};37296запросов/26600известных, только прошлый наблюдённый кадр.

## Интерпретация
Измерена доступность причинных отрицательных регионов и оракульная достижимость. Не доказаны различимость объектов, правильность весового обновления или польза стенду.

## Решение
Admission -> separate preregistered bounded weighted/unweighted target-model prototype; no automatic core training orHOTA. Insufficient/inconclusive -> retain this exact definition and do not change IoU/expansion/2pp based on the result. Gate: inconclusive

## Нельзя заявлять
A nearby detector box may be a duplicate or a wrong identity, while conservative overlap exclusion can discard the hard overlapping distractors. Observable negatives may be common but uninformative; weighting can suppress useful target features. A full prototype must use identical causal masks forC/D/R1, oldmean parity, fixed before-evaluation settings and actual assignment controls; no presumed cause or HOTA.

## Критерии приёмки (записаны до прогона)
- ❌ Exact130video/37296query/26600known bindings, nonempty observed mask, every negative uses only last-observed frame/crop, same tracker class, conf>=0.25, IoU<=0.1 with the lattice-reconstructed observed box, >=1column outside its1.25expanded guard, supported center by the unchanged lattice rule. Admission requires lower paired-video95%CI of known evaluate26 queries with such an observed negative AND a reachable beneficial region action>0.02; upper<0.02 insufficient, otherwise inconclusive. This is only causal-input availability and oracle exposure, not an achieved benefit. No relaxing this criterion after result.  
  *runs/z13_observed_negatives_gate_20261025/report.json*

## Доказательства
- Отложенная мишень и критерий из зарегистрированной заранее очереди: `runs/z13_observed_negatives_gate_20261025/protocol.json` · 

