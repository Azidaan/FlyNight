---
aliases:
- 'Пространственное считывание: среднее против максимума внутри той же рамки'
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

# 🟢 Пространственное считывание: среднее против максимума внутри той же рамки

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_region_pooling_v1`

Среднее против максимума: критерий не пройден.

## Вопрос
Поможет ли сохранение локального максимума внутри реальной области при неизменном назначении и коэффициенте применения?

## Гипотеза
Рабочая гипотеза: усреднение может скрывать локальный отклик цели; максимум может его сохранить, но также усилить помеху. Причина не установлена.

## Ожидалось (до прогона)
Replace spatial averaging of the same standardized target-response field inside each unchanged detection region by its maximum. All8conditions D/C/R1/geometry timesmean/max replay simultaneously on the same46original videos and queries.

## Что сделано
2026-10-25

## Получено
runs/z13_region_pooling_20261025/report.json → paired: {'Dmax-baseline': {'mean_per_video': -0.0009622387967827609, 'ci95': [-0.0032271843582056575, 0.0011750875773459339], 'better': 6, 'n': 26}, 'Dmax-Dmean': {'mean_per_video': 5.844241213782783e-05, 'ci95': [-0.0011577923134542707, 0.0011534849072602532], 'better': 5, 'n': 26}, 'Dmax-Cmax': {'mean_per_video': -0.0003523104417584577, 'ci95': [-0.001384983816167442, 0.000580511842834349], 'better': 3, 'n': 26}, 'Dmax-R1max': {'mean_per_video': 0.0009325836667931204, 'ci95': [-0.000362844702467344, 0.002221142205294382], 'better': 5, 'n': 26}, 'Dmax-geometrymax': {'mean_per_video': -0.0007207840546707647, 'ci95': [-0.0017939081678738963, 0.00026771143458792935], 'better': 2, 'n': 26}};8парных условий, прежнийalpha0,25,46точныхповторов.

## Интерпретация
Изменено только пространственное агрегирование той же карты внутри тех же областей. Это сравнение конкретных считывателей; знак не доказывает причину всех отказов и не является HOTA.

## Решение
Pass -> separately budgeted frozen-policy closed-loop parity/price/HOTA with matched controls. Fail -> retain this max-pooling implementation and stop this fixed pooling branch. No extra scalar threshold/loss trials. Select another internal spatial/state node only from measured evidence. Основной критерий: False

## Нельзя заявлять
Max can amplify a distractor/noise peak and favor large/overlapping boxes. A positive comparison of this scalar readout does not prove that all mean-pooled learned representations failed because of averaging. C/R1/geometry controls and exact mean replay are mandatory; repeatedtrain26 and one-step labels are exploratory, not HOTA.

## Критерии приёмки (записаны до прогона)
- ✅ All46baseline outputs float32-exact; all prior peak centers<0.5px/scores<0.001. All mean outcomes must exactly reproduce the previous mean prototype atalpha0.25 on every saved query. Source/rows hashes and paired-video intervals independently checked. First2fit replays conservative2*mean_seconds*44+120 fit remaining1200s or preserve partial and stop.  
  *runs/z13_region_pooling_20261025/independent_audit.json*
- ❌ Dmax-baseline and Dmax-Dmean means both>=2pp with positive lower95%CI; Dmax-Cmax,R1max,geometrymax all lower95%CI>0. All conditions; no selection of a favorable arm or subset.  
  *runs/z13_region_pooling_20261025/report.json*

## Доказательства
- Парные8условий, прежнийalpha, критерии и бюджет до расчёта: `runs/z13_region_pooling_20261025/protocol.json` · 

