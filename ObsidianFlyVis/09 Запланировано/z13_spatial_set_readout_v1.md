---
aliases:
- Считывание FlyVis по активности отдельных колонок и их положениям
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-25'
---

# 🔵 Считывание FlyVis по активности отдельных колонок и их положениям

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_spatial_set_readout_v1`

## Вопрос
Даёт ли расположение активности внутри реальной области пользу, потерянную её средним?

## Гипотеза
Рабочая гипотеза: пары активность–положение различают похожие области при одинаковой средней активности. Структурная возможность проверена синтетически; причина реальных ошибок и польза не установлены.

## Ожидалось (до прогона)
Replace the internal regional embedding of FlyVis activity with a Deep-Sets-style mean of learned functions of every supported (68-channel activity,2relative coordinates) pair. Keep actual full-box/NULL actions and unchanged target descriptor. No quadrants, subsampling or occupancy gate. New outputs can change the chosen actual assignment action. D_spatial versus D_mean: latter repeats the68mean computed from the EXACT SAME new fullfloat32activity at each same supported coordinate before the SAME point network. Thus all positions/support/cardinality, train splits, initialization, parameters and loss identical. C_spatial stock fly; R1_spatial equal-dimensional rich retinal history; geometry has identical support/coordinates/context but zeros activity and target. Three seeds equally averaged, no best cell selection.

## Нельзя заявлять
Relative spatial evidence may be redundant/noisy; only16aggregate dimensions and1150steps may be insufficient. Target still the prior68descriptor, so failure does not test full spatial exemplar matching. Irregular support supplies geometry, hence geometry and repeated-mean controls mandatory. Point-set theory supplies an architecture, NOT a guarantee of universal representation with this finite embedding or tracking improvement.

## Критерии приёмки (записаны до прогона)
- ⏳ Unchanged130query/episode/frame/choices/masks binding and original normalization/weights/history. Every old peak center<0.5px,score<.001, old tolerances unchanged. Mean-control features are computed with the ORIGINAL pool_activity on the SAME new fullfloat32arrays, save/reload means exact. Against previous separately extracted mean arrays record maximum difference descriptively; separateGPUpasses are not assumed bit-exact. New models exactly reproduce saved validation predictions from weights.
- ⏳ Same five nested video folds onfit104 as previous trial; threshold selected only on disjoint calibration videos within each fold. Primary D_spatial mean-video(help-harm)/known>=.02 and ALL paired95%lower>0 versus zero,D_mean,C_spatial,R1_spatial,geometry. Seeds equally averaged. Conditional descriptive intervals on reused fit104; no information-absence claim on failure.
- ⏳ Only if internal admission passes:15new models fit84/cal20; once evaluate26, same criteria including frozen previous D_context from full-box followup. No automatic HOTA. Failed internal stops before26.

## Доказательства
- Отличия, равные контроли, критерии, источники и бюджет до новых признаков: `runs/z13_spatial_set_readout_20261025/protocol.json` · 
- Служебное восстановление CUDA после CPU-импорта; научный протокол и прежние сроки сохранены: `runs/z13_spatial_set_readout_20261025/recovery1/protocol.json` · 
- Упаковка16 только замороженного вывода, готовые46строк переиспользованы; прежние критерии и сроки: `runs/z13_spatial_set_readout_20261025/recovery2/protocol.json` · 

