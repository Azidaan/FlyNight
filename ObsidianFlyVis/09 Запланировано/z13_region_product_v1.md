---
aliases:
- Прямое поканальное сравнение модели цели и активности FlyVis
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/fly-problem-P8
исход: готов к запуску
направление: FlyVis
начат: '2026-10-25'
---

# ⚪ Прямое поканальное сравнение модели цели и активности FlyVis

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `z13_region_product_v1`

## Вопрос
Поможет ли сохранение исходного поканального взаимодействия перед независимым сжатием в считывателе?

## Гипотеза
Рабочая гипотеза: прямое взаимодействие68 исходных каналов полезнее, чем только произведение независимо обученных16мерных вложений. Это не установленная причина отказов.

## Ожидалось (до прогона)
Add direct68 channelwise products between original normalized pooled query activity and its unchanged observed-only ridge coefficients, before the two independent68->16 embeddings. Same prior6 assignment inputs remain for all arms. Equal blank controls add exactly68 zeros to the same162->32->1 network.

## Нельзя заявлять
Mean pooling may already discard the needed spatial information; adding products cannot restore it. A wider network can overfit rare assignment events, so equal-capacity blank and C/R1/geometry contrasts remain mandatory. Repeatedtrain26 and one-row labels are exploratory, not HOTA or causal proof.

## Критерии приёмки (записаны до прогона)
- ⏳ Reuse every frozen original activity/context/label cache; fit84-only RMS. No extraction or completed-model repetition. First full model2*elapsed*23+60 must fit original1800seconds. Independently reproduce every8326-row evaluation of all24models and paired intervals before catalog acceptance.
- ⏳ Dproduct-baseline and Dproduct-Dblank means both>=2pp; all6paired-video95%CI lower>0 against baseline,blank,Cproduct,R1product,geometryproduct,previousDcontext.24models,3seeds.

## Доказательства
- Неизменные входы, парные контроли, критерии и бюджет до расчёта: `runs/z13_region_product_20261025/protocol.json` · 

