---
aliases:
- Контекст исходного назначения внутри считывателя FlyVis
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/fly-problem-P8
исход: готов к запуску
направление: FlyVis
начат: '2026-10-25'
---

# ⚪ Контекст исходного назначения внутри считывателя FlyVis

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `z13_region_context_v1`

## Вопрос
Улучшат ли исходные назначения и конкуренция других треков решение считывателя о вмешательстве?

## Гипотеза
Рабочая гипотеза: часть различий полезного/нейтрального/вредного действия определяется отсутствующим в прежних входах контекстом назначения. Проверяется добавлением наблюдаемых6признаков против их нулевого контроля той же ёмкости.

## Ожидалось (до прогона)
Улучшить D минимум на2п.п. относительно равного пустого контроля и без правки; подтвердить преимущество над C/R1/геометрией.

## Нельзя заявлять
Assignment context may help geometric processing alone, without FlyVis. This is a small readout adaptation, not a graph neural tracker. One-row outcomes omit joint corrections and closed-loop feedback; repeatedtrain26 is exploratory.

## Критерии приёмки (записаны до прогона)
- ⏳ Cache130 baseline replays float32-exact, all37296query UIDs/causal IDs/predictions/labels and region order equal saved labels; first2videos2*mean_seconds*128+60 fits1200remaining. No completed cache video repeated. Training first complete model2*seconds*23+60 fits remaining1800s.
- ⏳ Dcontext-baseline and Dcontext-Dblank both means>=2pp, and all6paired-video lower95% bounds>0: baseline,Dblank,Ccontext,R1context,geometrycontext,previous signedD.

## Доказательства
- Критерии, равная ёмкость, наблюдаемые входы и бюджеты до расчёта: `runs/z13_region_context_20261025/protocol.json` · 

