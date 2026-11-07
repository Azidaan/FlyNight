---
aliases:
- Контекст исходного назначения внутри считывателя FlyVis
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

# 🟢 Контекст исходного назначения внутри считывателя FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_region_context_v1`

Наблюдаемый контекст назначения: критерий не пройден.

## Вопрос
Улучшат ли исходные назначения и конкуренция других треков решение считывателя о вмешательстве?

## Гипотеза
Рабочая гипотеза: часть различий полезного/нейтрального/вредного действия определяется отсутствующим в прежних входах контекстом назначения. Проверяется добавлением наблюдаемых6признаков против их нулевого контроля той же ёмкости.

## Ожидалось (до прогона)
Улучшить D минимум на2п.п. относительно равного пустого контроля и без правки; подтвердить преимущество над C/R1/геометрией.

## Что сделано
2026-10-25

## Получено
runs/z13_region_context_20261025/training/report.json → paired: {'Dcontext-baseline': {'mean_per_video': -0.0019292708621356988, 'ci95': [-0.005787639250481386, 0.0005324063981026563], 'better': 8, 'n': 26}, 'Dcontext-D_blank': {'mean_per_video': -0.0009474470603382699, 'ci95': [-0.003387031254748792, 0.0009350952585686619], 'better': 11, 'n': 26}, 'Dcontext-C_context': {'mean_per_video': -0.0003457244453321865, 'ci95': [-0.0025627777737961444, 0.00207630345317482], 'better': 7, 'n': 26}, 'Dcontext-R1_context': {'mean_per_video': -0.0019833658529395505, 'ci95': [-0.005808512379441493, 0.0004815036752694252], 'better': 8, 'n': 26}, 'Dcontext-geometry_context': {'mean_per_video': -0.0019292708621356988, 'ci95': [-0.005787639250481386, 0.0005324063981026563], 'better': 8, 'n': 26}, 'Dcontext-Dprevious': {'mean_per_video': -0.0025906306381335846, 'ci95': [-0.0064886488956715765, 0.000306103643212775], 'better': 7, 'n': 26}};24модели,3сида,8равных вариантов.

## Интерпретация
Контекст сопоставления проверен против его нулевого контроля той же ёмкости; это внутренний считыватель с внешними наблюдаемыми входами. Не HOTA и не полноценный графовый трекер.

## Решение
Pass -> separately budgeted frozen closed-loop parity/price/HOTA, native/C/R1/no-fly controls. Fail -> preserve this context implementation; no HOTA or core training by default, select a different measured internal node. Измеренный основной критерий: False

## Нельзя заявлять
Assignment context may help geometric processing alone, without FlyVis. This is a small readout adaptation, not a graph neural tracker. One-row outcomes omit joint corrections and closed-loop feedback; repeatedtrain26 is exploratory.

## Критерии приёмки (записаны до прогона)
- ✅ Cache130 baseline replays float32-exact, all37296query UIDs/causal IDs/predictions/labels and region order equal saved labels; first2videos2*mean_seconds*128+60 fits1200remaining. No completed cache video repeated. Training first complete model2*seconds*23+60 fits remaining1800s.  
  *runs/z13_region_context_20261025/independent_audit.json*
- ❌ Dcontext-baseline and Dcontext-Dblank both means>=2pp, and all6paired-video lower95% bounds>0: baseline,Dblank,Ccontext,R1context,geometrycontext,previous signedD.  
  *runs/z13_region_context_20261025/training/report.json*

## Доказательства
- Критерии, равная ёмкость, наблюдаемые входы и бюджеты до расчёта: `runs/z13_region_context_20261025/protocol.json` · 

