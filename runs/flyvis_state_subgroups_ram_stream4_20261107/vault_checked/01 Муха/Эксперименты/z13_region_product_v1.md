---
aliases:
- Прямое поканальное сравнение модели цели и активности FlyVis
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

# 🟢 Прямое поканальное сравнение модели цели и активности FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_region_product_v1`

Прямое поканальное сравнение: критерий не пройден.

## Вопрос
Поможет ли сохранение исходного поканального взаимодействия перед независимым сжатием в считывателе?

## Гипотеза
Рабочая гипотеза: прямое взаимодействие68 исходных каналов полезнее, чем только произведение независимо обученных16мерных вложений. Это не установленная причина отказов.

## Ожидалось (до прогона)
Add direct68 channelwise products between original normalized pooled query activity and its unchanged observed-only ridge coefficients, before the two independent68->16 embeddings. Same prior6 assignment inputs remain for all arms. Equal blank controls add exactly68 zeros to the same162->32->1 network.

## Что сделано
2026-10-25

## Получено
runs/z13_region_product_20261025/training/report.json → paired: {'Dproduct-baseline': {'mean_per_video': -0.0028809103966147835, 'ci95': [-0.009881007676796519, 0.001092909643319165], 'better': 9, 'n': 26}, 'Dproduct-D_blank': {'mean_per_video': 0.0014263217846815128, 'ci95': [-0.0005705092618417178, 0.003783513675510822], 'better': 12, 'n': 26}, 'Dproduct-C_product': {'mean_per_video': -0.002856428281865546, 'ci95': [-0.009808990545697026, 0.0010665401298839534], 'better': 9, 'n': 26}, 'Dproduct-R1_product': {'mean_per_video': -0.0028809103966147835, 'ci95': [-0.009881007676796519, 0.001092909643319165], 'better': 9, 'n': 26}, 'Dproduct-geometry_product': {'mean_per_video': -0.0028809103966147835, 'ci95': [-0.009881007676796519, 0.001092909643319165], 'better': 9, 'n': 26}, 'Dproduct-Dprevious': {'mean_per_video': -0.0009516395344790843, 'ci95': [-0.004758587075912108, 0.00147123359171454], 'better': 10, 'n': 26}};24модели,3сида,равный162входный нулевой контроль.

## Интерпретация
Изменён внутренний узел сопоставления активности с моделью цели. Все первичные выходы24моделей воспроизведены независимо; это последствия первых назначений, не HOTA и не доказанная причина всех отказов.

## Решение
Pass -> separately budgeted frozen closed-loop parity/price/HOTA with standardC and no-fly controls. Fail -> preserve this interaction implementation; do not enumerate further scalar losses or thresholds. Next action must examine the spatial pooling node using one preregistered cheap check before any new extraction. Основной критерий: False

## Нельзя заявлять
Mean pooling may already discard the needed spatial information; adding products cannot restore it. A wider network can overfit rare assignment events, so equal-capacity blank and C/R1/geometry contrasts remain mandatory. Repeatedtrain26 and one-row labels are exploratory, not HOTA or causal proof.

## Критерии приёмки (записаны до прогона)
- ✅ Reuse every frozen original activity/context/label cache; fit84-only RMS. No extraction or completed-model repetition. First full model2*elapsed*23+60 must fit original1800seconds. Independently reproduce every8326-row evaluation of all24models and paired intervals before catalog acceptance.  
  *runs/z13_region_product_20261025/independent_audit.json*
- ❌ Dproduct-baseline and Dproduct-Dblank means both>=2pp; all6paired-video95%CI lower>0 against baseline,blank,Cproduct,R1product,geometryproduct,previousDcontext.24models,3seeds.  
  *runs/z13_region_product_20261025/training/report.json*

## Доказательства
- Неизменные входы, парные контроли, критерии и бюджет до расчёта: `runs/z13_region_product_20261025/protocol.json` · 

