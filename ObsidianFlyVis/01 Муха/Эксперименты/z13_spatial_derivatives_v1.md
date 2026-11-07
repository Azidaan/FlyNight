---
aliases:
- 'Считывание FlyVis: пространственные производные вместо потери расположения соседей'
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

# 🟢 Считывание FlyVis: пространственные производные вместо потери расположения соседей

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_spatial_derivatives_v1`

Локальные пространственные производные: критерий не пройден.

## Вопрос
Помогут ли два направления изменения активности по соседним колонкам при том же состоянии и модели цели?

## Гипотеза
Рабочая гипотеза: среднее соседей скрывает локальное расположение признаков. Потеря направления подтверждается формулой и синтетическим примером; причинность реальных ошибок и польза новых признаков не установлены.

## Ожидалось (до прогона)
Change only the local spatial descriptor entering the target readout: retain original68standardized features (34activity+34neighbor means), append34horizontal+34vertical first-ring least-squares slopes. Fit identical observed-center ridge with unchanged savedlambda. Frozen C/D/R1 core/full unit-clock/history/input/norms/target/regionmean/null0/alpha0.25 unchanged. Seven paired conditions old/gradient forC/D/R1 plusgeometry;136dimensional zero-extension control reproduces old68ridge.

## Что сделано
2026-10-25

## Получено
runs/z13_spatial_derivatives_20261025/replay/report.json → paired: {'Dgradient-baseline': {'mean_per_video': -0.000894577804128659, 'ci95': [-0.0028583229024525354, 0.000872017656817422], 'better': 5, 'n': 26}, 'Dgradient-Dold': {'mean_per_video': 0.00012610340479192938, 'ci95': [0.0, 0.0003783102143757881], 'better': 1, 'n': 26}, 'Dgradient-C_gradient': {'mean_per_video': 0.0004760724392513178, 'ci95': [0.0, 0.0011457250026291497], 'better': 3, 'n': 26}, 'Dgradient-R1_gradient': {'mean_per_video': -0.0006994642740037581, 'ci95': [-0.002603229805675989, 0.0009190278331309512], 'better': 7, 'n': 26}, 'Dgradient-geometry': {'mean_per_video': -0.0006816989737060338, 'ci95': [-0.0018629765308342738, 0.00024058141451965443], 'better': 2, 'n': 26}, 'gradient_effect_D-R1': {'mean_per_video': -0.0005380174863547949, 'ci95': [-0.001994557302877814, 0.0006700098435470139], 'better': 2, 'n': 26}, 'gradient_effect_D-C': {'mean_per_video': 0.0005174974205386973, 'ci95': [0.0, 0.0012039876544427243], 'better': 3, 'n': 26}}

## Интерпретация
Парное вмешательство меняет только локальные входы модели цели: прежние68каналов дополнены двумя пространственными производными. Состояние/образец/нормы/регуляризация/политика прежние. Это первые назначения на многократно использованном train26, не HOTA; структурная потеря направления не доказывает причину реальных ошибок.

## Решение
Pass -> separate frozen-runtime parity/price and matched closed-loop gate before HOTA. Fail -> preserve this exact two-direction first-ring descriptor; no derivative scale/radius/lambda sweep on26 and no core training rescue. Distinguish representation change from proven natural error cause; choose another node only with measured prerequisite. Primary: False

## Нельзя заявлять
FlyVis channels already include directional neurons, so spatial slopes may be redundant. Adding features can overfit the721observed columns; unchanged lambda/norms may not suit136dimensions. Equal136zero extension isolates mere empty dimensions, not the additional useful degrees of freedom; equal C/R1 and D-specific interaction controls are essential. A synthetic local alias is not evidence that it caused actual errors. No HOTA or global cause from one-step train26.

## Критерии приёмки (записаны до прогона)
- ✅ Three tests: affine-field slopes including boundaries; orientation alias broken at center while old values retained; neighbor-order invariance and direct-matrix zero-padding ridge equivalence. All721offset matrices rank2. All8326old C/D/R1 center<0.5px/score<0.001, original raw feature arrays shared; all136zero-control maps reproduce68within1e-8absolute+1e-7relative. All new maps finite, exact source/label/crop binding; all26old action/baseline outputs exact. First2Dgroups conservative price fits original1800s or preservepartial/stop; completed chunks never repeated.  
  *runs/z13_spatial_derivatives_20261025/independent_audit.json*
- ❌ Dgradient-baseline and Dgradient-Dold means>=2pp; all7paired-video95%lower>0, including Cgradient,R1gradient,geometry and the intervention effects D-R1 and D-C. No favorable arm/subset/threshold/lambda selection. All mandatory; absolute improvement and FlyVis-specific addition both required.  
  *runs/z13_spatial_derivatives_20261025/replay/report.json*

## Доказательства
- Новый локальный узел, ранжирование12, равные контроли и критерии до GPU: `runs/z13_spatial_derivatives_20261025/protocol.json` · 

