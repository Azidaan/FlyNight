---
aliases:
- 'З13: направленный пространственный контекст считывателя'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/Z13
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-18'
---

# 🔵 З13: направленный пространственный контекст считывателя

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_directional_context_v1`

## Вопрос
Помогает ли сохранение расположения соседних сигналов найти цель и выбрать полезные поправки сверх геометрии?

## Гипотеза
Среднее соседей удаляет направления (свойство реализации); направленные контрасты могут улучшить различение цели и фона (гипотеза).

## Ожидалось (до прогона)
D directional minus D mean: lower95%video CI>0 and mean>=0.02. Also D directional minus D shuffled, C mean and R1 directional each lower95%CI>0. At1%: D directional map utility minus D mean map utility, D directional geometry, D shuffled map utility and R1 directional map utility all lower95%video CIs>0. Both localization and selection gates required.2/5% and C analyses secondary, not rescue.

## Критерии приёмки (записаны до прогона)
- ⏳ For every mean row: same key/kind/useful, center<0.5px,score<0.001; mean-map feature max difference<1e-10; mean utility OOF scores match saved scores<1e-10.
- ⏳ D directional minus D mean: lower95%video CI>0 and mean>=0.02. Also D directional minus D shuffled, C mean and R1 directional each lower95%CI>0.
- ⏳ At1%: D directional map utility minus D mean map utility, D directional geometry, D shuffled map utility and R1 directional map utility all lower95%video CIs>0. Both localization and selection gates required.2/5% and C analyses secondary, not rescue.

## Доказательства
- Условия и критерии до расчёта: `runs/z13_directional_context_20261018/protocol.json` · 

## Код
- `scripts/z13_directional_context.py`
- `tracking/z13_directional_context.py`
- `tests/test_z13_directional_context.py`
- `scripts/z13_context_progress.ps1`

## Данные
- `runs/z13_directional_context_20261018`

