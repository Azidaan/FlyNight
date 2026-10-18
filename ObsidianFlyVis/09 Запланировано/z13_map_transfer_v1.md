---
aliases:
- 'З13: перенос оценки полезности C/D/R1 с train26 на dev35'
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

# 🔵 З13: перенос оценки полезности C/D/R1 с train26 на dev35

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_map_transfer_v1`

## Вопрос
Переносится ли преимущество оценки карты C сверх геометрии на другие видео?

## Гипотеза
Вспомогательный положительный результат C на train26 может переноситься; это гипотеза, выбор C сделан после просмотра train26.

## Ожидалось (до прогона)
At1%: C map minus original, geometry and limited each lower95%paired-video CI>0. Separate FlyVis gate: C map minus R1 map lower95%CI>0. Both required to continue. D and2/5% secondary, cannot rescue primary.

## Критерии приёмки (записаны до прогона)
- ⏳ Every dev row keeps video/label/kind/useful; unchanged center difference<0.5px and score difference<0.001; fit20 normalization identical to saved dev. Use saved centers/outcomes for comparisons.
- ⏳ At1%: C map minus original, geometry and limited each lower95%paired-video CI>0. Separate FlyVis gate: C map minus R1 map lower95%CI>0. Both required to continue. D and2/5% secondary, cannot rescue primary.

## Доказательства
- Критерии, ограничения и исходные хеши до переноса: `runs/z13_map_transfer_20261018/protocol.json` · 

## Код
- `scripts/z13_map_transfer.py`
- `scripts/z13_map_transfer_progress.ps1`
- `tracking/z13_utility_transfer.py`
- `tests/test_z13_utility_transfer.py`

## Данные
- `runs/z13_map_transfer_20261018`

