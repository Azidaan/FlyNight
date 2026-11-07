---
aliases:
- 'З13: отдельный перенос пространственного считывания C и контроля R1'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/Z13
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-20'
---

# 🔵 З13: отдельный перенос пространственного считывания C и контроля R1

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_context_transfer_v1`

## Вопрос
Переносится ли вспомогательный результат пространственного считывания C на другие видео?

## Гипотеза
Фиксированное пространственное считывание C может улучшить центры и выбор поправок; перенос не установлен.

## Ожидалось (до прогона)
C directional minus C mean: paired video95%CI lower>0 and mean>=0.02; C directional minus R1 directional lower>0. At1% of all rows: C directional map utility minus C mean map utility, C directional geometry, and R1 directional map utility each paired video95%CI lower>0. Both localization and selection required.

## Критерии приёмки (записаны до прогона)
- ⏳ Unchanged mean centers<0.5px, scores<0.001, same labels/kind/useful on every video. Technical C directional center<0.5px, score<0.001, map features<1e-10 against the saved same training video. Normalization exactly equals saved dev normalization.
- ⏳ C directional minus C mean: paired video95%CI lower>0 and mean>=0.02; C directional minus R1 directional lower>0.
- ⏳ At1% of all rows: C directional map utility minus C mean map utility, C directional geometry, and R1 directional map utility each paired video95%CI lower>0. Both localization and selection required.

## Доказательства
- Условия до переноса; полная проверка пока не разрешена бюджетом: `runs/z13_context_transfer_20261020/protocol.json` · 
- Точное воспроизведение направления и прежнего контроля; это техническая готовность: `runs/z13_context_transfer_20261020/technical.json` · 

## Код
- `scripts/z13_context_transfer_20261020.py`
- `scripts/z13_context_transfer_progress.ps1`
- `scripts/z13_record_context_transfer_20261020.py`
- `scripts/z13_context_transfer_service_20261020.py`

## Данные
- `runs/z13_context_transfer_20261020`

