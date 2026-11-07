---
aliases:
- Равные признаки прежних областей D/C/R1 для временного считывания
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-25'
---

# 🔵 Равные признаки прежних областей D/C/R1 для временного считывания

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_temporal_region_features_v1`

Подготовка временных признаков; обучение ещё не запускалось.

## Вопрос
Можно ли получить равные причинные признаки прежних областей без изменения нынешнего ответа FlyVis?

## Гипотеза
Проверенная доступность прошлых наблюдений допускает отдельное извлечение признаков; способность различать личность пока не проверена.

## Ожидалось (до прогона)
Prepare previous-region68activity and observable geometry for one future learned cross-frame FlyVis readout. Existing current features/target/action labels stay immutable and are referenced byUID, not replaced. D/C/R1 have identical selected observations/regions.

## Нельзя заявлять
Earlier same-class regions may be distractors or too stale, and FlyVis may not encode enough identity. Availability/oracle passed, but learned benefit remains unknown. Spatial-set module stays deferred; this is a distinct temporal input, not resuming its75models.

## Критерии приёмки (записаны до прогона)
- ⏳ For every eligible current query, old primary peak center<.5px and score<.001. Same weights, input sequence, normalization and target coefficients. No tolerance weakening. Stored previous arrays finite and save/reload exact; all22090UIDs and previous region indices same D/C/R1. Replay from saved data before any future learner.
- ⏳ OneGPUmutex;16episode inference packing,not learning batch. First2Dgroups:2*measured seconds per weighted-frame*remainingD+Cweightedframes+600 must fit remaining7200seconds. If not, preserve and stop. R1 remains deterministic retinal feature computation onCPU, not CPU learning. Independent controller hiddenIdle; no coretraining/dev/HOTA/heldout75/download.

## Доказательства
- Правила причинных входов, воспроизведение и7200сбюджет доGPUизвлечения: `runs/z13_temporal_region_features_20261025/protocol.json` · 

