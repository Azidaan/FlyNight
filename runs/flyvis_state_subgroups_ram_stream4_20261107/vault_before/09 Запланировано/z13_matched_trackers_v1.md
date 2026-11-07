---
aliases:
- 'З13: трекеры и FlyVis в единых условиях на7val'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
- метка/benchmark
исход: идёт
направление: FlyVis
начат: '2026-10-24'
---

# 🔵 З13: трекеры и FlyVis в единых условиях на7val

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_matched_trackers_v1`

Общие условия и все сравнения зафиксированы до нового результата.

## Вопрос
Каковы HOTA внешних трекеров,стенда и двух D при одинаковых видео,детекциях,частоте,оценке и сшивке?

## Гипотеза
Часть прежнего видимого преимущества могла относиться к различию протоколов. Знак результата новой общей проверки не предполагается.

## Ожидалось (до прогона)
All7existing val videos, same original frames and960/conf.01 YOLO cache, same annotation alignment and official TrackEval. Primary total-system comparison includes native ByteTrack,BoT-SORT,TrackTrack,XYWHByteTrack,XYWH+legacyD,XYWH+centerD. Show without and with identical frozen base+MASA stitcher (gap150,dist5,threshold.6). No values from old200val or35train-derived dev in this table.

## Нельзя заявлять
Only7previously used val videos; not independent final test and no75heldout. Detector validation set used in its training monitoring, but these videos are absent from gradient-training list. Settings not optimized per tracker; cannot infer universal ranking or compare directly with published benchmarks. The old MASA head was trained on XYWH life pairs and can transfer differently to other trackers; report raw/MASA separately.

## Критерии приёмки (записаны до прогона)
- ⏳ All7videos and original cached inputs/metadata unchanged; no detector-training video overlap.
- ⏳ No-op XYWH output reproduced, every frame consumed and every annotation evaluated; all6systems both raw/MASA complete.
- ⏳ For any efficacy claim: paired-video95%CI lower>0 for that exact matched contrast; report all outcomes including unfavorable ones.

## Доказательства
- Общие условия и порядок до измерения: `runs/z13_matched_trackers_20261024/protocol.json` · 
- Предыдущая таблица не была единым рейтингом: `reports/stand_hota_comparability_20261024.json` · 

## Код
- `scripts/z13_matched_trackers_20261024.py`

