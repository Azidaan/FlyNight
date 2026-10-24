---
aliases:
- 'З13: трекеры и FlyVis в единых условиях на7val'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P8
- метка/benchmark
- муха
исход: положительный
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: трекеры и FlyVis в единых условиях на7val

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_matched_trackers_v1`

Общее сравнение выполнено; прежний смешанный рейтинг заменён измерениями на одинаковых входах. Положительное состояние этой карточки означает полноту и воспроизведение сравнения,а не автоматический успех FlyVis. Добавка каждой D определяется только её парным интервалом против того же XYWH.

## Вопрос
Каковы HOTA внешних трекеров,стенда и двух D при одинаковых видео,детекциях,частоте,оценке и сшивке?

## Гипотеза
Часть прежнего видимого преимущества могла относиться к различию протоколов. Знак результата новой общей проверки не предполагается.

## Ожидалось (до прогона)
All7existing val videos, same original frames and960/conf.01 YOLO cache, same annotation alignment and official TrackEval. Primary total-system comparison includes native ByteTrack,BoT-SORT,TrackTrack,XYWHByteTrack,XYWH+legacyD,XYWH+centerD. Show without and with identical frozen base+MASA stitcher (gap150,dist5,threshold.6). No values from old200val or35train-derived dev in this table.

## Что сделано
2026-10-24

## Получено
runs/z13_matched_trackers_20261024/report.json → pooled/per_video/paired. Полная таблица и проценты: runs/z13_matched_trackers_20261024/verified_summary.json → summary: {"bytetrack": {"raw_hota": 52.43696923664447, "same_masa_hota": 53.16783939689458, "pooled_delta_from_xywh_masa": -2.1221674409508537, "relative_percent_from_xywh_masa": -3.8382477455189106, "paired": {"mean_per_video": -2.121065164074618, "ci95": [-3.0408134429111917, -1.2418824582256531], "better": 0, "n": 7}}, "botsort": {"raw_hota": 54.638691526284546, "same_masa_hota": 55.15176073569407, "pooled_delta_from_xywh_masa": -0.13824610215136346, "relative_percent_from_xywh_masa": -0.2500381353845944, "paired": {"mean_per_video": 0.016447217095176825, "ci95": [-0.8370826637069635, 0.8109597448916526], "better": 4, "n": 7}}, "tracktrack": {"raw_hota": 54.71316055250933, "same_masa_hota": 55.38261685106664, "pooled_delta_from_xywh_masa": 0.09261001322121132, "relative_percent_from_xywh_masa": 0.16749864671353365, "paired": {"mean_per_video": 0.11647289914977474, "ci95": [-1.6436352762107842, 1.676056640129819], "better": 4, "n": 7}}, "xywh": {"raw_hota": 54.57308976106422, "same_masa_hota": 55.29000683784543, "pooled_delta_from_xywh_masa": 0.0, "relative_percent_from_xywh_masa": 0.0, "paired": null}, "legacyD": {"raw_hota": 54.55835588474417, "same_masa_hota": 55.36852718697456, "pooled_delta_from_xywh_masa": 0.07852034912912842, "relative_percent_from_xywh_masa": 0.14201544477904107, "paired": {"mean_per_video": 0.07045446599556929, "ci95": [-0.06488939933770452, 0.21856379785260827], "better": 3, "n": 7}}, "centerD": {"raw_hota": 54.53778289643504, "same_masa_hota": 55.32379954926378, "pooled_delta_from_xywh_masa": 0.03379271141835005, "relative_percent_from_xywh_masa": 0.06111902195537312, "paired": {"mean_per_video": 0.018274988900313156, "ci95": [0.0, 0.050673115053792274], "better": 2, "n": 7}}}.49GPU-заданий,28исходныхнативныхпрогонов,обеобщиепроцедуры на7видео завершены;сырые выходы и общий MASA-базис воспроизведены.

## Интерпретация
Общее сравнение выполнено; прежний смешанный рейтинг заменён измерениями на одинаковых входах. Положительное состояние этой карточки означает полноту и воспроизведение сравнения,а не автоматический успех FlyVis. Добавка каждой D определяется только её парным интервалом против того же XYWH.

## Решение
Использовать только эту общую таблицу с явным n=7 и ограничениями. Не выбирать параметры по этим результатам и не открывать75отложенных; расширение набора оформлять отдельно. Эта оценка не меняет веса и не принимает новую архитектуру FlyVis.

## Нельзя заявлять
Only7previously used val videos; not independent final test and no75heldout. Detector validation set used in its training monitoring, but these videos are absent from gradient-training list. Settings not optimized per tracker; cannot infer universal ranking or compare directly with published benchmarks. The old MASA head was trained on XYWH life pairs and can transfer differently to other trackers; report raw/MASA separately.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (3f903b040856042f18d65e571c205cd3434191ba) — Полное проверенное общее сравнение.
  > Общие условия и все сравнения зафиксированы до нового результата.

## Критерии приёмки (записаны до прогона)
- ✅ All7videos and original cached inputs/metadata unchanged; no detector-training video overlap.  
  *runs/z13_matched_trackers_20261024/report.json; verified_summary.json; controller_result.json; gpu_stage/controller_result.json*
- ✅ No-op XYWH output reproduced, every frame consumed and every annotation evaluated; all6systems both raw/MASA complete.  
  *runs/z13_matched_trackers_20261024/report.json; verified_summary.json; controller_result.json; gpu_stage/controller_result.json*
- ✅ For any efficacy claim: paired-video95%CI lower>0 for that exact matched contrast; report all outcomes including unfavorable ones.  
  *runs/z13_matched_trackers_20261024/report.json; verified_summary.json; controller_result.json; gpu_stage/controller_result.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 53.16783939689458 | 55.29000683784543 | 7 common val bytetrack sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.bytetrack_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 55.15176073569407 | 55.29000683784543 | 7 common val botsort sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.botsort_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 55.38261685106664 | 55.29000683784543 | 7 common val tracktrack sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.tracktrack_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 55.29000683784543 | 55.29000683784543 | 7 common val xywh sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.xywh_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 55.36852718697456 | 55.29000683784543 | 7 common val legacyD sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.legacyD_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 55.32379954926378 | 55.29000683784543 | 7 common val centerD sameMASA | `runs/z13_matched_trackers_20261024/report.json` · pooled.centerD_masa.HOTA |

## Доказательства
- Общие условия и порядок до измерения: `runs/z13_matched_trackers_20261024/protocol.json` · 
- Предыдущая таблица не была единым рейтингом: `reports/stand_hota_comparability_20261024.json` · 
- Полная проверка одинаковых условий и49заданий: `runs/z13_matched_trackers_20261024/verified_summary.json` · 
- Одна общая таблица и ограничения: `runs/z13_matched_trackers_20261024/results.md` · 
- Нативные трекеры без сшивки: `runs/z13_matched_trackers_20261024/raw_report.json` · 

## Код
- `scripts/z13_matched_trackers_20261024.py`

