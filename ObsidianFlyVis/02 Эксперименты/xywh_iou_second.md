---
aliases:
- 'Второй проход чистым IoU: возвращает FN, добавляет FP'
tags:
- эксперимент
- исход/completed_negative
- направление/tracker10
- метка/second-association
- метка/weak-detections
- метка/ablation
исход: отрицательный
направление: Tracker 1,0 и слабые детекции
начат: '2026-09-13'
завершён: '2026-09-13'
---

# 🔴 Второй проход чистым IoU: возвращает FN, добавляет FP

↑ [[02 Эксперименты/Направления/Tracker 1,0 и слабые детекции|Tracker 1,0 и слабые детекции]] · **Исход:** отрицательный · `xywh_iou_second`

XYWHIoUSecondByteTrack: первая ассоциация без изменений, на второй — чистое IoU вместо IoU × confidence. Отрицательная диагностическая абляция.

## Вопрос
Достаточно ли убрать повторный штраф confidence на второй ассоциации, чтобы безопасно использовать слабые детекции?

## Гипотеза
Слабые верные детекции (ночью 22.8% объектов лежат ниже порога) вернутся в треки без роста ложных.

## Ожидалось (до прогона)
Рост HOTA/AssA/IDF1 и снижение FN и IDSw без роста FP; синтетический тест сохраняет ID при confidence 0.2.

## Что сделано
Реализация, синтетические тесты (сохранение ID, неизменность первого прохода), семивидео прогон.

## Получено
FN −406, фрагментаций −47, но FP +456, IDSw +19; HOTA 54.5731 → 54.4247, AssA 54.5718 → 53.9853, IDF1 60.3800 → 59.7928.

## Интерпретация
Слабые детекции содержат полезный сигнал, но без селективной оценки их допуск небезопасен — нужна модель q_match, а рождение слабого трека должен отдельно контролировать q_birth.

## Решение
Не принимается как улучшение сильного контроля; сохраняется как абляция.

## Условия
- **baseline:** XYWHByteTrack
- **variant:** XYWHIoUSecondByteTrack
- **dataset:** BDD100K, исходные видео
- **split:** 7 видео val ~30 Гц
- **detector:** YOLO26s seed42, 960
- **tracker:** XYWHIoUSecondByteTrack

## Критерии приёмки (записаны до прогона)
- ✅ синтетический тест: ID сохраняется при идеальном IoU и confidence 0.2  
  *tests/test_xywh_bytetrack.py*
- ✅ первый проход численно совпадает с XYWHByteTrack  
  *tests/test_xywh_bytetrack.py*
- ❌ HOTA, AssA, IDF1 не ниже XYWHByteTrack; FP не растёт  
  *xywh_iou_second_validation7.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 54.424665774463 | 54.57308976106422 | 7 видео / XYWHByteTrack | `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.HOTA |
| [AssA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAssA) | 53.98526121155564 | 54.571813137230265 | 7 видео / XYWHByteTrack | `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.AssA |
| [FN↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FFN) | 4970 | 5376 | 7 видео / XYWHByteTrack | `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.CLR_FN |
| [FP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FFP) | 1738 | 1282 | 7 видео / XYWHByteTrack | `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.CLR_FP |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 545 | 526 | 7 видео / XYWHByteTrack | `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.IDSw |

## Доказательства
- метрики абляции: `runs/video_rate/xywh_iou_second_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics
- исправленная вторая ассоциация: `FLYVIS_STRESS_AND_SECOND_ASSOCIATION_REPORT.md` · §7
- итог по второй ассоциации: [§0↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F0.%20%D0%A2%D0%B5%D0%BA%D1%83%D1%89%D0%B5%D0%B5%20%D1%81%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%2013.09.2026)

## Код
- `tracking/xywh_bytetrack.py`
- `tests/test_xywh_bytetrack.py`

## Данные
- `runs/video_rate/xywh_iou_second_validation7`

## Коммиты
- `c5c0e20`

