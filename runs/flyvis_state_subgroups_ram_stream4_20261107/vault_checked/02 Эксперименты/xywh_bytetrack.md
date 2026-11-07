---
aliases:
- 'XYWHByteTrack: модель рамки с независимыми шириной и высотой'
tags:
- эксперимент
- исход/completed_positive
- направление/tracker10
- метка/xywh
- метка/kalman
- метка/bytetrack
- метка/30hz
исход: положительный
направление: Tracker 1,0 и слабые детекции
начат: '2026-09-13'
завершён: '2026-09-13'
---

# 🟢 XYWHByteTrack: модель рамки с независимыми шириной и высотой

↑ [[02 Эксперименты/Направления/Tracker 1,0 и слабые детекции|Tracker 1,0 и слабые детекции]] · **Исход:** положительный · `xywh_bytetrack`

Собственный вариант ByteTrack с фильтром Калмана по (x, y, w, h). На семи исходных видео ~30 Гц воспроизвёл BoT-SORT без компенсации камеры и без ReID.

## Вопрос
Даёт ли независимая оценка ширины и высоты рамки прирост при том же входе и тех же порогах?

## Ожидалось (до прогона)
HOTA/AssA/IDF1 не ниже штатного ByteTrack при сравнимой скорости.

## Что сделано
Реализован tracking/xywh_bytetrack.py; семивидео сравнение с ByteTrack, BoT-SORT без ReID и классическими полями движения; контроль реальной скорости.

## Получено
HOTA 52.44 → 54.57, AssA 52.97 → 54.57, IDF1 59.82 → 60.38, IDSw 573 → 526. Время ассоциации 1.95 мс/кадр, скорость конвейера ~31.5 кадр/с против 31.6 у ByteTrack.

## Решение
Принят как сильный общий контроль обеих линий (A и B).

## Ограничения
Семь видео; разбиений на обучающие/отложенные для порогов не было — пороги штатные (0.25 / 0.1 / буфер 30).

## Нельзя заявлять
Что XYWH лучше BoT-SORT с компенсацией камеры — сравнение проводилось без неё.

## Условия
- **baseline:** ByteTrack (ultralytics)
- **variant:** XYWHByteTrack
- **dataset:** BDD100K, исходные видео
- **split:** 7 видео val ~30 Гц, оценка на размеченных кадрах
- **detector:** YOLO26s seed42, 960
- **tracker:** XYWHByteTrack

## Критерии приёмки (записаны до прогона)
- ✅ HOTA, AssA, IDF1 не ниже ByteTrack на семи видео  
  *xywh_validation7.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 54.57308976106422 | 52.43696923664447 | 7 видео / ByteTrack | `runs/video_rate/xywh_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.HOTA |
| [AssA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAssA) | 54.571813137230265 | 52.97481252366944 | 7 видео / ByteTrack | `runs/video_rate/xywh_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.AssA |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 60.37998146431881 | 59.824299685065476 | 7 видео / ByteTrack | `runs/video_rate/xywh_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.IDF1 |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 526 | 573 | 7 видео / ByteTrack | `runs/video_rate/xywh_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics.IDSw |

## Доказательства
- метрики на семи видео: `runs/video_rate/xywh_validation7.json` · results.h0p25_l0p1_b30_m0p8.metrics
- отчёт об эксперименте 30 Гц: `FLYVIS_VIDEO_RATE_EXPERIMENT_REPORT.md` · 
- 13.09, 17:41 — сильные контроли: Журнал изменений

## Код
- `tracking/xywh_bytetrack.py`
- `tests/test_xywh_bytetrack.py`

## Данные
- `runs/video_rate/xywh_validation7`

## Коммиты
- `889913f`

