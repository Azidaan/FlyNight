---
aliases:
- 'Цель 0: заморозка воспроизводимого стенда трекера (7 видео, 30 Гц)'
tags:
- эксперимент
- исход/infrastructure
- направление/infra
- метка/goal0
- метка/manifest
- метка/reproducibility
- метка/tracker
исход: инфраструктура
направление: Инфраструктура и данные
начат: '2026-09-16'
завершён: '2026-09-16'
---

# 🔧 Цель 0: заморозка воспроизводимого стенда трекера (7 видео, 30 Гц)

**Направление:** [[03 Направления/Инфраструктура и данные|Инфраструктура и данные]] · **Исход:** инфраструктура · `bench_freeze_v1`

Манифест хэшей весов, видео, кэшей детекций и полей движения, версий библиотек и параметров трекеров; повторный прогон неизменённых XYWHByteTrack, ByteTrack и FlyVisTracker по сохранённым кэшам с побайтовым сравнением рамок и ID с опубликованными выходами; проверка тождества двух копий кэша детекций; инвариантность к порядку видео; статическая проверка отсутствия доступа к разметке в tracking/.

## Вопрос
Воспроизводится ли контрольная точка: дают ли неизменённые трекеры на том же кэше те же рамки, ID и метрики, что опубликованы в runs/video_rate и runs/flyvis_stress_gate?

## Ожидалось (до прогона)
Тест воспроизводимости, не прирост. Ожидается нулевое расхождение рамок/ID у XYWHByteTrack (h0.25/l0.1/b30/m0.8) с runs/video_rate/xywh_validation7 и у flyvis_full_calibrated / flyvis_tiled_calibrated / bytetrack с runs/flyvis_stress_gate/<video>/results; тождество npz кэшей детекций в runs/video_rate и runs/flyvis_stress_gate; независимость выходов от порядка видео.

## Что сделано
Написаны scripts/freeze_bench.py и tests/test_bench_freeze.py (10 тестов, RED→GREEN). Прогон 326 с на CPU с приоритетом Idle: манифест (sha256 весов, 7 видео, 82 файла кэшей и разметки; версии ultralytics 8.4.84, trackeval 1.3.0, flyvis 1.2.0 из tools/flyvis-runtime; параметры трекеров), тождество двух копий кэша детекций, повторные прогоны XYWHByteTrack (h0.25/l0.1/b30/m0.8), bytetrack, flyvis_full_calibrated, flyvis_tiled_calibrated по сохранённым кэшам, обратный порядок видео, статическая проверка tracking/. Два первых запуска упали до стадии 2 по инфраструктурным причинам (отсутствующий motion/dis.json у трёх стресс-видео — частота теперь берётся из любого motion/*.json; неверные имена аргументов run_configuration) и не считаются попытками опыта.

## Получено
Все семь критериев выполнены: кэши детекций тождественны по всем 7 видео (397 025 детекций); рамки, ID, классы и баллы повторных прогонов совпали с опубликованными покадрово без единого расхождения (XYWH: 1 318 кадров, 13 144 рамки; три стресс-трекера — по 7 видео каждый); разности всех метрик TrackEval (HOTA, DetA, AssA, IDF1, MOTA, IDSw, FP, FN, Frag) равны нулю; обратный порядок видео даёт тождественные выходы; в tracking/ нет обращений к разметке. Частота по контейнеру 29,77–30,13 кадр/с; размеченных кадров 1 318 из 7 857.

## Решение
Стенд заморожен; цель 0 закрыта. Цели 1, 1а и 2 разрешены. Сравнение новых вариантов ведётся только с этим манифестом и оценщиком trackeval 1.3.0.

## Ограничения
Воспроизведение идёт по сохранённым кэшам детекций и полей движения; повторная инференция YOLO26 и FlyVis не выполняется, поэтому детерминизм самих детектора и сети FlyVis этой карточкой не проверяется. Исходных обучающих видео нет — разбиение train/dev/test в манифест не входит (цель 1а).

## Нельзя заявлять
Совпадение повторного прогона не является научным результатом и не подтверждает ни один прирост.

## Условия
- **dataset:** 7 исходных val-видео BDD100K ~30 Гц; кэши runs/video_rate и runs/flyvis_stress_gate
- **split:** не применяется: воспроизведение опубликованных выходов на тех же видео
- **detector:** weights/baseline_seed42/best.pt, 960 px, conf 0.01, NMS IoU 0.7 (из identity кэша)
- **tracker:** XYWHByteTrack, BYTETracker (ultralytics 8.4.84), FlyVisTracker по кэшу полей

## Критерии приёмки (записаны до прогона)
- ✅ две копии кэша детекций (video_rate, flyvis_stress_gate) побайтово тождественны по всем 7 видео  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ повторный прогон XYWHByteTrack даёт те же рамки и ID, что runs/video_rate/xywh_validation7, на каждом видео  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ повторный прогон FlyVisTracker (full и tiled, калиброванный) и ByteTrack даёт те же рамки и ID, что runs/flyvis_stress_gate/<video>/results  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ метрики TrackEval повторного прогона совпадают с опубликованными сводками  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ выходы не зависят от порядка обработки видео (прямой и обратный порядок тождественны)  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ tracking/ не импортирует и не читает разметку (статическая проверка)  
  *runs/bench_freeze/reproduction.json → acceptance*
- ✅ манифест содержит sha256 весов, видео, кэшей, версий библиотек, параметров трекеров и версии оценщика  
  *runs/bench_freeze/reproduction.json → acceptance*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [[04 Термины и метрики/HOTA|HOTA]] | 54.57308976106422 | 54.57308976106422 | 7 видео, XYWHByteTrack, повторный прогон против xywh_validation7 | `runs/bench_freeze/reproduction.json` · xywh.metrics.HOTA.rerun |
| [[04 Термины и метрики/HOTA|HOTA]] | 29.112995077709662 | 29.112995077709662 | 7 видео, flyvis_tiled_calibrated, повторный прогон против flyvis_stress_gate | `runs/bench_freeze/reproduction.json` · stress.flyvis_tiled_calibrated.metrics.HOTA.rerun |

## Доказательства
- критерии и выходы цели 0: 3.1. Цель 0
- семь булевых критериев, все true: `runs/bench_freeze/reproduction.json` · acceptance
- хэши и версии стенда: `runs/bench_freeze/manifest.json` · libraries, weights, videos, caches
- 16.09 — запись о закрытии цели 0: Журнал изменений

## Код
- `scripts/freeze_bench.py`
- `tests/test_bench_freeze.py`

## Данные
- `runs/bench_freeze`

