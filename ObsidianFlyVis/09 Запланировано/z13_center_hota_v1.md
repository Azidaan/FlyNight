---
aliases:
- 'З13: HOTA сохранённого центрированного считывателя'
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/fly-problem-P8
исход: готов к запуску
направление: FlyVis
начат: '2026-10-21'
---

# ⚪ З13: HOTA сохранённого центрированного считывателя

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `z13_center_hota_v1`

Зарегистрировано подключение принятого центрированного считывателя к стенду; HOTA не измерена.

## Вопрос
Даёт ли ранее подтверждённый центрированный считыватель пользу полному стенду сверх отсутствия поправки, R1 и прежней D?

## Гипотеза
Функциональный выигрыш центров может переноситься в замкнутый трекер; это ещё не измерено.

## Ожидалось (до прогона)
Одна фиксированная диагностическая политика1% без подбора по dev/HOTA. C/D/R1, прежние35dev,YOLO26/ByteTrack/MASA без изменений. Техническая проба до1200с; далее только оставшееся окно автора, законченные видео сохраняются. Неполная очередь не даёт вердикта об эффективности.

## Нельзя заявлять
Эта политика диагностическая, прежний провал90%/50 не переписывается. Dev35 многократно использован, не независимое подтверждение. Сравнение полной реализации не изолирует только форму регрессионной цели; D−C отдельно, вклад анатомии не доказан. Неполные видео/очередь не измерение HOTA.

## Критерии приёмки (записаны до прогона)
- ⏳ Pure regression and incrementalR1 bit-exact tests; saved first8 natural train26 episode controls C/D/R1: center<0.5px,score<0.001,same threshold decisions, same lambda. Compiler/eager same tolerances and decisions. One complete train video tau=None output/frame IDs exactly equal stand. Same old MASA head recipe reproduced and frozen; all35 baseline stitched JSONs must equal prior stand before comparison.
- ⏳ Dcenter-stand: pooledHOTA>=+0.10 and pairedvideo95%lower>0
- ⏳ Dcenter-R1center: pairedvideo95%lower>0
- ⏳ Dcenter-legacyD: pairedvideo95%lower>0

## Доказательства
- Политика/критерии/срок до HOTA: `runs/z13_center_hota_20261021/protocol.json` · 
- Сохранённая ранее положительная функциональная версия: `runs/z13_target_center_dev_20261018/report.json` · 

## Код
- `tracking/z13_ridge_localizer.py`
- `scripts/z13_center_hota_20261021.py`
- `scripts/z13_center_hota_controller_20261021.py`

