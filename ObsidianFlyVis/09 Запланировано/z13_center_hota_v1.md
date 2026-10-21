---
aliases:
- 'З13: HOTA сохранённого центрированного считывателя'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-21'
---

# 🔵 З13: HOTA сохранённого центрированного считывателя

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_center_hota_v1`

Технические критерии выполнены; идёт неизменная очередь35видео. Полной HOTA ещё нет.

## Вопрос
Даёт ли ранее подтверждённый центрированный считыватель пользу полному стенду сверх отсутствия поправки, R1 и прежней D?

## Гипотеза
Функциональный выигрыш центров может переноситься в замкнутый трекер; это ещё не измерено.

## Ожидалось (до прогона)
Одна фиксированная диагностическая политика1% без подбора по dev/HOTA. C/D/R1, прежние35dev,YOLO26/ByteTrack/MASA без изменений. Техническая проба до1200с; далее только оставшееся окно автора, законченные видео сохраняются. Неполная очередь не даёт вердикта об эффективности.

## Нельзя заявлять
Эта политика диагностическая, прежний провал90%/50 не переписывается. Dev35 многократно использован, не независимое подтверждение. Сравнение полной реализации не изолирует только форму регрессионной цели; D−C отдельно, вклад анатомии не доказан. Неполные видео/очередь не измерение HOTA.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-21 (42052207d76850dd4c4c19e246e7443014a40b1e) — Фактический запуск зарегистрированной пробы.
  > Зарегистрировано подключение принятого центрированного считывателя к стенду; HOTA не измерена.
- **Суть**, 2026-10-21 (a6a11d83911faef52aaca4f0db1701b5b56a5a0b) — Полная очередь после технического допуска.
  > Идёт техническое воспроизведение и замер полной версии; HOTA ещё не измерялась.

## Критерии приёмки (записаны до прогона)
- ✅ Pure regression and incrementalR1 bit-exact tests; saved first8 natural train26 episode controls C/D/R1: center<0.5px,score<0.001,same threshold decisions, same lambda. Compiler/eager same tolerances and decisions. One complete train video tau=None output/frame IDs exactly equal stand. Same old MASA head recipe reproduced and frozen; all35 baseline stitched JSONs must equal prior stand before comparison.  
  *runs/z13_center_hota_20261021/technical.json*
- ⏳ Dcenter-stand: pooledHOTA>=+0.10 and pairedvideo95%lower>0
- ⏳ Dcenter-R1center: pairedvideo95%lower>0
- ⏳ Dcenter-legacyD: pairedvideo95%lower>0

## Доказательства
- Политика/критерии/срок до HOTA: `runs/z13_center_hota_20261021/protocol.json` · 
- Сохранённая ранее положительная функциональная версия: `runs/z13_target_center_dev_20261018/report.json` · 
- Фактический запуск и неизменные источники: `runs/z13_center_hota_20261021/probe_launch.json` · 
- Фактический запуск и неизменные источники: `runs/z13_center_hota_20261021/service_contract.json` · 
- Отдельное завершение только неиспользованного вычислительного бюджета пробы: `runs/z13_center_hota_20261021/completion_contract.json` · 
- Отдельное проверенное служебное восстановление; прежняя ошибка сохранена: `runs/z13_center_hota_20261021/recovery1_contract.json` · 
- Отдельное проверенное служебное восстановление; прежняя ошибка сохранена: `runs/z13_center_hota_20261021/recovery1_reproduction.json` · 
- Отдельное проверенное служебное восстановление; прежняя ошибка сохранена: `runs/z13_center_hota_20261021/probe_completion_controller_result.json` · 
- Технический допуск и полный запуск: `runs/z13_center_hota_20261021/technical.json` · 
- Технический допуск и полный запуск: `runs/z13_center_hota_20261021/full_launch.json` · 
- Технический допуск и полный запуск: `runs/z13_center_hota_20261021/execution_identity.json` · 

## Код
- `tracking/z13_ridge_localizer.py`
- `scripts/z13_center_hota_20261021.py`
- `scripts/z13_center_hota_controller_20261021.py`

