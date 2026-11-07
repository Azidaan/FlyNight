---
aliases:
- 'Диагностика: нелинейная модель на признаках считывания FlyVis (tiled и фовеация)'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/goal2
- метка/diagnostic
- метка/nonlinear
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-09-16'
завершён: '2026-09-16'
---

# 🔴 Диагностика: нелинейная модель на признаках считывания FlyVis (tiled и фовеация)

↑ [[01 Муха/Темы/Считывание активности и события|Считывание активности и события]] · **Исход:** отрицательный · `flyvis_readout_nonlinear_diag`

Не критерий. Проверка возражения «логистическая регрессия слишком слаба»: HistGradientBoosting на тех же парах и разбиении по группам base+DIS+eye, +decoded, +fovea T4T5/all34, +shuffled, +tiled T4T5/all34, +fovea all34+decoded.

## Вопрос
Появляется ли условный вклад считывания FlyVis при нелинейной модели там, где линейная его не нашла?

## Ожидалось (до прогона)
Если и здесь прирост над base+DIS+eye не устойчив по видео и не превосходит decoded — закрытие по точке A1 устойчиво к выбору модели.

## Что сделано
Скрипт scripts/diag_readout_nonlinear.py; запуск 16.09, 02:3x MSK на CPU.

## Получено
HGB на отложенных: base+DIS+eye AP 0,967989 (Brier 0,010923); +decoded 0,967990; +fovea T4T5 0,966360; +fovea all34 0,967054; +shuffled fovea 0,965983; +tiled T4T5 0,967031; +tiled all34 0,966626; +fovea all34+decoded 0,966787. Ни один вариант FlyVis не выше базы; повидео-разности fovea all34 − база: −0,00014, −0,00073, −0,00172.

## Решение
Возражение о слабости линейной модели снято: при нелинейной модели вклад считывания FlyVis отсутствует. Закрытие по A1 устойчиво.

## Ограничения
Один сид, без настройки гиперпараметров; сравнение по повидео-разностям AP без бутстрапа.

## Критерии приёмки (записаны до прогона)
- ✅ диагностика выполнена, повидео-разности AP записаны  
  *runs/qmatch_readout_gate/nonlinear_diag/report.json → per_video_ap_deltas*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9670541805276605 | 0.9679893852465502 | HGB, отложенные: +fovea all34 / base+DIS+eye | `runs/qmatch_readout_gate/nonlinear_diag/report.json` · feature_groups.base_dis_eye_fovea_all34.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.967990066444518 | 0.9679893852465502 | HGB: +decoded / base | `runs/qmatch_readout_gate/nonlinear_diag/report.json` · feature_groups.base_dis_eye_decoded.test.average_precision |

## Доказательства
- решение по A1 должно быть устойчиво к очевидным возражениям: 1.6. Точки остановки линии A

## Код
- `scripts/diag_readout_nonlinear.py`

## Данные
- `runs/qmatch_readout_gate/nonlinear_diag`

