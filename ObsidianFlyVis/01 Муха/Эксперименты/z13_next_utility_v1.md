---
aliases:
- 'З13: оценка полезности по наблюдаемым признакам'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/Z13
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-18'
---

# 🟢 З13: оценка полезности по наблюдаемым признакам

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_next_utility_v1`

## Вопрос
Поможет ли обучение точному событию полезности при фиксированных центрах D?

## Гипотеза
Learn a utility output on fixed saved centers; no response maps in this limited implementation.

## Ожидалось (до прогона)
At global top1%, both paired95%video-bootstrap lower bounds of D utility minus original and D utility minus geometry exceed0. Secondary2/5% cannot rescue failure.

## Что сделано
2026-10-18

## Получено
runs/z13_parallel_next_20261018/utility/report.json (arms.D.coverage): {"original": {"0.01": {"selected": 84, "known": 64, "unknown": 20, "useful": 2, "harmful": 62, "useful_share_known": 0.03125}, "0.02": {"selected": 167, "known": 132, "unknown": 35, "useful": 4, "harmful": 128, "useful_share_known": 0.030303030303030304}, "0.05": {"selected": 417, "known": 338, "unknown": 79, "useful": 9, "harmful": 329, "useful_share_known": 0.026627218934911243}}, "geometry": {"0.01": {"selected": 84, "known": 38, "unknown": 46, "useful": 5, "harmful": 33, "useful_share_known": 0.13157894736842105}, "0.02": {"selected": 167, "known": 83, "unknown": 84, "useful": 18, "harmful": 65, "useful_share_known": 0.21686746987951808}, "0.05": {"selected": 417, "known": 232, "unknown": 185, "useful": 47, "harmful": 185, "useful_share_known": 0.2025862068965517}}, "utility": {"0.01": {"selected": 84, "known": 36, "unknown": 48, "useful": 5, "harmful": 31, "useful_share_known": 0.1388888888888889}, "0.02": {"selected": 167, "known": 82, "unknown": 85, "useful": 17, "harmful": 65, "useful_share_known": 0.2073170731707317}, "0.05": {"selected": 417, "known": 233, "unknown": 184, "useful": 49, "harmful": 184, "useful_share_known": 0.21030042918454936}}}

## Интерпретация
Ограниченный оценщик по геометрии и одному баллу не прошёл основной критерий; преимущество над геометрией не доказано. Полные карты ответа не проверялись.

## Решение
limited_observables_utility_gate_not_passed

## Нельзя заявлять
Нет HOTA, доказательства пользы стенду, гарантии риска на неизвестных исходах или вывода о неработоспособности FlyVis.

## Критерии приёмки (записаны до прогона)
- ✅ Разделение по видео и исключение истинных ошибок/рамок из входов проверены  
  *runs/z13_parallel_next_20261018/utility/report.json*
- ❌ At global top1%, both paired95%video-bootstrap lower bounds of D utility minus original and D utility minus geometry exceed0. Secondary2/5% cannot rescue failure.  
  *runs/z13_parallel_next_20261018/utility/report.json*

## Доказательства
- Условия до запуска: `runs/z13_parallel_next_20261018/protocol.json` · 
- Полный первичный результат: `runs/z13_parallel_next_20261018/utility/report.json` · 

## Код
- `scripts/z13_parallel_next.py`

## Данные
- `runs/z13_parallel_next_20261018/utility`

