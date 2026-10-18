---
aliases:
- 'З13: перенос оценки полезности C/D/R1 с train26 на dev35'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/Z13
- метка/fly-problem-P8
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-18'
---

# 🟡 З13: перенос оценки полезности C/D/R1 с train26 на dev35

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_map_transfer_v1`

## Вопрос
Переносится ли преимущество оценки карты C сверх геометрии на другие видео?

## Гипотеза
Вспомогательный положительный результат C на train26 может переноситься; это гипотеза, выбор C сделан после просмотра train26.

## Ожидалось (до прогона)
At1%: C map minus original, geometry and limited each lower95%paired-video CI>0. Separate FlyVis gate: C map minus R1 map lower95%CI>0. Both required to continue. D and2/5% secondary, cannot rescue primary.

## Что сделано
2026-10-18

## Получено
runs/z13_map_transfer_20261018/report.json: {"paired": {"C_map_utility-R1_map_utility": {"mean_per_video": 0.005365867431023146, "ci95": [-0.00018939068450346628, 0.013498889740670842], "better": 7, "n": 35}, "C_map_utility-D_map_utility": {"mean_per_video": -0.0028076530635480473, "ci95": [-0.009873493577950981, 0.004197872251848711], "better": 7, "n": 35}}, "primary_pass": false, "fly_control_pass": false, "proceed": false, "decision": "ranking_transfer_not_confirmed; no_dev_retuning_or_tracker_application", "dev_evaluated": true, "hota_measured": false, "cuda_initialized": false, "limitations": "C selected from auxiliary results. Reused dev35 is not blind. Core D fit104; head train26; evaluation dev35 disjoint. One seed, no safety guarantee for unknowns, no HOTA, no heldout75.", "deployment_caveat": "Global1% rank is diagnostic and uses all scores; not an online threshold or risk guarantee. No tracker application in this protocol."}; arms.C.paired_selected_success.0.01: {"map_utility-original": {"mean_per_video": 0.007515924762081761, "ci95": [0.002248085112719259, 0.01535670708786101], "better": 10, "n": 35}, "map_utility-geometry": {"mean_per_video": -6.235081784428627e-05, "ci95": [-0.00861369993102984, 0.009575725210015023], "better": 4, "n": 35}, "map_utility-limited": {"mean_per_video": -0.000901141656635125, "ci95": [-0.0099134835888687, 0.008931056473626122], "better": 3, "n": 35}}

## Интерпретация
Перенос ранжирования C не подтверждён по заданным критериям; интервал с нулём означает неопределённость, а не равенство. Новый подбор поdev не выполняется.

## Решение
ranking_transfer_not_confirmed; no_dev_retuning_or_tracker_application

## Нельзя заявлять
Нет HOTA, проверки75, слепогоdev или гарантии безопасности неизвестных исходов. Глобальный отбор1% не является рабочим онлайн-порогом.

## Критерии приёмки (записаны до прогона)
- ✅ Every dev row keeps video/label/kind/useful; unchanged center difference<0.5px and score difference<0.001; fit20 normalization identical to saved dev. Use saved centers/outcomes for comparisons.  
  *runs/z13_map_transfer_20261018/report.json*
- ❌ At1%: C map minus original, geometry and limited each lower95%paired-video CI>0. Separate FlyVis gate: C map minus R1 map lower95%CI>0. Both required to continue. D and2/5% secondary, cannot rescue primary.  
  *runs/z13_map_transfer_20261018/report.json*

## Доказательства
- Критерии, ограничения и исходные хеши до переноса: `runs/z13_map_transfer_20261018/protocol.json` · 
- Полный либо частичный первичный итог переноса: `runs/z13_map_transfer_20261018/report.json` · 

## Код
- `scripts/z13_map_transfer.py`
- `scripts/z13_map_transfer_progress.ps1`
- `tracking/z13_utility_transfer.py`
- `tests/test_z13_utility_transfer.py`

## Данные
- `runs/z13_map_transfer_20261018`

