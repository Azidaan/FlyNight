---
aliases:
- 'З13: обучение D через модель цели с центрированной регрессией'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/Z13
- метка/fly-problem-P2
- метка/fly-problem-P8
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-18'
---

# 🟡 З13: обучение D через модель цели с центрированной регрессией

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_aligned_core_v1`

## Вопрос
Улучшает ли новый режим обучения D её центры относительно начальной D и штатной C?

## Гипотеза
Градиент через применяемую модель цели может сделать признаки D полезнее; причинность прежнего отставания не установлена.

## Ожидалось (до прогона)
250 шагов с фиксированным условием продолжения до1000; только ядро734. НачальнаяD воспроизводится, считывание и нормировка фиксированы. Dev35 только после полного train-критерия; HOTA/75 не запускаются.

## Что сделано
2026-10-18

## Получено
runs/z13_aligned_core_20261018/report.json: {"stages": {"250": {"paired": {"D_adapted-D_center": {"mean_per_video": -0.023298048830072363, "ci95": [-0.04691967773201367, 0.00026424445447485013], "better": 6, "n": 26}, "D_adapted-C_center": {"mean_per_video": -0.026270126835693377, "ci95": [-0.07473049994354074, 0.010352550999269204], "better": 15, "n": 26}, "D_adapted-R1_center": {"mean_per_video": 0.14288394774112867, "ci95": [0.10338471969913246, 0.1908572635275388], "better": 25, "n": 26}}, "primary_pass": false, "summary": {"rows": 8326, "positives": 1786, "useful": 375, "useful_mean_video": 0.23158776147325486, "coverage": {"0.01": {"selected": 84, "known": 64, "unknown": 20, "useful": 3, "harmful": 61, "useful_share_known": 0.046875}, "0.02": {"selected": 167, "known": 128, "unknown": 39, "useful": 6, "harmful": 122, "useful_share_known": 0.046875}, "0.05": {"selected": 417, "known": 338, "unknown": 79, "useful": 7, "harmful": 331, "useful_share_known": 0.020710059171597635}}}}}, "decision": "stop_after_predeclared_pilot", "steps": 250, "dev_evaluated": false, "hota_measured": false, "limitations": "Tests one complete new training regime, not a causal isolation of alignment alone. One seed, reused train26/dev. No HOTA. Utility gate failed, so tracker intervention is not authorized by these metrics alone."}

## Интерпретация
Заранее записанный критерий продолжения или полного преимущества не выполнен; эта реализация не подтверждена. Это не закрывает линию FlyVis.

## Решение
stop_after_predeclared_pilot

## Нельзя заявлять
Нет HOTA, проверки отложенных75 или причинного доказательства, что именно согласование объясняло прежний разрыв D/C. Один сид; train26 и dev повторно использованы.

## Критерии приёмки (записаны до прогона)
- ✅ Before updates: all initialD train26 rows paired, max center diff<0.5px, score diff<0.001; lambda unchanged. Do not weaken tolerances.  
  *runs/z13_aligned_core_20261018/report.json*
- ⏳ At1000steps: lower95%video CI Dadapted-Dinitial>0, mean>=2pp, and lower95%CI Dadapted-Ccenter>0. Other comparisons descriptive. No checkpoint selection by dev.  
  *runs/z13_aligned_core_20261018/report.json*

## Доказательства
- Постановка, бюджет и критерии до GPU: `runs/z13_aligned_core_20261018/protocol.json` · 
- Пройденная техническая проверка градиента без обновления весов: `runs/z13_parallel_next_20261018/differentiable/report.json` · 
- Фактический запуск, устройство и независимый срок: `runs/z13_aligned_core_20261018/controller.json` · 
- Реальное окончание своего дерева: `runs/z13_aligned_core_20261018/controller_result.json` · 
- Результаты фиксированных ступеней: `runs/z13_aligned_core_20261018/report.json` · 

## Код
- `scripts/z13_aligned_core.py`
- `tracking/z13_target_center_torch.py`
- `tracking/z13_aligned_loss.py`

## Данные
- `runs/z13_aligned_core_20261018`

