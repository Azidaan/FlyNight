---
aliases:
- 'З13: обучение D через модель цели с центрированной регрессией'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/Z13
- метка/fly-problem-P2
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-18'
---

# 🔵 З13: обучение D через модель цели с центрированной регрессией

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_aligned_core_v1`

## Вопрос
Улучшает ли новый режим обучения D её центры относительно начальной D и штатной C?

## Гипотеза
Градиент через применяемую модель цели может сделать признаки D полезнее; причинность прежнего отставания не установлена.

## Ожидалось (до прогона)
250 шагов с фиксированным условием продолжения до1000; только ядро734. НачальнаяD воспроизводится, считывание и нормировка фиксированы. Dev35 только после полного train-критерия; HOTA/75 не запускаются.

## Критерии приёмки (записаны до прогона)
- ⏳ Before updates: all initialD train26 rows paired, max center diff<0.5px, score diff<0.001; lambda unchanged. Do not weaken tolerances.
- ⏳ At1000steps: lower95%video CI Dadapted-Dinitial>0, mean>=2pp, and lower95%CI Dadapted-Ccenter>0. Other comparisons descriptive. No checkpoint selection by dev.

## Доказательства
- Постановка, бюджет и критерии до GPU: `runs/z13_aligned_core_20261018/protocol.json` · 
- Пройденная техническая проверка градиента без обновления весов: `runs/z13_parallel_next_20261018/differentiable/report.json` · 
- Фактический запуск, устройство и независимый срок: `runs/z13_aligned_core_20261018/controller.json` · 

## Код
- `scripts/z13_aligned_core.py`
- `tracking/z13_target_center_torch.py`
- `tracking/z13_aligned_loss.py`

## Данные
- `runs/z13_aligned_core_20261018`

