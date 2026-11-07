---
aliases:
- 'Б2 — временная ветвь перед детектором: проверка обучения по его ошибке'
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/infrastructure
исход: готов к запуску
направление: FlyVis
начат: '2026-10-27'
---

# ⚪ Б2 — временная ветвь перед детектором: проверка обучения по его ошибке

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `b2_predetection_gradient_v1`

Готовая ограниченная проверка реализации; в локальной очереди после Б1. Научная гипотеза не проверена.

## Вопрос
Проходит ли обучающая ошибка по рабочему пути выбранного узла с реальными входами и согласованными контролями?

## Ожидалось (до прогона)
Insert additive temporal feature correction in first YOLO detection-head input P3, before boxes/scores are produced. Same input/output adapters across arms; YOLO weights frozen. Three scratch Adam updates for common adapters and for free core parameters in D; objective is frozen one-to-one YOLO detection loss on the last frame. The final head forward_head removes only input detach, retaining fixed head values and weights. Scratch weights are not promoted.

## Нельзя заявлять
Implementation and cost on one real clip, not training convergence, detector quality, or scientific B2 acceptance. Ordinary recurrent control parameter count differs and is reported.

## Критерии приёмки (записаны до прогона)
- ⏳ At zero output all raw one-to-one detector boxes/scores match the same frozen detector exactly
- ⏳ After output warm-up, real detection loss produces finite nonzero core gradients in D and adapter gradients in all arms
- ⏳ YOLO parameters unchanged; record actual steps, memory, time, per-arm parameters; no HOTA claim

## Доказательства
- Условия и критерии до расчёта: `runs/b123_20261027/b2_gradient/protocol.json` · 
- Автор разрешил Б1–Б3 до 07:30 МСК: `runs/b123_20261027/authorization.json` · 

## Код
- `tracking/b2_feature_temporal.py`
- `scripts/b2_gradient_profile_20261027.py`

