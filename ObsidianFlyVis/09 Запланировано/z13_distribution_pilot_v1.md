---
aliases:
- 'З13: распределённая пространственная цель считывания, ограниченный пилот'
tags:
- эксперимент
- исход/running
- направление/flyvis
исход: идёт
направление: FlyVis
начат: '2026-10-15'
---

# 🔵 З13: распределённая пространственная цель считывания, ограниченный пилот

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_distribution_pilot_v1`

## Вопрос
Does a spatial target improve localization compared with identical extra training on a one-hot target?

## Гипотеза
Учёт расстояния до цели в обучении считывания может уменьшить ошибку положения; причинность прежнего провала не установлена.

## Ожидалось (до прогона)
Парные продолжения hard/spatial из одинаковых весов B/C/D, 500 шагов каждое, только считыватель; train104/26. Одно окно до 20:00 MSK.

## Нельзя заявлять
Не проверяются HOTA, независимое подтверждение, новое обучение ядра или реализация всего PrDiMP.

## Критерии приёмки (записаны до прогона)
- ⏳ D spatial minus D hard useful-position fraction on known-positive rows: paired video CI lower >0
- ⏳ D spatial mean of per-video median relative center errors <= D hard
- ⏳ Separate FlyVis signal gate: D spatial minus B spatial useful-position fraction CI lower >0

## Доказательства
- Постановка до обучения: `runs/z13_distribution_pilot_20261015/protocol.json` · 
- Обзор и ссылка на первоисточник PrDiMP; не готовая FlyVis: `research_flyvis_decision_trees_2026-10-15/СВОДКА.md` · 
- Один собственный процесс и независимый предел 20:00: `runs\z13_distribution_pilot_20261015\controller.json` · 

