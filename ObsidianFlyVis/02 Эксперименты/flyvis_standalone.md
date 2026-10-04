---
aliases:
- Самостоятельный FlyVisTracker (линия A, первая версия)
tags:
- эксперимент
- исход/infrastructure
- направление/flyvis
- метка/flyvis
- метка/hungarian
- метка/lifecycle
- муха
исход: инфраструктура
направление: FlyVis
начат: '2026-09-13'
завершён: null
---

# 🔧 Самостоятельный FlyVisTracker (линия A, первая версия)

**Направление:** [[03 Направления/FlyVis|FlyVis]] · **Исход:** инфраструктура · `flyvis_standalone`

Автономный трекер после YOLO26: постоянные ID, венгерское назначение, состояния tentative / confirmed / lost / removed; FlyVis подключён через общий интерфейс поля движения, тем же каркасом проверяются Farnebäck и DIS.

## Ожидалось (до прогона)
Работающее собственное ядро с заменяемым источником движения — для последовательной абляции компонентов; низкий первый результат допустим.

## Что сделано
tracking/flyvis_tracker.py, tracking/flyvis_motion.py, адаптер FlyVis, тесты; позже (c5c0e20) выделены MotionFusionPolicy и AssignmentPolicy с численным совпадением стандартных модулей.

## Получено
Каркас для стресс-проверки; жизненный цикл и модель состояния ещё монолитны (этап 3 выполнен частично).

## Решение
Линия A сохраняется как экспериментальная; не копирует BoT-SORT под новым названием.

## Условия
- **dataset:** BDD100K, исходные видео
- **detector:** YOLO26s seed42, 960
- **tracker:** FlyVisTracker

## Доказательства
- add standalone FlyVis tracking experiments: `4690dad` · 
- этап 3 частично выполнен: `FLYVIS_TRACKER_RESEARCH_ROADMAP.md` · §6.4

## Код
- `tracking/flyvis_tracker.py`
- `tracking/flyvis_motion.py`
- `tracking/flyvis_adapter.py`
- `tests/test_flyvis_tracker.py`

## Коммиты
- `4690dad`
- `c5c0e20`

