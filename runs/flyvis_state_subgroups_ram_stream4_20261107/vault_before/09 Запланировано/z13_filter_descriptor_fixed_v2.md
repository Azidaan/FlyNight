---
aliases:
- 'З13: коэффициенты цели на неизменных кандидатах,версия2'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
- метка/DiMP
- метка/engineering-future
исход: идёт
направление: FlyVis
начат: '2026-10-24'
---

# 🔵 З13: коэффициенты цели на неизменных кандидатах,версия2

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_filter_descriptor_fixed_v2`

Новая версия технического переноса зарегистрирована до исходов; научная проверка коэффициентов/равного контроля прежняя.

## Вопрос
Помогают ли наблюдаемые коэффициенты различающего фильтра сверх среднего образца при строго неизменных кандидатах,архитектуре и бюджете обучения?

## Гипотеза
Current response map already uses this filter; coefficients may add no information. Coefficients are not feature means, hence separate descriptor embedding in both matched arms. This test does not diagnose intrinsic representational limits.

## Ожидалось (до прогона)
Исходные104train/26оценки,те же36CPUмоделей/3сида/1150шагов145950предъявлений. CPUизвлечение18000с,обучение1800с,до26.10 07:40МСК. Версия1 сохраняется с невоспроизведённым дискретным вторичным максимумом.

## Нельзя заявлять
Не доказано побитовое воспроизведение прежнего GPUпрохода или прежнего выбора вторичных максимумов. Кандидаты научного опыта берутся только из исходного неизменного кэша. Карта уже использует фильтр; нет гарантии новой информации,причинного диагноза всейFlyVis,безопасногоотказа илиHOTA. Повторно используемыеtrain26.

## Критерии приёмки (записаны до прогона)
- ⏳ Version2 before outcomes: all immutable cached query candidates/centers/features/scores and labels used by both matched arms remain byte-for-byte unchanged (source hashes). Observed descriptors: projection<=1e-10,seed maxabs<=.001; reproduce original fixed observed anchors count,centers<.5px,scores<.001; additional candidate-feature maxabs<=.001 for new chunks. Do not regenerate candidate sets in the scientific comparison. Original version failed regenerated peak-set replay and remains inconclusive; this version does not assert that predicate is reproduced. Log all regenerated selection differences. Previously successful chunks retain their stricter original regeneration checks and hashes,not recomputed. Все9наборов/18старыхмоделей/32шага проверены.
- ⏳ D ridge-D mean >=2п.п. и нижняя95%повидеоДИ>0.
- ⏳ D ridge лучше D old activity,D old geometry,R1 ridge,D ridge geometry; все нижние95%ДИ>0.

## Доказательства
- Фиксированные данные и критерии до исходов версии2: `runs/z13_filter_descriptor_fixed_20261024/cache/protocol.json` · 

## Код
- `scripts/z13_filter_descriptor_fixed_20261024.py`
- `tracking/z13_fixed_candidate_replay.py`
- `tests/test_z13_fixed_candidate_replay.py`
- `scripts/z13_filter_descriptor_fixed_console.ps1`

