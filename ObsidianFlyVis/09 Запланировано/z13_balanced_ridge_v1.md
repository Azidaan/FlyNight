---
aliases:
- 'З13: равный суммарный вес цели и фона при подгонке модели'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-22'
---

# 🔵 З13: равный суммарный вес цели и фона при подгонке модели

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_balanced_ridge_v1`

Критерии и фиксированные веса до расчёта.

## Вопрос
Улучшит ли фиксированная балансировка функции ошибки локализацию D и выбор её полезных ответов?

## Гипотеза
Равный суммарный вес наблюдаемой рамки и фона может изменить различающую модель; причинная роль дисбаланса пока не установлена.

## Ожидалось (до прогона)
Одна функция ошибки, прежняя гауссова цель,68признаков и lambda; D/C/R1,полные26train,проверенное воспроизведение и парные интервалы. Сначала стоимость/воспроизведение8эпизодов, без вывода об эффективности по пробе.

## Нельзя заявлять
Не воспроизведение DiMP, не причинное доказательство дисбаланса, не калиброванный отказ и не HOTA. Поисковый повторно используемый train26; прежние90%/50не заменяются. Ни ядро, ни detector/ByteTrack/MASA не обучаются и не меняются.

## Критерии приёмки (записаны до прогона)
- ⏳ Uniform control reproduces every saved8326row per arm: center<0.5px,score<0.001,identical saved lambda. All input hashes fixed.
- ⏳ D balanced-D uniform: mean paired-video useful-center fraction>=0.02 and95%lower>0 on all present train26 rows.
- ⏳ D balanced-D uniform:95%lower>0 for selected_success_fraction_positive at equal1%quota84.
- ⏳ D balanced-R1 balanced:95%lower>0 on all present train26 rows; R1 receives identical history and weighting.

## Доказательства
- Одна замена функции ошибки с критериями до исхода: `runs/z13_balanced_ridge_20261022/protocol.json` · 

## Код
- `tracking/z13_balanced_ridge.py`
- `tests/test_z13_balanced_ridge.py`
- `scripts/z13_balanced_ridge_20261022.py`

