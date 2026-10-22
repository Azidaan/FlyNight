---
aliases:
- 'З13: односторонняя ошибка фона в модели цели'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-22'
---

# 🔵 З13: односторонняя ошибка фона в модели цели

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_hinge_target_v1`

До расчёта зарегистрирована отдельная замена остаточной ошибки фона.

## Вопрос
Поможет ли прекращение штрафа уже подавленного фона улучшить D сверх согласованной балансировки, включая отбор полезных ответов?

## Гипотеза
Освобождение от регрессии точных низких значений фона может улучшить различающую модель; исправление отказа не установлено.

## Ожидалось (до прогона)
D/C/R1,полные26train,одна функция ошибки; прежние гауссова цель,веса0,5/0,5,68признаков,история,коэффициент регуляризации. CPUдо7200с,сначала стоимость/воспроизведение.

## Нельзя заявлять
Не полная DiMP и не её точная формула; не HOTA,не калиброванный отказ,не обучение734параметров. train26 используется повторно,старые90%/50не заменены.

## Критерии приёмки (записаны до прогона)
- ⏳ Reproduce previous balanced control for all8326rows/arm: center<0.5px,score<0.001,same lambda. Every observed solver converges at frozen tolerance; hashes unchanged.
- ⏳ D hinge-D balanced mean paired-video useful-center fraction>=0.02 and95%lower>0 on all present train26 rows.
- ⏳ D hinge-D balanced95%lower>0 for selected_success_fraction_positive at equal1%quota84.
- ⏳ D hinge-R1 hinge95%lower>0 on all present train26 rows, identical history and objective.

## Доказательства
- Одна замена с критериями до исхода: `runs/z13_hinge_target_20261023/protocol.json` · 
- Обязательный согласованный контроль и непройденный отказ: `runs/z13_balanced_ridge_20261022/report.json` · 

## Код
- `tracking/z13_hinge_target.py`
- `tests/test_z13_hinge_target.py`
- `scripts/z13_hinge_target_20261023.py`

