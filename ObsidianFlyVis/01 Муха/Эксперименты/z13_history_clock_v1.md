---
aliases:
- Временное состояние FlyVis по реальному интервалу между кадрами
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P3
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Временное состояние FlyVis по реальному интервалу между кадрами

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_history_clock_v1`

Реальное время состояния: критерий не пройден.

## Вопрос
Улучшит ли реальное прошедшее время внутри состояния мухи полезные назначения при том же считывателе?

## Гипотеза
Рабочая гипотеза: сжатие интервалов и добавление времени на повторённые записи могут искажать динамические признаки; это не доказанная причина ошибок.

## Ожидалось (до прогона)
Change only the internal temporal resampling rule: a stored interval delta_video_frames advances the existing frozen dynamics by delta nominal30Hz steps, with linear stimulus interpolation; delta0 advances neither neural nor rich-retinal EMA state. Keep all genuine samples, same crops, weights, normalization, observed-center ridge target, saved lambda, readout and alpha0.25.

## Что сделано
2026-10-25

## Получено
runs/z13_history_clock_20261025/replay/report.json → paired: {'Dclock-baseline': {'mean_per_video': -0.0008777015120375459, 'ci95': [-0.002838969849595577, 0.0008949421304652833], 'better': 5, 'n': 26}, 'Dclock-Dold': {'mean_per_video': 0.0001429796968830426, 'ci95': [0.0, 0.0004289390906491278], 'better': 1, 'n': 26}, 'Dclock-C_clock': {'mean_per_video': -0.00016752838607930886, 'ci95': [-0.0005720742410910249, 0.00016943409013893597], 'better': 1, 'n': 26}, 'Dclock-R1_clock': {'mean_per_video': 0.0001509669993730154, 'ci95': [-0.0014031898523073415, 0.0015444980454233873], 'better': 7, 'n': 26}, 'Dclock-geometry': {'mean_per_video': -0.0006648226816149205, 'ci95': [-0.0017586562460666724, 0.00019403619854544058], 'better': 2, 'n': 26}, 'clock_effect_D-R1': {'mean_per_video': 0.0003124137870219786, 'ci95': [0.0, 0.0007942616641828931], 'better': 2, 'n': 26}}; центры отдельно runs/z13_history_clock_20261025/features/report.json.

## Интерпретация
Изменён только ход внутреннего времени при тех же весах, данных и считывателе. Это последствия первых назначений на повторно использованных train26, не HOTA; частный эффект центров не спасает провал назначения.

## Решение
Pass -> separately budgeted frozen-runtime clock parity/price and matched closed-loop HOTA. Fail -> preserve this actual-clock implementation and allC/R1 controls; do not retune substep/time constants/lambda on26 or start core training by default. Choose next internal input/state node only with a new measured prerequisite. Основной критерий: False

## Нельзя заявлять
Linear interpolation cannot recover the actual missing images, and more realistic elapsed time can be worse for D trained on the shortened timeline. Old normalization/lambda remain fixed and may not suit changed state distributions. The first search delay is a data/runtime design choice, not proven scientific failure. No claim of HOTA, cause or benefit from gap counts alone; repeatedtrain26 exploratory.

## Критерии приёмки (записаны до прогона)
- ✅ Unit-clock features exactly equal original with tested mock backbone and real rich retina; mixed zero/gap node+edge state masking tested. First2groups of each C/D/R1 old path must reproduce saved centers<0.5px/scores<0.001. Every8326map/label/crop binding retained. First2Dgroups conservative actual virtual-step rate fits1800remaining, or preserve partial and stop. Subsequent26CPUreplays all old branch outcomes exactly equal previous mean decoder atalpha0.25; rows/hashes/intervals independently verified.  
  *runs/z13_history_clock_20261025/independent_audit.json*
- ❌ Dclock-baseline and Dclock-Dold net-beneficial mean-video frequencies both>=2pp; all6paired95%lower>0 including Cclock,R1clock,geometry and (Dclock-Dold)-(R1clock-R1old). Conditional center shifts are secondary and cannot rescue a failed assignment gate.  
  *runs/z13_history_clock_20261025/replay/report.json*

## Доказательства
- Новый временной узел, равные контроли, фиксированные параметры и бюджеты до откликов: `runs/z13_history_clock_20261025/protocol.json` · 

