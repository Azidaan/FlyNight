---
aliases:
- Временное состояние FlyVis по реальному интервалу между кадрами
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/fly-problem-P3
- метка/fly-problem-P8
исход: готов к запуску
направление: FlyVis
начат: '2026-10-25'
---

# ⚪ Временное состояние FlyVis по реальному интервалу между кадрами

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `z13_history_clock_v1`

## Вопрос
Улучшит ли реальное прошедшее время внутри состояния мухи полезные назначения при том же считывателе?

## Гипотеза
Рабочая гипотеза: сжатие интервалов и добавление времени на повторённые записи могут искажать динамические признаки; это не доказанная причина ошибок.

## Ожидалось (до прогона)
Change only the internal temporal resampling rule: a stored interval delta_video_frames advances the existing frozen dynamics by delta nominal30Hz steps, with linear stimulus interpolation; delta0 advances neither neural nor rich-retinal EMA state. Keep all genuine samples, same crops, weights, normalization, observed-center ridge target, saved lambda, readout and alpha0.25.

## Нельзя заявлять
Linear interpolation cannot recover the actual missing images, and more realistic elapsed time can be worse for D trained on the shortened timeline. Old normalization/lambda remain fixed and may not suit changed state distributions. The first search delay is a data/runtime design choice, not proven scientific failure. No claim of HOTA, cause or benefit from gap counts alone; repeatedtrain26 exploratory.

## Критерии приёмки (записаны до прогона)
- ⏳ Unit-clock features exactly equal original with tested mock backbone and real rich retina; mixed zero/gap node+edge state masking tested. First2groups of each C/D/R1 old path must reproduce saved centers<0.5px/scores<0.001. Every8326map/label/crop binding retained. First2Dgroups conservative actual virtual-step rate fits1800remaining, or preserve partial and stop. Subsequent26CPUreplays all old branch outcomes exactly equal previous mean decoder atalpha0.25; rows/hashes/intervals independently verified.
- ⏳ Dclock-baseline and Dclock-Dold net-beneficial mean-video frequencies both>=2pp; all6paired95%lower>0 including Cclock,R1clock,geometry and (Dclock-Dold)-(R1clock-R1old). Conditional center shifts are secondary and cannot rescue a failed assignment gate.

## Доказательства
- Новый временной узел, равные контроли, фиксированные параметры и бюджеты до откликов: `runs/z13_history_clock_20261025/protocol.json` · 

