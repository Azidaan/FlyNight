---
aliases:
- 'Выход модели цели FlyVis: реальная рамка вместо одного центра'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Выход модели цели FlyVis: реальная рамка вместо одного центра

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_box_proposal_v1`

Выход полной рамки: основной критерий не пройден.

## Вопрос
Даёт ли полная геометрия предложенной области пользу сверх того же центра и равного считывания без мухи?

## Гипотеза
Подтверждён геометрический запас действия полной рамки. Рабочая гипотеза: обученная на прежнем образце модель цели способна выбрать область достаточно хорошо, чтобы этот запас использовать.

## Ожидалось (до прогона)
Change the internal learned target-model output from a peak center to an observed detection XYWH proposal, then use the unchanged tracker distance. The target-conditioned ridge map is learned from the same observed exemplar and reused frozen. No offline new core/readout training; a working output-decoder prototype. Read unchanged final-query C/D/R1 maps. Project every actual high same-class detection into original eye crop. Retain center-in-crop regions containing at least one observed hex column. Region logit is the existing centered/scaled map mean inside that rectangle, NULLlogit0. Choose argmax including NULL, with NULL winning ties. Chosen real detection yields target XYWH; interpolate original mean toward it using alpha. Center-only twin changes only XY; box twin changes XYWH. Same selection/support/map/calibration/other state for each twin. Standard get_dists and full assignment are unchanged.

## Что сделано
2026-10-25

## Получено
runs/z13_box_proposal_20261025/report.json → paired/counts: {'paired': {'Dbox-baseline': {'mean_per_video': -0.0015279390120082625, 'ci95': [-0.005023995696353085, 0.0017842763096149424], 'better': 7, 'n': 26}, 'Dbox-Dold': {'mean_per_video': -0.0015279390120082625, 'ci95': [-0.005023995696353085, 0.0017842763096149424], 'better': 7, 'n': 26}, 'Dbox-D_center': {'mean_per_video': -0.0015850459555776454, 'ci95': [-0.003783113716658327, 0.0003126343040499594], 'better': 3, 'n': 26}, 'Dbox-C_box': {'mean_per_video': 0.00018087655300193847, 'ci95': [-0.00033597850989691247, 0.0007518206034750168], 'better': 3, 'n': 26}, 'Dbox-R1_box': {'mean_per_video': 0.0009725637711002501, 'ci95': [-0.0020068413945629133, 0.004165618791269389], 'better': 10, 'n': 26}, 'Dbox-geometry_box': {'mean_per_video': 4.953926311805501e-05, 'ci95': [-0.0004703610803158189, 0.0005658186867705043], 'better': 3, 'n': 26}, 'box_effect_D-C': {'mean_per_video': 0.00033418701181859065, 'ci95': [-2.6688053476566264e-19, 0.0008932953011900381], 'better': 2, 'n': 26}, 'box_effect_D-R1': {'mean_per_video': -0.0005714244960431026, 'ci95': [-0.0018391573835774809, 0.000572832226926201], 'better': 3, 'n': 26}}, 'counts': {'D_center': {'known': 5822, 'unknown': 2504, 'beneficial': 17, 'harmful': 10}, 'D_box': {'known': 5822, 'unknown': 2504, 'beneficial': 22, 'harmful': 24}, 'C_center': {'known': 5822, 'unknown': 2504, 'beneficial': 20, 'harmful': 11}, 'C_box': {'known': 5822, 'unknown': 2504, 'beneficial': 22, 'harmful': 25}, 'R1_center': {'known': 5822, 'unknown': 2504, 'beneficial': 14, 'harmful': 23}, 'R1_box': {'known': 5822, 'unknown': 2504, 'beneficial': 18, 'harmful': 31}, 'geometry_center': {'known': 5822, 'unknown': 2504, 'beneficial': 18, 'harmful': 10}, 'geometry_box': {'known': 5822, 'unknown': 2504, 'beneficial': 23, 'harmful': 26}, 'Dold': {'known': 5822, 'unknown': 2504, 'beneficial': 0, 'harmful': 0}}}

## Интерпретация
Проверен рабочий выбор по отклику обученной модели цели без разметки на входе. Центр и полная рамка одной выбранной области — парные варианты. Это первые назначения на повторно используемом train26, не HOTA и не предел полезности FlyVis.

## Решение
Pass -> separately preregister runtime parity/price and matched closed-loop HOTA with no-FlyVis/C/R1 controls. Fail -> keep this exact readout negative; no alpha/NULL/pooling sweeps on26. Record what succeeded diagnostically versus what worked in the actual prototype, and select the next genuinely different node with evidence.

## Нельзя заявлять
Selecting real detector boxes can merely duplicate existing detector geometry and drift to neighbors. Region means/normalized confidence already failed in other interfaces, so this deliberately minimal intervention may still fail. Map features may lack identity/box-quality evidence. Equal eye and geometry-only controls must isolate FlyVis; even positive train26 shadow results require separate frozen closed-loop protocol. CPU1200s+20, hidden independent controller. No GPU/new features/core training/download/dev/HOTA/75. Same reused train26, sparse labels, baseline-state one-step decisions only. Core trained on different objective; no assumption that named paper solves our task. No novelty/absolute stand benefit from oracle or conditional center result.

## Критерии приёмки (записаны до прогона)
- ✅ All46raw baseline outputs bit-exact, all8326evaluate and all fit canonical C/D/R1 map top1 centers<0.5px/scores<0.001, unchanged crops/pred/frame/label binding. Dold semantic outcomes exactly equal old replay for its selected top1, or neutral if refused. Complete mean restored, alpha0 no-action exact. All source hashes. First2fit conservative2*mean*44+90 must fit original1200s; completed videos not repeated.  
  *runs/z13_box_proposal_20261025/independent_audit.json*
- ❌ Dbox-baseline mean>=2pp and every paired95%video lower>0: Dbox versus baseline, saved Dold, same Dcenter, Cbox,R1box,geometrybox; plus differential box effect D-C and D-R1. All required. Partial positive contrast preserved but cannot promote failed full criterion to HOTA.  
  *runs/z13_box_proposal_20261025/report.json*

## Доказательства
- Одна рабочая замена выхода, равный центр/рамка и критерии до исполнения: `runs/z13_box_proposal_20261025/protocol.json` · 

