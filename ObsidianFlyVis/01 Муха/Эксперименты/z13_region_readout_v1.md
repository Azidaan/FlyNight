---
aliases:
- 'З13: считывание карты по областям детекций прямо в сопоставление'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- метка/SiamMOT
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟡 З13: считывание карты по областям детекций прямо в сопоставление

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_region_readout_v1`

Проверен конкретный региональный декодер; основной критерий не пройден.

## Вопрос
Помогает ли внутреннее чтение отклика по реальным рамкам сверх того же интерфейса без мухи, при неизменном центре трекера?

## Гипотеза
Геометрическое улучшение центра не гарантирует верного назначения; иной выход считывателя может использовать уже имеющийся отклик без разрушения рамки. Это проверяемая гипотеза, не установленное исправление.

## Ожидалось (до прогона)
Replace the internal readout output: use the existing target-response field to score actual detection regions, rather than overwrite the predicted box center with one peak. Test this zero-training prototype on exact one-step assignments before any closed-loop run. All high-score detections are projected to the unchanged query retinal crop. A region is supported only if its center is in the existing hex field and its rectangle contains retinal columns. Its logit is the mean standardized target response inside the rectangle; NULL logit0; joint softmax over supported regions+NULL. New fused cost=(1-alpha)*old_cost+alpha*(1-det_conf*region_probability); unsupported costs unchanged. Exact alpha0 baseline. Same maps, support, calibration grid and association rules for C/D/R1. Geometry control uses only a fixed negative squared-distance field around the original prediction, replacing the response field in the identical decoder.

## Что сделано
2026-10-25

## Получено
runs/z13_region_readout_20261025/recovery1/report.json → choices: {"D": 0.25, "C": 0.25, "R1": 0.25, "geometry": 0.25}; counts: {"D": {"rows": 8326, "known": 5822, "beneficial": 12, "harmful": 14, "unknown": 2504}, "C": {"rows": 8326, "known": 5822, "beneficial": 13, "harmful": 15, "unknown": 2504}, "R1": {"rows": 8326, "known": 5822, "beneficial": 13, "harmful": 16, "unknown": 2504}, "geometry": {"rows": 8326, "known": 5822, "beneficial": 16, "harmful": 12, "unknown": 2504}}; paired: {"D-baseline": {"mean_per_video": -0.0010206812089205884, "ci95": [-0.002998706694060538, 0.0007632546494682034], "better": 5, "n": 26}, "D-C": {"mean_per_video": -4.14249812873795e-05, "ci95": [-0.0005044908622975538, 0.00041392988105272356], "better": 2, "n": 26}, "D-R1": {"mean_per_video": -0.0001614467876489632, "ci95": [-0.001693409346563374, 0.001227882530524721], "better": 6, "n": 26}, "D-geometry": {"mean_per_video": -0.0008078023784979632, "ci95": [-0.001976136901284644, 0.00014970531034850065], "better": 2, "n": 26}}. Полные46 исходных выходов повторены точно; reproduction.json и videos/*.json проверены.

## Интерпретация
Здесь менялся выход внутреннего считывания: оценка реальных областей вместо переноса рамки на один пик. Эта конкретная версия не подтверждает преимущество D сверх всех обязательных контролей и практического порога. Неудача не устанавливает невозможность другого регионального считывания.

## Решение
Не внедрять этот декодер в стенд и не подбирать alpha по26видео. Сохранить реализацию и полную сетку калибровки, включая нулевой вариант.

## Нельзя заявлять
Cost calibration and proposal geometry may explain the entire effect without FlyVis. Region means may dilute the target, share response between overlapping boxes, or merely reject all detections. The no-FlyVis geometry/R1 controls are mandatory. Individual shadow actions do not capture simultaneous interactions, feedback or later matching stages, so a positive result is only a gate for separate closed-loop verification. Repeated26train, sparse original labels, baseline-state one-step diagnostic. Not HOTA, safe refusal, final-val performance or proven intrinsic core improvement. No GPU/new video/features/core training/dev/20val tuning/75/downloads.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (f9f16b44f8dcb975630bf9e55623d67486fa1a8b) — Сохранён проверенный итог конкретного регионального декодера.
  > Исполняемый внутренний декодер и одинаковые контроли зарегистрированы до результата.

## Критерии приёмки (записаны до прогона)
- ✅ All46 baseline outputs float32-exact. Every query bound to the saved tracker prediction/identity and its original label index. Before use, last-expert map top1 center must match canonical candidate within0.5px and score within0.001 on all fit/evaluate rows. Alpha0 whole assignments exactly equal baseline. Full tracker states/costs unchanged after every shadow evaluation. All source hashes unchanged. First2fit videos:2*mean_seconds*remaining44+120 must fit remaining1200s, else save partial and stop.  
  *runs/z13_region_readout_20261025/recovery1/finish_recorded.json*
- ❌ On all26 reused training evaluation videos, D net action frequency versus unchanged baseline must have mean>=0.02 and lower paired-video95%CI>0. D must also beat geometry-only, R1 and C region decoders with all lower95%CI>0. All conditions required; other comparisons cannot rescue. Unknown outcomes reported separately and never assigned invented labels.  
  *runs/z13_region_readout_20261025/recovery1/report.json*

## Доказательства
- Механизм, контроли и критерии до расчёта: `runs/z13_region_readout_20261025/protocol.json` · 
- Все назначения, одинаковые контроли и парные интервалы: `runs/z13_region_readout_20261025/recovery1/report.json` · 

## Код
- `scripts/z13_region_readout_20261025.py`
- `tracking/z13_region_readout.py`
- `tests/test_z13_region_readout.py`
- `scripts/z13_region_readout_console.ps1`

