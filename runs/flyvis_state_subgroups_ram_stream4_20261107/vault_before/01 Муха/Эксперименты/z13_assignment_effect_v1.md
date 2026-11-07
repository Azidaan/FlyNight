---
aliases:
- 'З13: последствия кандидатов считывателя для полного сопоставления'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: последствия кандидатов считывателя для полного сопоставления

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_assignment_effect_v1`

Точные однокадровые последствия проверены;оракульный запас не прошёл критерий перехода.

## Вопрос
Есть ли доступные кандидатам D полезные исправления фактического сопоставления,и совпадают ли они с прежними метками центра?

## Гипотеза
Метка улучшения центра может не совпадать с пользой сопоставлению;иной целевой сигнал может лучше обучать отказ.

## Ожидалось (до прогона)
Prepare supervision for a replacement FlyVis readout trained on the association consequence of its center action,rather than distance-to-GT-center utility. CPU exact baseline replay only;no new model trained here. For every original cached candidate on real annotated query frames,temporarily replace only the lost-track mean center,compute its first-association cost row,and solve the complete unchanged Hungarian assignment with that single row replaced. Restore exact mean before real tracker update. Baseline trajectory remains unchanged. Other tracks participate. Store full assignment alternatives,not independent nearest-detection guesses.

## Что сделано
2026-10-24

## Получено
runs/z13_assignment_effect_20261024/recovery1/report.json → oracle_beneficial_fraction/candidate_counts: {"oracle": {"D": {"mean_per_video": 0.016333290216897698, "ci95": [0.011763925813050998, 0.021382755368597604], "better": 22, "n": 26}, "C": {"mean_per_video": 0.016481833838614117, "ci95": [0.012459568550211287, 0.020952017849391054], "better": 23, "n": 26}, "R1": {"mean_per_video": 0.013639793430509573, "ci95": [0.009716834465446862, 0.018430945325481524], "better": 23, "n": 26}}, "counts": {"D": {"known": 5822, "candidates": 17466, "beneficial": 165, "harmful": 297, "center_useful_candidates": 846, "center_useful_beneficial": 37, "center_useful_harmful": 17, "center_useful_neutral": 792}, "C": {"known": 5822, "candidates": 17466, "beneficial": 175, "harmful": 298, "center_useful_candidates": 840, "center_useful_beneficial": 50, "center_useful_harmful": 24, "center_useful_neutral": 766}, "R1": {"known": 5822, "candidates": 17466, "beneficial": 176, "harmful": 372, "center_useful_candidates": 411, "center_useful_beneficial": 21, "center_useful_harmful": 16, "center_useful_neutral": 374}}}

## Интерпретация
Метки относятся к полному первому сопоставлению на неизменённой исходной траектории;не учитывают последующие этапы и обратную связь. Это не HOTA и не реализованный прирост.

## Решение
Не обучать эту версию по непройденному запасу;сохранить все данные и выбрать следующий узел.

## Нельзя заявлять
One-step baseline-state counterfactuals omit feedback,later low-score/unconfirmed association and long-term identity. Sparse annotations and last observed identity limit semantic labels. Higher oracle reserve need not be learnable. This is an association-aligned readout target hypothesis,not DeepMOT reproduction or a change to YOLO/ByteTrack/MASA.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (e811778e67c8b32a403f3e86f21646eb6b09217e) — Полный диагностический набор последствий.
  > Сначала точный контроль исходного трекера и цена,потом полный диагностический набор;без обучения.

## Критерии приёмки (записаны до прогона)
- ✅ All130videos replay float32 raw outputs exactly against saved baseline;every cached label prediction matches replay mean castfloat32 exactly;all rows bound byvideo,track,frame,identity. First2fixed fit20videos measure walltime;2*mean_seconds*remaining128+180must fit actual remaining original budget,else retain2jobs and stop. Completed videos not repeated.  
  *runs/z13_assignment_effect_20261024/recovery1/report.json;130videojobs;price.json*
- ❌ On all26reused training evaluation videos,mean fraction of known eligible queries with at least one D candidate producing net beneficial first-stage assignment >=0.02 and paired-video95%lower>0 against0. If upper<0.02 this candidate set has insufficient measured reserve;otherwise inconclusive. C/R1 reserve and geometry-preserving no-action baseline reported. Old center-useful flags versus benefit/harm table reported. Oracle is diagnostic only;it never becomes a runtime policy.  
  *runs/z13_assignment_effect_20261024/recovery1/report.json*

## Доказательства
- Условия реального сопоставления,критерии и цена до расчёта: `runs/z13_assignment_effect_20261024/protocol.json` · 

## Код
- `scripts/z13_assignment_effect_20261024.py`

