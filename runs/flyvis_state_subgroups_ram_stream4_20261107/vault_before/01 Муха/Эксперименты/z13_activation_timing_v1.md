---
aliases:
- 'З13: потолок раннего включения считывания до состояния потери'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- метка/SiamMOT
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 З13: потолок раннего включения считывания до состояния потери

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_activation_timing_v1`

Мишень раннего включения проверена; это диагностический потолок, не результат новой FlyVis.

## Вопрос
Does the Lost-only activation boundary exclude correctable first-stage association errors on still-Tracked objects?

## Гипотеза
A wrong match can keep a track in Tracked, so the current Lost-only readout is never called at that event. This is a structural hypothesis, not evidence that earlier FlyVis predictions will be accurate.

## Ожидалось (до прогона)
Consider keeping the internal FlyVis state and readout active on currently Tracked targets, before the first incorrect association, instead of starting only after Lost. This stage tests the available target for that change, not a new runtime policy. Replay all26 baseline videos exactly. At real annotated frames only, bind pool rows to the most recent observed annotated identity, freshness <=12 frames, with the same sparse-annotation gap exclusions as assignment_effect. Report all known rows by incoming Tracked/Lost state. For an initially incorrect row whose target has a high-score detection, translate only the predicted center to the annotated target-box center, preserving width/height and all other state. Recompute the whole unchanged Hungarian assignment with that single row replaced, then restore the original state before the real tracker update. For the primary gate, additionally require the target center inside the existing predicted eye crop and within0.6 lattice spacing of a retinal column, exactly the original field criterion. Out-of-field oracles are reported separately. Detection identities are assigned to all raw detections before the unchanged high-score filter, as in assignment_effect. No video, feature extraction, FlyVis inference, training or HOTA.

## Что сделано
2026-10-25

## Получено
runs/z13_activation_timing_20261025/report.json → counts: {"tracked_pool": 50338, "tracked_known": 35630, "tracked_correct": 32153, "tracked_high_present": 32238, "lost_pool": 30903, "lost_unknown": 19462, "tracked_unknown": 14708, "tracked_high_absent": 3392, "lost_known": 11441, "lost_correct": 9413, "lost_high_absent": 9304, "tracked_incorrect": 3477, "tracked_first_incorrect_final_tracked": 3360, "tracked_oracle_evaluated": 726, "tracked_oracle_beneficial": 110, "tracked_oracle_harmful": 1, "tracked_infield_beneficial": 102, "lost_incorrect": 2028, "lost_high_present": 2137, "lost_oracle_evaluated": 1890, "lost_oracle_beneficial": 442, "lost_oracle_harmful": 4, "lost_infield_beneficial": 285, "tracked_first_incorrect_final_lost": 109, "tracked_first_incorrect_final_removed": 8}; paired_added_tracked_center_oracle: {"mean_per_video": 0.002122305158337479, "ci95": [0.0016007953574131055, 0.0027166093323069355], "better": 24, "n": 26}. Все26 исходных выходов совпали точно; первичные события videos/*.json.

## Интерпретация
В отличие от прежних проверок потерянных треков, здесь отдельно измерены события ещё ведомых треков. Аннотированный центр используется только как диагностическое идеальное действие. Практический критерий запаса для этой правки центра не пройден. Это не доказывает бесполезность ранней памяти, иных размеров рамки или сопоставления идентичностей.

## Решение
Не запускать полное извлечение ранних эпизодов для этой правки центра. Сохранить полный разбор причин; нужна иная проверяемая мишень, а не подбор порога после этого итога.

## Нельзя заявлять
A perfect annotated center is unavailable at runtime, is not an upper bound for arbitrary box/identity modules, and can still fail IoU matching with unchanged dimensions. Sparse annotation and local identity freshness do not reveal every ID switch. Early activation may increase cost and erroneous interventions. A positive ceiling is necessary evidence for a prototype, not sufficient evidence of learnability or improved HOTA. Reused26train and sparse annotations. First-stage baseline-state counterfactual diagnostic, not HOTA, deployed policy, causal proof about FlyVis representations, or tracking benchmark. No dev/20val tuning/75/downloads/GPU.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (4a28ce3ba1a8f2a414c06d14669dab6ce57d6207) — Сохранён проверенный итог диагностики момента включения.
  > Короткая проверка мишени для иной границы работы внутреннего состояния; не новый обученный вариант.

## Критерии приёмки (записаны до прогона)
- ✅ All26 raw baseline outputs must match saved float32 rows exactly. Full pool costs unchanged after every shadow action. Identity binding uses past observations only; evaluation truth is never passed to tracker.update. End state after the real full update is reported separately, so first-stage miss is not presented as final loss. Sources and original thresholds unchanged. First two fixed videos measure cost; 2*mean_seconds*remaining24+60 must fit remaining900s, otherwise preserve partial results without a verdict.  
  *runs/z13_activation_timing_20261025/finish_recorded.json*
- ❌ For each video, count individually beneficial IN-FIELD center-oracle actions on incoming Tracked rows divided by ALL known pool rows (Tracked+Lost), using the same denominator for early and Lost-only ceilings. Mean added opportunity >=0.02 and paired-video95%lower>0 are both required before preparing an all-active FlyVis extraction. This is a conservative practical gate for this center-only action, not a universal criterion for timing mechanisms. Interacting oracle actions are not combined or claimed jointly attainable.  
  *runs/z13_activation_timing_20261025/report.json*

## Доказательства
- Действие, критерии и цена записаны до расчёта: `runs/z13_activation_timing_20261025/protocol.json` · 
- Полный диагностический потолок, исходные назначения и одинаковый знаменатель: `runs/z13_activation_timing_20261025/report.json` · 

## Код
- `scripts/z13_activation_timing_20261025.py`
- `scripts/z13_activation_timing_console.ps1`
- `tests/test_z13_activation_timing.py`

