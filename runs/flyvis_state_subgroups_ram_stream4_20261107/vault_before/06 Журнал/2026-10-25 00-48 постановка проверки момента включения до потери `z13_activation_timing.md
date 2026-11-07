---
tags:
- журнал
- журнал/муха
дата: '2026-10-25'
время: 00:48
---

# 2026-10-25 00:48 — постановка проверки момента включения до потери `z13_activation_timing_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Consider keeping the internal FlyVis state and readout active on currently Tracked targets, before the first incorrect association, instead of starting only after Lost. This stage tests the available target for that change, not a new runtime policy.

*Вопрос.* Does the Lost-only activation boundary exclude correctable first-stage association errors on still-Tracked objects?

*Метод и критерии до исходов.* Replay all26 baseline videos exactly. At real annotated frames only, bind pool rows to the most recent observed annotated identity, freshness <=12 frames, with the same sparse-annotation gap exclusions as assignment_effect. Report all known rows by incoming Tracked/Lost state. For an initially incorrect row whose target has a high-score detection, translate only the predicted center to the annotated target-box center, preserving width/height and all other state. Recompute the whole unchanged Hungarian assignment with that single row replaced, then restore the original state before the real tracker update. For the primary gate, additionally require the target center inside the existing predicted eye crop and within0.6 lattice spacing of a retinal column, exactly the original field criterion. Out-of-field oracles are reported separately. Detection identities are assigned to all raw detections before the unchanged high-score filter, as in assignment_effect. No video, feature extraction, FlyVis inference, training or HOTA. All26 raw baseline outputs must match saved float32 rows exactly. Full pool costs unchanged after every shadow action. Identity binding uses past observations only; evaluation truth is never passed to tracker.update. End state after the real full update is reported separately, so first-stage miss is not presented as final loss. Sources and original thresholds unchanged. First two fixed videos measure cost; 2*mean_seconds*remaining24+60 must fit remaining900s, otherwise preserve partial results without a verdict. For each video, count individually beneficial IN-FIELD center-oracle actions on incoming Tracked rows divided by ALL known pool rows (Tracked+Lost), using the same denominator for early and Lost-only ceilings. Mean added opportunity >=0.02 and paired-video95%lower>0 are both required before preparing an all-active FlyVis extraction. This is a conservative practical gate for this center-only action, not a universal criterion for timing mechanisms. Interacting oracle actions are not combined or claimed jointly attainable.

*Отличие от испытанного.* Earlier memory/temporal-candidate variants change observations inside already-Lost episodes; scalar-confidence and observed-contrast variants reorder the same already-Lost top1 proposals. This diagnostic asks about events entirely excluded from those episodes. It does not repeat their fitting or relax their failed criteria.

*Контраргумент.* A perfect annotated center is unavailable at runtime, is not an upper bound for arbitrary box/identity modules, and can still fail IoU matching with unchanged dimensions. Sparse annotation and local identity freshness do not reveal every ID switch. Early activation may increase cost and erroneous interventions. A positive ceiling is necessary evidence for a prototype, not sufficient evidence of learnability or improved HOTA.

SiamMOT: https://openaccess.thecvf.com/content/CVPR2021/html/Shuai_SiamMOT_Siamese_Multi-Object_Tracking_CVPR_2021_paper.html. Переносится мотивация проверки, не численный результат. CPU до900с, независимый контроллер и отдельная консоль. Старые критерии не меняются.
