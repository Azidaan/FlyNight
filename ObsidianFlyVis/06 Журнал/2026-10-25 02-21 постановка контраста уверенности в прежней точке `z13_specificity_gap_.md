---
tags:
- журнал
дата: '2026-10-25'
время: 02:21
---

# 2026-10-25 02:21 — постановка контраста уверенности в прежней точке `z13_specificity_gap_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Replace the internal confidence formula, keeping the canonical proposed center unchanged: normalized own response at its argmax minus mean normalized foreign-filter response at that SAME column. No learned coefficients/grid or threshold optimization.

*Критерии.* Frozen topceil(1%*4844)=49 per arm/score with ascending label tie-break; compare gap vs original height on the IDENTICAL cohort. Per-video net=(beneficial-harmful)/all known eligible queries in that video. Require Dgap-baseline0, Dgap-Dheight, Dgap-Cgap,Dgap-R1gap all lower95%paired-videoCI>0. A pass is only a target for a separate deployable full-cohort calibration; never automatic HOTA.

*Граница вывода.* The strict cohort uses known observed identity to construct a diagnostic intervention, and is not an inference-ready admission rule. Counterfactual maps may amplify background; three donors may be unrepresentative. Selection is retrospective fixed-coverage, not a threshold frozen on independent calibration data. Reused train26; single-step canonical states omit feedback. A positive outcome is not a calibrated refusal or whole-tracker improvement.

CPU300с+20с, независимый контроллер, сохранённые карты. Другая формула, не повтор удаления SD или фиксированного косинуса наблюдавшегося фона.
