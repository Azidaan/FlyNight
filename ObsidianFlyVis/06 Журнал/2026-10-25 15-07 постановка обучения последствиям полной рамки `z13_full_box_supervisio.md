---
tags:
- журнал
дата: '2026-10-25'
время: '15:07'
---

# 2026-10-25 15:07 — постановка обучения последствиям полной рамки `z13_full_box_supervision_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Replace the internal regional readout action semantics and corresponding supervision: selected real region now transfers complete XYWH and uses unchanged native get_dists, instead of a categorical cost row. Explicit unmatched still sets this row costs to1; fixed unchanged logit still means no action. No row or target identity is supplied at runtime.

*Отличие от испытанного.* Earlier signed/context heads learned outcomes of categorical replacement costs. The frozen-map full-box prototype did not learn safety for the actual geometric action. This trains identical old readout architectures/features against new full-box outcomes. It is a coupled output-interface/target change, not proof that one scalar loss caused previous failures. Regional support is the exact prior trainable support, including nearest-column check, not the wider fixed-map prototype support.

*Фиксировано.* Reuse all D/C/R1 raw68 regional activity, observed target coefficients, original assignment context6, eight geometry inputs and query UID order. Same24heads (4representations x context/blank x3seeds),1150steps,batch128,Adam0.001,signed good-harm loss, fixed fit20 threshold grid. Train84; calibrate20; evaluate26. Only outcome labels and applied candidate-box action change together. No new feature/core computation or GPU.

*Критерии до прогона.* Preregister D_context as primary, no choosing blank after evaluation: net baseline effect mean>=2pp and all6paired95%video lower>0 versus baseline,D_blank,C_context,R1_context,geometry_context,previous Dcontext categorical action. Report all24models and frozen fit20 thresholds, including zero-action and unfavorable outcomes. All130 baseline raw outputs exactly reproduce original. All37296query bindings/region orders match original labels and saved context; same observed features reused. Label arrays contain actual full assignment outcomes for each full-box/null action, not IoU or identity surrogate. Complete means restored; unchanged action exact. All input hashes; every model/hash/row/interval independently audited. First2label replays2*mean*128+60 must fit1200s. Firsttraining model2*time*23+60 must fit1800s. No completed video/model repeated.

*Возражение.* The old representations may not identify absent targets even with correct action labels. Severe imbalance may yield no action; the same geometry controls may suffice. Action reserve does not imply learnability. Both20fit and26evaluation repeatedly used, so this is exploratory, not independent confirmation. No manual guards or threshold edits after evaluation.

*Источник.* SiamMOT https://arxiv.org/html/2105.11595 sections3.1-3.3 equations1,3,6 learns visibility and full-box output before association. Supports aligning a learned output with its geometric action, not our signed loss or an expected improvement. This is an adaptation inside FlyVis readout, not reproduction of SiamMOT.

CPU1200с метки → CPU1800с24модели, каждый+20с, независимые скрытые контроллеры. Pass -> separately preregister matched runtime and closed-loop HOTA with native/C/R1/no-FlyVis controls. Fail -> preserve this exact action-supervision implementation and stop this categorical-to-box family; no new loss/guard/threshold sweep on26. A different internal mechanism needs its own measured rationale.
