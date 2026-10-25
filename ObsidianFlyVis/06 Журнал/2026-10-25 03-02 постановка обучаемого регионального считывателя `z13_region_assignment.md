---
tags:
- журнал
- журнал/муха
дата: '2026-10-25'
время: 03:02
---

# 2026-10-25 03:02 — постановка обучаемого регионального считывателя `z13_region_assignment_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Replace the internal FlyVis target readout: learned joint categorical region choice and explicit unmatched action, using frozen target-model coefficients and raw retinal activity pooled at real high-score same-class detection boxes. Preserve the surrounding detector, tracker and MASA. No learned core parameters in this trial.

*Критерии.* Average per-video net assignment effect across all3seeds, then paired-video95%CI. D-baseline mean>=0.02 and lower>0; D-C,D-R1,D-geometry all lower>0. Report each seed and all counts, unknowns and chosen thresholds. Passing is a gate for separately budgeted closed-loop HOTA, never automatic HOTA or final novelty.

*Равные контроли.* Identical action sets, labels, architecture, seeds, minibatch UIDs and optimizer forD,C,R1,geometry. Geometry uses the same D tensors but zeros both activity and target-model coefficients before embedding. C is the standard fly; D remains the retained trained core. R1 is the rich retinal-input control. New full readout vs no action, thenD vs each matched control.

*Отличие.* Unlike old three-peak rankers/filter descriptors, actions are real detector regions with actual assignment-effect labels, whose separate oracle now passed. Unlike fixed region_mean pooling, raw68 activity and observed coefficients are learned jointly with candidate context and unmatched output. Original head architecture is reused to avoid attributing differences to an untested larger network.

*Возражение.* The oracle used truth and is equally available to non-fly representations. Low-resolution activity may not discriminate real instances; geometry may explain all utility. Some targets have no high-score detection, so this action cannot help them. One-step training omits feedback and later stages. The protocol changes the complete readout output/learning target, not an isolated causal test of a single scalar. StandardC vsD core histories differ by design. Sparse labels and repeatedly reused train26 limit generalization.

*Бюджет.* labels900s;features5400s;training1800s, each+20s within global window. CPU first2fit replays:2*mean*102+60 must fit remaining labels budget. GPU first2D groups:2*seconds/episode-time*remaining D+C episode-time+300 must fit remaining feature budget. CPU first completed model:2*elapsed*11+60 must fit remaining training budget. Failed price gate preserves completed work and stops later phases.

Источник: runs/z13_region_assignment_20261025/protocol.json. SiamMOT https://arxiv.org/html/2105.11595v1 sections3,5 motivates region-conditioned matching; KeepTrack https://arxiv.org/html/2103.16556v2 motivates candidate competition and explicit unmatched. Neither supplies our action loss or expected gain.
