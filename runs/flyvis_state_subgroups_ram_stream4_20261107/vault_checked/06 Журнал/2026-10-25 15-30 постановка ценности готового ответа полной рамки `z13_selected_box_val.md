---
tags:
- журнал
дата: '2026-10-25'
время: '15:30'
---

# 2026-10-25 15:30 — постановка ценности готового ответа полной рамки `z13_selected_box_value_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Internal final response decoder predicts signed value of its one frozen target-model-selected actual box. If predicted value>0 transfer the complete box with alpha1; else unchanged. The selected box is fixed before this decoder. NULL or unavailable exact pooled region stays unchanged. No manual unmatched/free guards.

*Признаки.* 150numbers:68raw pooled normalized-by-dataset activity,68componentwise products of that activity and observed target ridge coefficients,8original geometry and6observed original assignment context. Original per-vector LayerNorm is not part of this new analytic decoder. Geometry control zeros both68feature blocks, keeps same150coefficients. D/C/R1 same pipeline, own frozen selected region; all query UIDs retained.

*Обучение и отличие.* Only20fit videos fit the ridge value model against exact saved alpha1 beneficial-harmful outcomes of that selected box. 84other videos are not used because their full-box outcomes are absent. Standardize each feature using only supported known fit20 rows; sd floor1e-6; intercept=training mean; ridge lambda1 on mean squared error, fixed before run. No grid, calibration, seed selection, coverage quota or threshold fitting. Evaluate26 only after all4models saved. This is a narrower single-proposal action-value test, not resumption of the130video candidate-choice experiment.

*Критерии.* Dvalue-baseline mean>=2pp and all5paired95%video lower>0: baseline,prior Dbox alpha0.25 selected on20fit,D/C/R1 matched no-new-core controls (C,R1,geometry). Full criteria required; no HOTA from a partial contrast. Values are net one-step assignment effects, not center accuracy.

*Возражение.* 20videos may be too few for rare beneficial actions; zero-dominated labels may suppress all actions. The fixed target-conditioned proposal may already be wrong; value decoder cannot replace it. Geometry may explain any gain. No proper held-out confirmation:26is repeatedly reused and fit20 has informed development. This changes both representation/readout family and fitting target relative to the earlier neural categorical head, so it cannot isolate one cause.

*Источник.* SiamMOT https://arxiv.org/html/2105.11595 section3.1 learns target visibility jointly with its box output; supports learning refusal for the actual geometric response. Our signed ridge value and150inputs are a separate adaptation, not their model or demonstrated efficacy.

Pass -> separate frozen matched closed-loop parity/price/HOTA with all controls. Fail -> preserve this exact single-proposal value decoder; no lambda/threshold/feature/normalizer grid on26. Full all-region supervision remains untested, with4videos preserved.
