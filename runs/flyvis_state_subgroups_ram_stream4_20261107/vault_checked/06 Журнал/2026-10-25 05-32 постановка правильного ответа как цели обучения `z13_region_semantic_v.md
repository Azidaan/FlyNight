---
tags:
- журнал
дата: '2026-10-25'
время: 05:32
---

# 2026-10-25 05:32 — постановка правильного ответа как цели обучения `z13_region_semantic_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Постановка.* Change the acceptable action target, not the likelihood family or input: include every own-correct, nonharmful action instead of only strictly beneficial changes. Include unchanged in the positive marginal whenever the original own assignment was correct. If neither an action nor baseline is correct, unchanged is the conservative expressible fallback. Same original88->32->1 region model, frozen68activity/68target/geometric8 inputs, seeds0/1/2,UID minibatches,1150updates,batch128,Adam0.001 and fit20 threshold grid. Utility controls are the12existing original equal-weight marginal likelihood models, all8326 outputs re-inferred exactly before launch; do not retrain.12newsemantic D/C/R1/geometry models. Failed context/product inputs are not carried forward.

*Мишень.* fit84:17859known queries,14646original own decisions correct,16737safe-own-correct actions vs1356strictly beneficial actions. Source runs/z13_region_supervision_audit_20261025/report.json. These are existing semantic outcome booleans, not new labels for unknown identity. Unknown queries remain excluded from gradients.

*Критерии.* Dsemantic-baseline and Dsemantic-Dutility means both>=2pp; all5paired-video95%lower>0: baseline,utilityD,semanticC,semanticR1,semanticgeometry. All26videos and3seeds. Same net-beneficial assignment metric; semantic label accuracy cannot rescue a failed primary. All12utility models exactly replayed; actual fit84 two-query CPU forward/backward finite. Mask/NULL/fallback gradients and utility special-case value+gradient equivalence tested. Same full caches/sources/UID batches, no completed training repeated. First full model2*seconds*11+60 fits original1800s; independent controller only own tree. Independently re-infer12new models/rows/calibration and paired intervals before final acceptance.

*Отличие.* Earlier semantic_sampling changed hard-row frequency while preserving center-useful labels. Balancing changed rare query weights; signed MSE changed regression objective; none made correct neutral decisions acceptable while retaining correct baseline. This trial changes the supervised decision target under the same marginal likelihood and architecture as its reused utility control.

*Возражение.* Correct neutral actions need not improve the stand; expanding the acceptable set may dilute rare useful corrections or leave an easy unchanged solution. Fixed response/mean-pooled representation may remain insufficient. One-row outcomes omit feedback; repeatedtrain26 is exploratory, not HOTA or proven causal diagnosis.

*Бюджет.* CPU1800с+20с, скрыто, save 2026-10-25T06:02:38.449337+03:00 / hard 2026-10-25T06:02:58.449337+03:00. НикакихGPU/ядра/dev/HOTA/75/загрузок. Источник runs/z13_region_semantic_20261025/protocol.json. Bhat et al.,DiMP,ICCV2019 https://arxiv.org/html/1904.07220v1 sections3.1,3.5: discriminative target/background labels and prediction error, not only occasions when an external tracker can be improved. This supports comparing supervision semantics; our acceptable-set loss is not the DiMP optimizer or an inherited published gain.
