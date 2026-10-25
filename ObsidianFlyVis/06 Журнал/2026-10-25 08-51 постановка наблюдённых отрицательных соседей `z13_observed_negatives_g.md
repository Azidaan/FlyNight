---
tags:
- журнал
дата: '2026-10-25'
время: 08:51
---

# 2026-10-25 08:51 — постановка наблюдённых отрицательных соседей `z13_observed_negatives_gate_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Действие.* Candidate target-specific ridge update: weight4on separable same-class detector-negative columns in the last OBSERVED exemplar, weight1elsewhere; same Gaussian target/core/readout and matched unweightedC/D/R1 controls. This queue permits only the cheap availability prerequisite, not extraction or fitting.

*Критерий до расчёта.* Exact130video/37296query/26600known bindings, nonempty observed mask, every negative uses only last-observed frame/crop, same tracker class, conf>=0.25, IoU<=0.1 with the lattice-reconstructed observed box, >=1column outside its1.25expanded guard, supported center by the unchanged lattice rule. Admission requires lower paired-video95%CI of known evaluate26 queries with such an observed negative AND a reachable beneficial region action>0.02; upper<0.02 insufficient, otherwise inconclusive. This is only causal-input availability and oracle exposure, not an achieved benefit. No relaxing this criterion after result.

*Возражение.* A nearby detector box may be a duplicate or a wrong identity, while conservative overlap exclusion can discard the hard overlapping distractors. Observable negatives may be common but uninformative; weighting can suppress useful target features. A full prototype must use identical causal masks forC/D/R1, oldmean parity, fixed before-evaluation settings and actual assignment controls; no presumed cause or HOTA.

*Источник.* Zhu et al.,Distractor-aware Siamese Networks for Visual Object Tracking,ECCV2018,section3.2semantic negative pairs and3.3incremental learning https://arxiv.org/html/1808.06048 . Our observed-mask weighting is a limited ridge-target adaptation, not DaSiamRPN or its published gains. Existing Gaussian targets already include ordinary background; this tests separately identified same-class observed neighbors.

*Бюджет.* CPU180с+20с, безGPU/обучения/dev/HOTA/75/загрузок, скрыто, save 2026-10-25T08:54:47.627146+03:00 / hard 2026-10-25T08:55:07.627146+03:00. Источник runs/z13_observed_negatives_gate_20261025/protocol.json.
