---
tags:
- журнал
дата: '2026-10-21'
время: 02:55
---

# 2026-10-21 02:55 — постановка обучаемого выбора кандидатов `z13_candidate_ranker_v1`

↑ [[06 Журнал/Дни/2026-10-21|2026-10-21]]

*Постановка.* Replace fixed cosine/geometric candidate matching with a small learned choice over the same3 current peaks, observed seed activity and current tracker prediction. This changes the selected center, unlike prior confidence-only heads.

All known fit20 rows only; BCE labels each candidate useful iff known-positive, err<=0.25*target-scale and err<current-prediction-error. Known absence rows all negative; unknown rows excluded. Class weight=min(20,negative/positive), training only. Fixed50 epochs, AdamW lr0.001,weight_decay0.01,batch128. No checkpoint selection. Fixed seeds0,1,2; mean logits ensemble, never select best seed. Always choose one candidate; no calibrated refusal is claimed.

Критерии до исходов: C ensemble-C top1: mean gain>=0.02 and paired-video95%lower>0. C ensemble-C geometry_ablation, C ensemble-C nearest_Kalman and C ensemble-R1 ensemble: all lower>0. All3 individual C seed gains against top1 must be nonnegative points. All conditions required.

Меняется сам выбранный центр внутри считывания; это отличается от прежних внешних оценок качества фиксированного центра. Причина неудачи фиксированного сопоставления не установлена. Ядро не обучается; CPU до3600с, сохранение07:35, конец дерева до07:39:30.
