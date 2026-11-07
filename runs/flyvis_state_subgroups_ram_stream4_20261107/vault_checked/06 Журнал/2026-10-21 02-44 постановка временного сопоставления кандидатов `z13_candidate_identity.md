---
tags:
- журнал
дата: '2026-10-21'
время: 02:44
---

# 2026-10-21 02:44 — постановка временного сопоставления кандидатов `z13_candidate_identity_v1`

↑ [[06 Журнал/Дни/2026-10-21|2026-10-21]]

*Постановка.* Replace tallest-peak choice by a target identity carried through one-to-one candidate assignments. Fixed core, same3 candidates and all available episode frames. Missing assignment returns no correction.

Same20 train only. Appearance alpha=[0.25,0.5,0.75,1.0], missing_cost=[0.25,0.5,0.75]; geometry control alpha=0, same3 costs. Choose highest per-video mean useful fraction on fit20; deterministic grid order breaks ties. No parameter changes after train26.

Критерии до исходов: C association-C top1: point gain>=0.02 and paired-video95% lower>0. C association-C geometry and C association-R1 association: both lower>0. All three required; D secondary, not a rescue.

Надёжность отказа и HOTA не следуют из этого пилота. Функциональный CPU-этап до1800с, без нового ядра и без настройки на dev.
