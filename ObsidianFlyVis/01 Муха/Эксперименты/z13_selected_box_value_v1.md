---
aliases:
- Внутренний отказ по ценности готовой полной рамки FlyVis
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Внутренний отказ по ценности готовой полной рамки FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_selected_box_value_v1`

Ценность готовой полной рамки: критерий не пройден.

## Вопрос
Можно ли использовать готовый ответ полной рамки, обучив отказ его фактическим последствиям?

## Гипотеза
Рабочая гипотеза: готовые метки полного ответа позволяют изучить пользу против исходного решения без дорогостоящей разметки всех альтернатив.

## Ожидалось (до прогона)
Internal final response decoder predicts signed value of its one frozen target-model-selected actual box. If predicted value>0 transfer the complete box with alpha1; else unchanged. The selected box is fixed before this decoder. NULL or unavailable exact pooled region stays unchanged. No manual unmatched/free guards. 150numbers:68raw pooled normalized-by-dataset activity,68componentwise products of that activity and observed target ridge coefficients,8original geometry and6observed original assignment context. Original per-vector LayerNorm is not part of this new analytic decoder. Geometry control zeros both68feature blocks, keeps same150coefficients. D/C/R1 same pipeline, own frozen selected region; all query UIDs retained.

## Что сделано
2026-10-25

## Получено
runs/z13_selected_box_value_20261025/report.json → paired/counts: {'paired': {'Dvalue-baseline': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}, 'Dvalue-Dprior': {'mean_per_video': 0.0021500252590945096, 'ci95': [-0.0016259243890012205, 0.0064890309434590026], 'better': 9, 'n': 26}, 'Dvalue-C': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}, 'Dvalue-R1': {'mean_per_video': 0.0005940990118390587, 'ci95': [-0.00020024782740678485, 0.0018104264734094926], 'better': 3, 'n': 26}, 'Dvalue-geometry': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}}, 'counts': {'D': {'rows': 8326, 'known': 5822, 'supported': 3820, 'applied': 49, 'beneficial': 2, 'harmful': 0, 'unknown': 2504}, 'C': {'rows': 8326, 'known': 5822, 'supported': 3890, 'applied': 49, 'beneficial': 0, 'harmful': 0, 'unknown': 2504}, 'R1': {'rows': 8326, 'known': 5822, 'supported': 3448, 'applied': 29, 'beneficial': 1, 'harmful': 1, 'unknown': 2504}, 'geometry': {'rows': 8326, 'known': 5822, 'supported': 3865, 'applied': 25, 'beneficial': 0, 'harmful': 0, 'unknown': 2504}}}

## Интерпретация
Проверен более узкий внутренний отказ для одного готового ответа. Все четыре модели и строки независимо пересчитаны из старых меток фактического действия. Это не завершение выбора среди всех рамок и не HOTA;26видео повторно используются.

## Решение
Pass -> separate frozen matched closed-loop parity/price/HOTA with all controls. Fail -> preserve this exact single-proposal value decoder; no lambda/threshold/feature/normalizer grid on26. Full all-region supervision remains untested, with4videos preserved. Основной критерий: False

## Нельзя заявлять
20videos may be too few for rare beneficial actions; zero-dominated labels may suppress all actions. The fixed target-conditioned proposal may already be wrong; value decoder cannot replace it. Geometry may explain any gain. No proper held-out confirmation:26is repeatedly reused and fit20 has informed development. This changes both representation/readout family and fitting target relative to the earlier neural categorical head, so it cannot isolate one cause.

## Критерии приёмки (записаны до прогона)
- ✅ Exact UID/kind/real-index bindings to original pooled features/context and saved full-box rows. All46baseline replays and C/D/R1 maps were verified in parent BOX; reused here without recomputation. Persist every supported/unsupported/NULL decision including unknowns. Independently recompute all model predictions/outcomes/video intervals and verify hashes; no GPU. CPU600s+20, never extend other attempt budgets.  
  *runs/z13_selected_box_value_20261025/independent_audit.json*
- ❌ Dvalue-baseline mean>=2pp and all5paired95%video lower>0: baseline,prior Dbox alpha0.25 selected on20fit,D/C/R1 matched no-new-core controls (C,R1,geometry). Full criteria required; no HOTA from a partial contrast. Values are net one-step assignment effects, not center accuracy.  
  *runs/z13_selected_box_value_20261025/report.json*

## Доказательства
- Точный более узкий ответ и условия до расчёта: `runs/z13_selected_box_value_20261025/protocol.json` · 

