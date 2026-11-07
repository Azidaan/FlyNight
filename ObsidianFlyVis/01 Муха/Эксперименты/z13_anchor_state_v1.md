---
aliases:
- 'Состояние FlyVis: отдельный запрос от сохранённого образца'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P3
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Состояние FlyVis: отдельный запрос от сохранённого образца

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_anchor_state_v1`

Независимые запросы от образца: критерий не пройден.

## Вопрос
Помогает ли накопленная между образцом и запросом активность или её исключение улучшит назначения?

## Гипотеза
Рабочая гипотеза: история поисковых кадров может сохранять мешающий отклик; это не установленная причина. Удаление истории может также лишить муху полезного движения.

## Ожидалось (до прогона)
One internal state intervention: suppress updates from intervening search frames. For each query, execute original frozen core on exactly its original four exemplar frames followed by this query. Keep the exemplar itself, all34features/neighbors, original norms/lambda/Gaussian target/mean region readout/null0/alpha0.25. Not combined with failed uncertainty or NULL mixture. Seven conditions C/D/R1 old/anchor plus geometry.

## Что сделано
2026-10-25

## Получено
runs/z13_anchor_state_20261025/replay/report.json → paired: {'Danchor-baseline': {'mean_per_video': -0.0011120743487861519, 'ci95': [-0.003068058784933635, 0.0006451375851377589], 'better': 4, 'n': 26}, 'Danchor-Dold': {'mean_per_video': -9.139313986556336e-05, 'ci95': [-0.0005403605149029773, 0.0003375459507835645], 'better': 1, 'n': 26}, 'Danchor-C_anchor': {'mean_per_video': 0.0, 'ci95': [0.0, 0.0], 'better': 0, 'n': 26}, 'Danchor-R1_anchor': {'mean_per_video': 0.0006012856688248952, 'ci95': [-0.0012865402155445365, 0.002503025894650473], 'better': 9, 'n': 26}, 'Danchor-geometry': {'mean_per_video': -0.0008991955183635266, 'ci95': [-0.001940043462159755, -0.00010295818104426801], 'better': 1, 'n': 26}, 'anchor_effect_D-R1': {'mean_per_video': 0.0007627324564738584, 'ci95': [-0.00031266377295046673, 0.002110775922080651], 'better': 3, 'n': 26}}

## Интерпретация
Парное вмешательство меняет только накопление поискового состояния, сохраняя образец/текущий запрос и считыватель. Это поисковая проверка первых назначений на многократно использованном train26, не HOTA. Неопределённый эффект не доказывает ни вредность всей памяти, ни невозможность пространственного переноса.

## Решение
Pass -> distinct runtime parity/price and frozen matched closed-loop gate; no HOTA until that protocol. Fail -> keep this precise state ablation; do not tune reset frequency or retrain core to rescue. Use primary outcome to distinguish harmful-history evidence from absent evidence; separately design spatial correspondence only with a measurable cheap prerequisite. Primary: False

## Нельзя заявлять
Intervening search frames may contain useful target motion or disambiguation. Removing them changes available information intentionally and could hurt D even if learned spatial transport would help. Frozen norms/weights were learned for full recurrence; a failed ablation does not disprove KYS. Positive first-assignment effect is not HOTA or proof of all confidence failures. Original teacher/tracker errors in exemplar are retained, not corrected by truth.

## Критерии приёмки (записаны до прогона)
- ✅ Three nonlinear-state tests: first-search/exemplar exact; middle-history invariance; exemplar remains influential. Exposure lower95%>2% before extraction. Every8326old C/D/R1 center<0.5px/score<0.001 against retained original. Identical causal exemplar/first-search prefix arrays reused from the old producer; first-search score maps exactly equal on every retained query, no separate-GPU-pass numerical comparison. Full source/row bindings, finite maps, exact alpha0/baseline and all old actions. First2Dgroup conservative price must fit original1800s; completed chunks reused.  
  *runs/z13_anchor_state_20261025/independent_audit.json*
- ❌ Danchor-baseline and Danchor-Dold means>=2pp, all6paired-video95%lower>0, includingC/R1/geometry and D-R1 intervention interaction. Reused train26 exploratory. No alpha/normalization/lambda fitting or favorable history/subset selection.  
  *runs/z13_anchor_state_20261025/replay/report.json*

## Доказательства
- Одно изменение состояния, равные контроли и критерии до GPU: `runs/z13_anchor_state_20261025/protocol.json` · 

