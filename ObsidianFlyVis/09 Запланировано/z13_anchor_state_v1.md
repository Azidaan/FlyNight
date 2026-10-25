---
aliases:
- 'Состояние FlyVis: отдельный запрос от сохранённого образца'
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/fly-problem-P3
- метка/fly-problem-P8
исход: готов к запуску
направление: FlyVis
начат: '2026-10-25'
---

# ⚪ Состояние FlyVis: отдельный запрос от сохранённого образца

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `z13_anchor_state_v1`

## Вопрос
Помогает ли накопленная между образцом и запросом активность или её исключение улучшит назначения?

## Гипотеза
Рабочая гипотеза: история поисковых кадров может сохранять мешающий отклик; это не установленная причина. Удаление истории может также лишить муху полезного движения.

## Ожидалось (до прогона)
One internal state intervention: suppress updates from intervening search frames. For each query, execute original frozen core on exactly its original four exemplar frames followed by this query. Keep the exemplar itself, all34features/neighbors, original norms/lambda/Gaussian target/mean region readout/null0/alpha0.25. Not combined with failed uncertainty or NULL mixture. Seven conditions C/D/R1 old/anchor plus geometry.

## Нельзя заявлять
Intervening search frames may contain useful target motion or disambiguation. Removing them changes available information intentionally and could hurt D even if learned spatial transport would help. Frozen norms/weights were learned for full recurrence; a failed ablation does not disprove KYS. Positive first-assignment effect is not HOTA or proof of all confidence failures. Original teacher/tracker errors in exemplar are retained, not corrected by truth.

## Критерии приёмки (записаны до прогона)
- ⏳ Three nonlinear-state tests: first-search/exemplar exact; middle-history invariance; exemplar remains influential. Exposure lower95%>2% before extraction. Every8326old C/D/R1 center<0.5px/score<0.001 against retained original. Identical causal exemplar/first-search prefix arrays reused from the old producer; first-search score maps exactly equal on every retained query, no separate-GPU-pass numerical comparison. Full source/row bindings, finite maps, exact alpha0/baseline and all old actions. First2Dgroup conservative price must fit original1800s; completed chunks reused.
- ⏳ Danchor-baseline and Danchor-Dold means>=2pp, all6paired-video95%lower>0, includingC/R1/geometry and D-R1 intervention interaction. Reused train26 exploratory. No alpha/normalization/lambda fitting or favorable history/subset selection.

## Доказательства
- Одно изменение состояния, равные контроли и критерии до GPU: `runs/z13_anchor_state_20261025/protocol.json` · 

