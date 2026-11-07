---
aliases:
- 'З13: достаточно ли точного знания присутствия для отказа'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 З13: достаточно ли точного знания присутствия для отказа

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_presence_scope_v1`

Проверены метки и идеальное бинарное знание присутствия при прежнем порядке пиков; критерий не пройден.

## Вопрос
Исправит ли только знание присутствия отказ, если центр и порядок по высоте среди присутствующих целей прежние?

## Гипотеза
Недостающий обучаемый выход отсутствия — кандидат, но полезность этого действия должна пройти проверку до обучения.

## Ожидалось (до прогона)
Read only metadata arrays for all130 training videos; partition every label into present, annotated target outside square, annotated target outside hex field, target not annotated in query, unknown observed identity/stale, or unknown query annotation gap. Report augmented and natural data separately. No raw hex frames, pixels, dev or heldout loaded. No annotation correction. On identical known5822 natural queries forD/C/R1 select ceil(1%*5822)=59. Base sorts original score; visibility-priority sorts kind1 first, then original score, then label. Unknown2504 stay excluded from BOTH arms and separately counted, never relabeled absent. This is an oracle diagnostic restricted to the known cohort, not a deployable policy or an upper bound on all possible continuous visibility scorers. Separately, suppress kind0 only in the unchanged actual top1% selection without refilling to isolate pure rejection from reranking.

## Что сделано
2026-10-25

## Получено
runs/z13_presence_scope_20261025/report.json → label_counts: {'fit104': {'all_augmented_and_natural': {'target_center_in_eye': 13235, 'target_not_annotated_in_query': 24658, 'unknown_observed_identity_or_stale': 15760, 'annotated_target_outside_hex_field': 403, 'annotated_target_outside_square': 3428, 'unknown_query_annotation_gap': 396}, 'natural_only': {'target_center_in_eye': 6650, 'target_not_annotated_in_query': 12332, 'unknown_observed_identity_or_stale': 7999, 'annotated_target_outside_square': 1622, 'annotated_target_outside_hex_field': 174, 'unknown_query_annotation_gap': 193}}, 'evaluate26': {'all_augmented_and_natural': {'target_center_in_eye': 3632, 'target_not_annotated_in_query': 6728, 'unknown_observed_identity_or_stale': 4841, 'unknown_query_annotation_gap': 79, 'annotated_target_outside_square': 975, 'annotated_target_outside_hex_field': 106}, 'natural_only': {'target_not_annotated_in_query': 3513, 'unknown_observed_identity_or_stale': 2463, 'target_center_in_eye': 1786, 'annotated_target_outside_square': 471, 'unknown_query_annotation_gap': 41, 'annotated_target_outside_hex_field': 52}}}; counts: {'D_height': {'beneficial': 0, 'harmful': 0, 'neutral': 59, 'unknown': 0, 'selected': 59}, 'D_visibility': {'beneficial': 4, 'harmful': 3, 'neutral': 52, 'unknown': 0, 'selected': 59}, 'C_height': {'beneficial': 0, 'harmful': 0, 'neutral': 59, 'unknown': 0, 'selected': 59}, 'C_visibility': {'beneficial': 2, 'harmful': 1, 'neutral': 56, 'unknown': 0, 'selected': 59}, 'R1_height': {'beneficial': 0, 'harmful': 0, 'neutral': 59, 'unknown': 0, 'selected': 59}, 'R1_visibility': {'beneficial': 1, 'harmful': 3, 'neutral': 55, 'unknown': 0, 'selected': 59}}; paired: {'Dvisibility-baseline': {'mean_per_video': -0.0004095023585848609, 'ci95': [-0.0018239277270206005, 0.0006545601242106642], 'better': 4, 'n': 26}, 'Dvisibility-D_height': {'mean_per_video': -0.0004095023585848609, 'ci95': [-0.0018239277270206005, 0.0006545601242106642], 'better': 4, 'n': 26}, 'Dvisibility-C_visibility': {'mean_per_video': -9.644531758664337e-05, 'ci95': [-0.0009206331023159019, 0.0005719187875321704], 'better': 3, 'n': 26}, 'Dvisibility-R1_visibility': {'mean_per_video': -0.00018136916293255687, 'ci95': [-0.0017587747557746534, 0.0010803267048571125], 'better': 6, 'n': 26}}. Отдельно сохранён unchanged-frozen-policy отказ без пополнения.

## Интерпретация
Это привилегированная проверка достаточности одного бинарного признака для конкретного порядка выбора; не предел всех обучаемых оценок, не доказательство причин и не новая версия FlyVis.

## Решение
Pass: separately register target-visibility learner with fit-only calibration and same centers/controls. Fail: do not train this presence-only version; next action must change localization or target-conditioned within-present ranking, not another scalar threshold. At most this one prerequisite for the proposed action. Итог основного критерия: False.

## Нельзя заявлять
Even perfect target presence may retain badly localized or unnecessary corrections. Existing old heads already learned kind0; this is not a claim no refusal training existed. Current ridge confidence omits that head. Missing query annotation is not proof of physical absence. Unknown initial identity can come from sparse supervision. Annotation-based oracle is optimistic and cannot prove a learned model can recover it.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (ffbf266be9c155451599135a73a6c0cb38867d0a) — Проверены исходы диагностического знания присутствия.
  > Одна диагностическая проверка перед возможной заменой отказа; не внедрение.

## Критерии приёмки (записаны до прогона)
- ✅ Все130видео/все8326строк каждого варианта точно связаны; разбиения исчерпывающие; неизвестные не переименованы в отрицательные.  
  *runs/z13_presence_scope_20261025/independent_audit.json*
- ❌ Dvisibility vs baseline0, Dheight, Cvisibility, R1visibility: all four lower95%paired-videoCI>0. Only this specific presence-priority repair is eligible for a separate learned version if all pass; otherwise stop presence-only correction preserving this ranking. Do not conclude absence learning in general cannot help.  
  *runs/z13_presence_scope_20261025/report.json*

## Доказательства
- Действие, альтернатива и критерии до расчёта: `runs/z13_presence_scope_20261025/protocol.json` · 
- Разбиение меток и все парные сравнения: `runs/z13_presence_scope_20261025/report.json` · 

## Код
- `tracking/z13_presence_scope.py`
- `tests/test_z13_presence_scope.py`
- `scripts/z13_presence_scope_20261025.py`
- `scripts/z13_presence_scope_console.ps1`

