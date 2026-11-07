---
tags:
- журнал
дата: '2026-10-25'
время: '14:32'
---

# 2026-10-25 14:32 — итог выбора геометрического действия `z13_proposal_geometry_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_proposal_geometry_20261025/recovery1/report.json → remaining895_detection_partition/remaining895_first_helpful_action/paired: {'remaining895_detection_partition': {'correct_high_in_eye': 553, 'correct_high_outside_eye': 334, 'identity_not_annotated_here_not_proven_absent': 4, 'annotated_object_without_labelled_raw_detection': 1, 'correct_detection_only_below_high_threshold': 3}, 'remaining895_first_helpful_action': {'boxes_eye': 176, 'centers_eye': 113, 'boxes_frame': 71, 'none_of_tested_actions': 443, 'oracle_gt_box_eye': 4, 'centers_frame': 82, 'null': 4, 'oracle_gt_center_eye': 2}, 'paired': {'null-baseline': {'mean_per_video': 0.010938744069494187, 'ci95': [0.0074364404126687515, 0.015018934603363142], 'better': 22, 'n': 26}, 'centers_eye-baseline': {'mean_per_video': 0.026993698388551486, 'ci95': [0.019500768530289193, 0.03457753994915234], 'better': 21, 'n': 26}, 'boxes_eye-baseline': {'mean_per_video': 0.0524127276987803, 'ci95': [0.04232102625676238, 0.0625369349210125], 'better': 25, 'n': 26}, 'centers_frame-baseline': {'mean_per_video': 0.040889259771183815, 'ci95': [0.03075086341998858, 0.052255166112204626], 'better': 24, 'n': 26}, 'boxes_frame-baseline': {'mean_per_video': 0.07252156438672289, 'ci95': [0.06000494888042196, 0.08582558827957609], 'better': 25, 'n': 26}, 'oracle_gt_center_eye-baseline': {'mean_per_video': 0.020758870645376366, 'ci95': [0.014336022798098078, 0.027336333567793775], 'better': 21, 'n': 26}, 'oracle_gt_box_eye-baseline': {'mean_per_video': 0.029378982408607868, 'ci95': [0.02148663565678659, 0.03748551161520981], 'better': 23, 'n': 26}, 'centers_eye-Dpeak3': {'mean_per_video': 0.010660408171653788, 'ci95': [0.005103790162745841, 0.0166618781558551], 'better': 15, 'n': 26}, 'boxes_eye-centers_eye': {'mean_per_video': 0.025419029310228818, 'ci95': [0.01849229629111801, 0.032841937480310404], 'better': 24, 'n': 26}, 'boxes_frame-boxes_eye': {'mean_per_video': 0.020108836687942586, 'ci95': [0.013509412629053717, 0.028203906465408005], 'better': 22, 'n': 26}}}

*Вывод.* Диагностическая достижимость действий с идеальным выбором, без преимущества какого-либо представления. Правильная детекция/центр/рамка и полезное назначение различаются. Не причинное доказательство неисправности ядра и не HOTA.

*Решение.* If a geometry gate passes, register one learned FlyVis internal proposal/box readout with equal C/R1/geometry controls, fit-only settings and original D reproduction; do not repeat failed size-regression or peak rankers. If both fail, preserve counts and stop these proposed center/full-box implementations; choose a different action with evidence rather than tune gates. Допуск центра: True; полной рамки: True.

HOTA не измерялась, процент к стенду и доля потолка обученного узла не определены. Геометрический оракул не добавляется в лучшие результаты FlyVis.
