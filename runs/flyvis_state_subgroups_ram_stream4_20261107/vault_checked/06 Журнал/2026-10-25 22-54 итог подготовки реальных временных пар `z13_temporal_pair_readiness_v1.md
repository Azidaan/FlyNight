---
tags:
- журнал
дата: '2026-10-25'
время: '22:54'
---

# 2026-10-25 22:54 — итог подготовки реальных временных пар `z13_temporal_pair_readiness_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено (`z13_temporal_pair_readiness_v1`).* runs/z13_temporal_pair_readiness_20261025/report.json → counts/per_split/oracle26: {"counts": {"rows": 37296, "known": 26600, "current_supported": 22090, "old_supported": 14252, "restored_supported": 18824, "gained_support": 4926, "lost_support": 354, "labeled_pairs": 16605, "positive_pair": 14786, "oracle_all": 1543, "oracle_old": 698, "oracle_restored": 1303, "oracle_labeled": 1231, "oracle_target_linked": 993, "oracle_gained": 628, "oracle_lost": 23, "oracle_restored_gap12": 1227, "positive_correspondences": 23612, "negative_correspondences": 52955}, "evaluate26": {"rows": 8326, "known": 5822, "current_supported": 4781, "old_supported": 3061, "restored_supported": 4064, "gained_support": 1077, "lost_support": 74, "labeled_pairs": 3507, "positive_pair": 3173, "oracle_all": 325, "oracle_old": 142, "oracle_restored": 278, "oracle_labeled": 262, "oracle_target_linked": 220, "oracle_gained": 142, "oracle_lost": 6, "oracle_restored_gap12": 258, "positive_correspondences": 4993, "negative_correspondences": 10767}, "oracle26": {"oracle_all": {"mean_per_video": 0.05142068680371603, "ci95": [0.04160032885690229, 0.06108713813385377], "better": 25, "n": 26}, "oracle_old": {"mean_per_video": 0.020687911027830796, "ci95": [0.015621519569993823, 0.025558923539599303], "better": 23, "n": 26}, "oracle_restored": {"mean_per_video": 0.04435231019388354, "ci95": [0.03571805983489297, 0.05293940350436568], "better": 25, "n": 26}, "oracle_labeled": {"mean_per_video": 0.04195101572979871, "ci95": [0.03405192290323297, 0.04974194459029022], "better": 25, "n": 26}, "oracle_target_linked": {"mean_per_video": 0.03547083105427311, "ci95": [0.027773732064335862, 0.043108326811484465], "better": 25, "n": 26}, "oracle_gained": {"mean_per_video": 0.024830543369812306, "ci95": [0.018781363453595143, 0.03123463143023305], "better": 25, "n": 26}, "oracle_lost": {"mean_per_video": 0.0011661442037595566, "ci95": [0.00019976347182229534, 0.0024705689091136895], "better": 5, "n": 26}, "oracle_restored_gap12": {"mean_per_video": 0.040894287048802505, "ci95": [0.032616670641142494, 0.04898171630395831], "better": 25, "n": 26}}}

*Вывод.* Измерена готовность данных при фиксированном правиле прошлого внутри исходного эпизода. Номера трекера не являются метками личности. Неизвестные пары исключены из меток соответствия, но не из исходных знаменателей. Наличие областей/меток не доказывает обучаемость и пользу FlyVis.

*Решение.* Сохранить подготовленные входы и отдельную разметку. Новый расчёт D/C/R1 и обучение в этот согласованный шаг не входят. Следующий опыт требует проверки воспроизведения состояния, равных признаков и цены до обучения.

*Подтверждённые улучшения.* Новых confirmed_results нет: это подготовка данных без обучаемого результата. HOTA не измерялась; процент изменения стенда и реализованная моделью доля потолка не определены.
