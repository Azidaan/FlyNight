---
tags:
- журнал
дата: '2026-10-25'
время: 03:39
---

# 2026-10-25 03:39 — постановка равного веса запросов вмешательства `z13_region_balanced_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Постановка.* Меняется только взвешивание ошибки целого запроса: 16503 без полезного действия и 1356 с ним на fit84; вес положительного класса 12.170353982300885. Нормировка по сумме весов пакета. Прежние признаки/пакеты/инициализации/1150шагов/три сида, fit20-порог и train26. Это не балансировка колонок balanced_ridge. *Основание.* D/fit84/seed0: принудительно942 полезных из1356 доступных, при обычном выборе24; fit20:96 из160 и0. Источник runs/z13_region_fit_audit_20261025/report.json. Причина дисбаланса остаётся гипотезой; у геометрии принудительный выбор тоже часто удачен. D-baseline mean>=2pp and all five paired-video lower95% bounds>0: baseline,C,R1,geometry,old uniform D. CPU1800с с независимым контроллером, без GPU и повторного извлечения. Источник runs/z13_region_balanced_20261025/protocol.json. Lin et al., ICCV2017, Focal Loss for Dense Object Detection, section3.1 balanced cross entropy: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf . Mechanistic analogy, not the same tracking task or full focal loss.
