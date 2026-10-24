---
tags:
- журнал
дата: '2026-10-24'
время: 00:40
---

# 2026-10-24 00:40 — зарегистрировано общее сравнение трекеров `z13_matched_trackers_v1`

↑ [[06 Журнал/Дни/2026-10-24|2026-10-24]]

*Постановка.* Автор: «Хочу результаты на одинаковых условиях». All7existing val videos, same original frames and960/conf.01 YOLO cache, same annotation alignment and official TrackEval. Primary total-system comparison includes native ByteTrack,BoT-SORT,TrackTrack,XYWHByteTrack,XYWH+legacyD,XYWH+centerD. Show without and with identical frozen base+MASA stitcher (gap150,dist5,threshold.6). No values from old200val or35train-derived dev in this table.

Common high.25,low.1,new.25,buffer30,fuse_scoreTrue. Each native algorithm retains its documented association-cost settings and GMC; these costs have different meanings, so a common numeric match threshold is not imposed. ReID disabled in all native trackers; identical MASA postprocessing is reported separately for every system. These fixed settings are not claimed optimal.

Сначала CPU-прогон4базовых трекеров на всех7val и точное воспроизведение XYWH. Затем цена GPU/сигнатур/D и общая сшивка при достаточном остатке окна. Сохранение07:35,конец07:39:30MSK24.10;CPU3600с. Новые данные/75отложенных/подбор по val исключены. Ограничения: Only7previously used val videos; not independent final test and no75heldout. Detector validation set used in its training monitoring, but these videos are absent from gradient-training list. Settings not optimized per tracker; cannot infer universal ranking or compare directly with published benchmarks. The old MASA head was trained on XYWH life pairs and can transfer differently to other trackers; report raw/MASA separately.

Числа пока не получены; прежние200val/35dev в новую таблицу не переносятся.
