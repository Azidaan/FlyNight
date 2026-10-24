---
tags:
- журнал
- журнал/муха
дата: '2026-10-24'
время: '23:23'
---

# 2026-10-24 23:23 — постановка меток по последствиям сопоставления `z13_assignment_effect_v1`

↑ [[06 Журнал/Дни/2026-10-24|2026-10-24]]

*Действие.* Prepare supervision for a replacement FlyVis readout trained on the association consequence of its center action,rather than distance-to-GT-center utility. CPU exact baseline replay only;no new model trained here.

*Критерии.* All130videos replay float32 raw outputs exactly against saved baseline;every cached label prediction matches replay mean castfloat32 exactly;all rows bound byvideo,track,frame,identity. First2fixed fit20videos measure walltime;2*mean_seconds*remaining128+180must fit actual remaining original budget,else retain2jobs and stop. Completed videos not repeated. On all26reused training evaluation videos,mean fraction of known eligible queries with at least one D candidate producing net beneficial first-stage assignment >=0.02 and paired-video95%lower>0 against0. If upper<0.02 this candidate set has insufficient measured reserve;otherwise inconclusive. C/R1 reserve and geometry-preserving no-action baseline reported. Old center-useful flags versus benefit/harm table reported. Oracle is diagnostic only;it never becomes a runtime policy.

*Возражение.* One-step baseline-state counterfactuals omit feedback,later low-score/unconfirmed association and long-term identity. Sparse annotations and last observed identity limit semantic labels. Higher oracle reserve need not be learnable. This is an association-aligned readout target hypothesis,not DeepMOT reproduction or a change to YOLO/ByteTrack/MASA.

DeepMOT[§4↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F4.%20%D0%A7%D1%82%D0%BE%20%D0%B7%D0%B0%D0%BA%D1%80%D1%8B%D1%82%D0%BE%2C%20%D0%B8%20%D0%BF%D0%BE%20%D0%BA%D0%B0%D0%BA%D0%BE%D0%BC%D1%83%20%D0%BE%D1%81%D0%BD%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8E),https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_How_to_Train_Your_Deep_Multi-Object_Tracker_CVPR_2020_paper.html поддерживает согласование обучения с решением трекера;наши точные однокадровые метки — отдельная гипотеза. Первый2fit20ролика — воспроизведение/цена,а не вывод об эффективности. CPU3600с,безвидео/GPU/ядра/dev/val20подбора/75/загрузок.
