---
tags:
- лучший-результат
- результат/tracker
---

# Центрированная D лучше временного входа R1;пользы сверх стенда не подтверждено

↑ [[11 Лучшие результаты/Польза полному стенду|Польза полному стенду]]

[Исходная карточка: гипотеза, критерии и полный итог↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fz13_val20_common_v1)

**Результат:** +0,11 пункта; 95% доверительный интервал [+0,00447; +0,22]; видео: 20.

**Метрика:** HOTA.

**Сравнение:** centerD_masa-centerR1_masa

**Выборка:** Фиксированные20ранее использованныхval

**Условия:** Original-frame960/conf.01/iou.7 detections,high.25/low.1/new.25/buffer30, native matching/GMC defaults,ReIDoff. Same MASA R50weights and base_masa scorer trained earlier on130train,gap150,dist5,threshold.6. Native raw outputs equal independent CPU20 exactly. Strict original annotation mapping from verified cache,including conflict repair retained. No fitting/tuning on20val.

**Ограничения:** All200val videos were used in older low-frame-rate tracker research; detector val monitored during weight selection. New20 are additional original-frame caches, not untouched independent confirmation or final test. No outcome-dependent exclusion; ambiguous alignment stops preparation, no replacement video. Additional20 were selected before outcomes, but are not untouched final test. Native algorithms not optimally tuned. MASA scorer trained on XYWH pairs may transfer unequally;show raw and MASA. One saved D is not a seed-independence result. Prior7 subset and these20 remain separate tables;do not compare their raw numbers as changes. C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included. Вторичное системное сравнение;преимущество D над стендом без мухи не подтверждено.

**Исход полной карточки:** положительный.

В полном эксперименте есть непройденные критерии; это отдельное положительное сравнение, а не успех всего эксперимента.

**Первичный источник:** `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json`, ключ `paired.centerD_masa-centerR1_masa`.

Относительный процент не вычислен: абсолютная база не указана либо равна нулю.

Доля измеренного потолка не определена: соответствующий потолок для этого сравнения не записан.
