---
tags: [проблема-проекта, проблема/open]
aliases: [С1]
---

# Дополнительная польза FlyVis пока не подтверждена

[[Обзор проекта/Проблемы|Проблемы]]

Статус: **открыта**. Компонент: Модуль FlyVis и его вклад во всю систему.

## 1. Что установлено

Полный пилот Б1 и аудит двадцати видео завершены. Весь вариант D лучше исходного трекера, но его добавка к той же основе без FlyVis и к контролям C/R1/B не подтверждена. Это факт об этой реализации и одном начальном значении, не обо всём направлении. Последующие полные проверки стоимости назначения и собственной пространственной карты также не подтвердили принятую добавку FlyVis. Проверка собственной карты дала HOTA +0,011110 п.п. к B0 с 95% ДИ [−0,073633;0,102324] и FP +35; контроль по давности — −0,004454 п.п. с ДИ, содержащим ноль, и FP +29. Источники: runs/singleton_verify_20261108/full/comparison_report.json; runs/singleton_age_20261108/full/comparison_report.json. Локальная прибавка на запросах не стала решением полного трекера. При сохранении точных штатных рамок и возврате номера только после подтверждения новая версия FlyVis дала HOTA+0,006806п.п.,95%ДИ[−0,003559;0,018089]. Геометрия с той же давностью дала+0,006532п.п., её ДИ тоже содержит0. Добавка FlyVis и принятый выигрыш всей системы не установлены (runs/native_confirm_risk_20261108/full/comparison_report.json). При новом относительном совмещении полныйFlyVisдалHOTAкакB0;он разрешил1вмешательство. IMAGEдал+0,003723п.п.,95%ДИ[0;0,009625],непринят;его преимуществонадгеометриейи согласованным случайным отказом не установлено. Это отрицание принятой прибавки конкретной политики, не отсутствие информации во всех пространственных картах (runs/bbox_normalized_signals_20261108/full/comparison_report.json).

## 2. Рабочие объяснения

Возможная версия: полезного временного сигнала недостаточно для дополнительного решения поверх сильной основы; не доказано.

Возможная версия: обучение и способ применения ограничивают доступную пользу; различие постановок подтверждено, причинный эффект не измерен.

## 3. Следующая различающая проверка

Первая проблема остаётся открытой. Для оценки правила отказа нужны другие независимо размеченные свои и чужие машины с сопоставимым числом ячеек. Порог по этим двум парам не устанавливать.

## 4. Что позволит считать проблему решённой

Дополнительный вклад конкретного модуля проходит заранее заданное парное сравнение с сильным согласованным контролем и воспроизводится; польза всей системы учитывается отдельно.

## 5. Повторяющиеся сочетания

[[Проблемы проекта/Повторения/Сложная часть не обходит простой контроль|Сложная часть не обходит простой контроль]]

## 6. Основания и история случаев

[Полный результат и все парные сравнения Б1](file:///E:/Dissertation/runs/continuation_20261028/b1_tracker/report.json)

[Полная независимая проверка двадцати видео](file:///E:/Dissertation/runs/continuation_20261028/b1_tracker/independent_audit.json)

[Первая проверка отклика; не доказательство пользы ядра для трекера](file:///E:/Dissertation/runs/flyvis_y1_latency_fp64_20261104/analysis.json)

[Первая проверка отклика; не доказательство пользы ядра для трекера](file:///E:/Dissertation/runs/flyvis_y1_latency_fp64_20261104/independent_audit.json)

[Проверка этой версии пространственного решения и исходного общего назначения](file:///E:/Dissertation/runs/spatial_identity_maps_20261107/comparison_report.json)

[Проверка этой версии пространственного решения и исходного общего назначения](file:///E:/Dissertation/runs/spatial_identity_maps_20261107/bottleneck_report.json)

[Проверка этой версии пространственного решения и исходного общего назначения](file:///E:/Dissertation/runs/spatial_identity_maps_20261107/native_diagnosis.json)

[Проверка этой версии пространственного решения и исходного общего назначения](file:///E:/Dissertation/runs/spatial_identity_maps_20261107/competitor_histories.json)

[Полная первая версия назначения не прошла критерий](file:///E:/Dissertation/runs/identity_assignment_20261108/full_precision/comparison_report.json)

[Полная собственная карта не прошла критерий](file:///E:/Dissertation/runs/singleton_verify_20261108/full/comparison_report.json)

[Контроль давности тоже не прошёл критерий](file:///E:/Dissertation/runs/singleton_age_20261108/full/comparison_report.json)

[Независимая проверка полного пространственного пилота](file:///E:/Dissertation/runs/singleton_verify_20261108/full/independent_audit.json)

[Независимая проверка давности](file:///E:/Dissertation/runs/singleton_age_20261108/full/independent_audit.json)

[Граница причинного объяснения текущего входа](file:///E:/Dissertation/runs/singleton_age_20261108/current_input_dependency.json)

[Полный контроль подтверждения присутствия и отдельного возврата личности](file:///E:/Dissertation/runs/native_confirm_risk_20261108/full/comparison_report.json)

[Независимые траектории, метрики, возвраты и пространственные оценки](file:///E:/Dissertation/runs/native_confirm_risk_20261108/full/independent_audit.json)

[Полное сравнение при общем относительном совмещении](file:///E:/Dissertation/runs/bbox_normalized_signals_20261108/full/comparison_report.json)

[Независимая проверка](file:///E:/Dissertation/runs/bbox_normalized_signals_20261108/full/independent_audit.json)

[Снижение счётчика из-за устранённого разрыва, не потеря этого продолжения](file:///E:/Dissertation/runs/identity_evidence_20261108/return_count_diagnostic.json)

[На готовом fit-подмножестве недостаточно чужих пар для калибровки](file:///E:/Dissertation/runs/bbox_fit_readout_readiness_20261108/readiness.json)

[Межвидеоваякалибровкапо настоящимположительным и чужимfit-парам,безподборапо26/20val](file:///E:/Dissertation/runs/bbox_readout_calibration_20261108/comparison_report.json)

[Данные действующего допуска и непройденное условие наличия чужих примеров](file:///E:/Dissertation/runs/singleton_working_domain_20261108/recovery2/report.json)

[При каких условиях сдвиг пространственных карт можно сравнить в координатах кадра](file:///E:/Dissertation/runs/flyvis_local_shift_feasibility_20261108/identifiability.json)

[Есть ли прямые метки своих и чужих автомобилей и сопоставимые карты](file:///E:/Dissertation/runs/singleton_car_map_coverage_20261108/inventory.json)

[Узнаёт ли пространственная карта машину после поиска сдвига на двух размеченных парах](file:///E:/Dissertation/runs/singleton_two_map_scores_20261108/scores.json)

Последние содержательные случаи:

2026-11-08 — С исключением целого fit-видео:FlyVisподобранныйпорог8правильных/1чужихпринятий;старый0,25:7/1. IMAGE_FIT:4/1,R1_FIT:0/1,DIS_FIT:0/1. Условиегипотезыкалибровки=False. Источник:runs/bbox_readout_calibration_20261108/comparison_report.json;независимаяпроверка:independent_audit.json.

2026-11-08 — На20дополнительныхfit-видео в рабочем допуске:6правильных,1чужая,23неизвестных пары; минимум3чужих из2видео не достигнут.

2026-11-08 — При 19 сдвигах сходство FlyVis на чужой паре 390→448 осталось 0,3493 и на своей 1738→1834 осталось 0,8027. На чужой паре яркость выросла с −0,7733 до 0,9997 и градиенты с −0,5637 до 0,9206. На своей яркость осталась 0,9146 и градиенты 0,8275.

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Связанный результат или постановка](obsidian://open?vault=ObsidianFlyVis&file=%D0%9E%D0%BF%D0%BE%D1%80%D0%BD%D1%8B%D0%B5%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8%2F%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%203%20%E2%80%94%20%D0%A3%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C%20%D0%BD%D1%83%D0%B6%D0%BD%D1%8B%D0%B9%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%20%D0%B8%20%D0%B2%D1%8B%D0%B1%D0%B8%D1%80%D0%B0%D1%82%D1%8C%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D0%BE%D0%B5%20%D0%B8%D1%81%D0%BF%D1%80%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5)

[↗ Прежняя запись P1 — статус не пересматривался](obsidian://open?vault=ObsidianFlyVis&file=10%20%D0%9F%D1%80%D0%BE%D0%B1%D0%BB%D0%B5%D0%BC%D1%8B%20%D0%BC%D1%83%D1%85%D0%B8%2FP1%20%E2%80%94%20%D0%9F%D0%BE%D1%87%D1%82%D0%B8%20%D0%B2%D1%81%D1%8F%20%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BC%D1%83%D1%85%D0%B8%20%D1%83%D0%B6%D0%B5%20%D0%B5%D1%81%D1%82%D1%8C%20%D0%B2%20%D0%B5%D1%91%20%D0%B2%D1%85%D0%BE%D0%B4%D0%B5)

[↗ Прежняя запись P5 — статус не пересматривался](obsidian://open?vault=ObsidianFlyVis&file=10%20%D0%9F%D1%80%D0%BE%D0%B1%D0%BB%D0%B5%D0%BC%D1%8B%20%D0%BC%D1%83%D1%85%D0%B8%2FP5%20%E2%80%94%20%D0%97%D0%BD%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BC%D1%83%D1%85%D0%B8%20%D0%BD%D0%B5%20%D0%BC%D0%B5%D0%BD%D1%8F%D0%B5%D1%82%20%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%82%D0%B0%D0%BC%2C%20%D0%B3%D0%B4%D0%B5%20%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%D0%B0%D0%B6%D0%BD%D0%BE)

Внесено в карту 2026-10-28. История случаев и контрольные суммы источников: [реестр проблем](file:///E:/Dissertation/experiments/system_problems.json).
