---
aliases:
- 'З13: коэффициенты цели на неизменных кандидатах,версия2'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/DiMP
- метка/engineering-future
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: коэффициенты цели на неизменных кандидатах,версия2

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_filter_descriptor_fixed_v2`

Эта реализация описания цели коэффициентами не прошла заданные критерии. Причина недостатка пользы всей FlyVis этим опытом не установлена; фильтр уже участвовал в карте кандидатов.

## Вопрос
Помогают ли наблюдаемые коэффициенты различающего фильтра сверх среднего образца при строго неизменных кандидатах,архитектуре и бюджете обучения?

## Гипотеза
Current response map already uses this filter; coefficients may add no information. Coefficients are not feature means, hence separate descriptor embedding in both matched arms. This test does not diagnose intrinsic representational limits.

## Ожидалось (до прогона)
Исходные104train/26оценки,те же36CPUмоделей/3сида/1150шагов145950предъявлений. CPUизвлечение18000с,обучение1800с,до26.10 07:40МСК. Версия1 сохраняется с невоспроизведённым дискретным вторичным максимумом.

## Что сделано
2026-10-24

## Получено
runs/z13_filter_descriptor_fixed_20261024/train/report.json, paired: {"D_ridge_activity-D_mean_activity": {"mean_per_video": 0.0067190626618319664, "ci95": [-0.009892880829408134, 0.02389615638892894], "better": 12, "n": 26}, "D_ridge_geometry-D_mean_geometry": {"mean_per_video": 0.0, "ci95": [0.0, 0.0], "better": 0, "n": 26}, "D_ridge_activity-D_old_activity": {"mean_per_video": 0.00794889778907348, "ci95": [-0.003965104539360724, 0.019628055846345065], "better": 14, "n": 26}, "D_ridge_activity-D_old_geometry": {"mean_per_video": -0.04062409444666866, "ci95": [-0.08994660500873214, -0.0012932970604897368], "better": 8, "n": 26}, "D_ridge_activity-D_ridge_geometry": {"mean_per_video": -0.03525086114568024, "ci95": [-0.08543047575652651, 0.004616628326362322], "better": 9, "n": 26}, "D_ridge_activity-D_top1": {"mean_per_video": 0.007173625490076747, "ci95": [-0.00820043469568843, 0.020745075265218463], "better": 14, "n": 26}, "D_ridge_activity-D_kalman": {"mean_per_video": -0.03460089398136567, "ci95": [-0.08599253232180461, 0.00606236672970768], "better": 10, "n": 26}, "C_ridge_activity-C_mean_activity": {"mean_per_video": -0.001532513987613331, "ci95": [-0.0150078345202555, 0.011355848119872489], "better": 10, "n": 26}, "C_ridge_geometry-C_mean_geometry": {"mean_per_video": 0.0, "ci95": [0.0, 0.0], "better": 0, "n": 26}, "C_ridge_activity-C_old_activity": {"mean_per_video": 0.0017375900707214836, "ci95": [-0.012344164038457107, 0.016643565885121434], "better": 9, "n": 26}, "C_ridge_activity-C_old_geometry": {"mean_per_video": -0.014802147928938967, "ci95": [-0.04128689075496161, 0.012864817128894677], "better": 8, "n": 26}, "C_ridge_activity-C_ridge_geometry": {"mean_per_video": -0.020377233136344448, "ci95": [-0.045065861734467375, 0.004580365708918701], "better": 7, "n": 26}, "C_ridge_activity-C_top1": {"mean_per_video": 0.023940436812626732, "ci95": [0.0098379801092082, 0.03712936681399862], "better": 18, "n": 26}, "C_ridge_activity-C_kalman": {"mean_per_video": 0.005933897274171291, "ci95": [-0.02069919507675888, 0.03184168512863585], "better": 13, "n": 26}, "R1_ridge_activity-R1_mean_activity": {"mean_per_video": -0.0015843723991267476, "ci95": [-0.01006551284642195, 0.0059705237991264926], "better": 7, "n": 26}, "R1_ridge_geometry-R1_mean_geometry": {"mean_per_video": 0.0, "ci95": [0.0, 0.0], "better": 0, "n": 26}, "R1_ridge_activity-R1_old_activity": {"mean_per_video": 0.0037993176477700706, "ci95": [-0.0032157712849824947, 0.011317009654767132], "better": 8, "n": 26}, "R1_ridge_activity-R1_old_geometry": {"mean_per_video": -0.006791014519711337, "ci95": [-0.014769400025757454, 0.0013597143102856886], "better": 6, "n": 26}, "R1_ridge_activity-R1_ridge_geometry": {"mean_per_video": -0.0038167261170461678, "ci95": [-0.013623622362567176, 0.006952030799402849], "better": 6, "n": 26}, "R1_ridge_activity-R1_top1": {"mean_per_video": 0.06540299680441868, "ci95": [0.034918880415587716, 0.10935669102132536], "better": 23, "n": 26}, "R1_ridge_activity-R1_kalman": {"mean_per_video": -0.010879161795816456, "ci95": [-0.02211306312751038, 0.0004033310320237386], "better": 5, "n": 26}, "D_ridge_activity-R1_ridge_activity": {"mean_per_video": 0.10795262525685913, "ci95": [0.07331636053956131, 0.14620246077536084], "better": 24, "n": 26}, "D_ridge_activity-C_ridge_activity": {"mean_per_video": -0.019738889328170996, "ci95": [-0.07043757024003111, 0.02164046209022752], "better": 16, "n": 26}, "C_ridge_activity-R1_ridge_activity": {"mean_per_video": 0.1276915145850301, "ci95": [0.08306709448007245, 0.17591295733684953], "better": 21, "n": 26}}. Воспроизведение наблюдаемых кадров: runs/z13_filter_descriptor_fixed_20261024/cache/report.json. Завершены36моделей,1150шагов145950предъявлений на модель;число параметров 5361. Старые18контролей/32шага воспроизведены, обе геометрические абляции совпадают.

## Интерпретация
Эта реализация описания цели коэффициентами не прошла заданные критерии. Причина недостатка пользы всей FlyVis этим опытом не установлена; фильтр уже участвовал в карте кандидатов.

## Решение
Не заменять действующий считыватель; сохранять результат и выбирать следующий механизм по первичным измерениям. Не выбирать лучший сид или пороги на train26.

## Нельзя заявлять
Не доказано побитовое воспроизведение прежнего GPUпрохода или прежнего выбора вторичных максимумов. Кандидаты научного опыта берутся только из исходного неизменного кэша. Карта уже использует фильтр; нет гарантии новой информации,причинного диагноза всейFlyVis,безопасногоотказа илиHOTA. Повторно используемыеtrain26.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (4436fd1343a156ddfca48498cebcc2e06c80d65c) — Полный проверенный исход описания цели.
  > Новая версия технического переноса зарегистрирована до исходов; научная проверка коэффициентов/равного контроля прежняя.

## Критерии приёмки (записаны до прогона)
- ✅ Version2 before outcomes: all immutable cached query candidates/centers/features/scores and labels used by both matched arms remain byte-for-byte unchanged (source hashes). Observed descriptors: projection<=1e-10,seed maxabs<=.001; reproduce original fixed observed anchors count,centers<.5px,scores<.001; additional candidate-feature maxabs<=.001 for new chunks. Do not regenerate candidate sets in the scientific comparison. Original version failed regenerated peak-set replay and remains inconclusive; this version does not assert that predicate is reproduced. Log all regenerated selection differences. Previously successful chunks retain their stricter original regeneration checks and hashes,not recomputed. Все9наборов/18старыхмоделей/32шага проверены.  
  *runs/z13_filter_descriptor_fixed_20261024/train/report.json; reproduction.json; timing.json; controller_result.json; runs/z13_filter_descriptor_fixed_20261024/cache/report.json*
- ❌ D ridge-D mean >=2п.п. и нижняя95%повидеоДИ>0.  
  *runs/z13_filter_descriptor_fixed_20261024/train/report.json; reproduction.json; timing.json; controller_result.json; runs/z13_filter_descriptor_fixed_20261024/cache/report.json*
- ❌ D ridge лучше D old activity,D old geometry,R1 ridge,D ridge geometry; все нижние95%ДИ>0.  
  *runs/z13_filter_descriptor_fixed_20261024/train/report.json; reproduction.json; timing.json; controller_result.json; runs/z13_filter_descriptor_fixed_20261024/cache/report.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [useful_center_fraction↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D1%85%20%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%BE%D0%B2%2C%20%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5%20%D0%BF%D0%BE%20%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE) | 0.26205943579340396 | 0.255340373131572 | train26 observed-filter D | `runs/z13_filter_descriptor_fixed_20261024/train/report.json` · summary.D_ridge_activity.useful_mean_video |

## Доказательства
- Фиксированные данные и критерии до исходов версии2: `runs/z13_filter_descriptor_fixed_20261024/cache/protocol.json` · 
- Все парные исходы: `runs/z13_filter_descriptor_fixed_20261024/train/report.json` · 
- Воспроизведение старого чтения: `runs/z13_filter_descriptor_fixed_20261024/train/reproduction.json` · 
- Своё дерево завершено: `runs/z13_filter_descriptor_fixed_20261024/train/controller_result.json` · 

## Код
- `scripts/z13_filter_descriptor_fixed_20261024.py`
- `tracking/z13_fixed_candidate_replay.py`
- `tests/test_z13_fixed_candidate_replay.py`
- `scripts/z13_filter_descriptor_fixed_console.ps1`

