---
aliases:
- 'Итоговый трекер линии B: каскад буферов IoU + отложенное рождение по q_birth + сшивка обрывков по геометрии'
tags:
- эксперимент
- исход/inconclusive
- направление/own_tracker
- метка/line-B
- метка/end-to-end
- метка/hota
- метка/idsw
- метка/c-biou
- метка/deferred-birth
- метка/tracklet-stitching
исход: неопределённый
направление: Собственная архитектура трекера
начат: '2026-09-30'
завершён: '2026-09-30'
---

# 🟡 Итоговый трекер линии B: каскад буферов IoU + отложенное рождение по q_birth + сшивка обрывков по геометрии

↑ [[02 Эксперименты/Направления/Собственная архитектура трекера|Собственная архитектура трекера]] · **Исход:** неопределённый · `line_b_combined_v1`

Модули, проверенные по отдельности (cbiou_cascade_v1, [birth_deferred_train_v1↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_deferred_train_v1), [stitch_appearance_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fstitch_appearance_v1)), впервые вместе. Трекеры: стенд; каскад 0,3→0,6; отложенное рождение N = 2; каскад + отложенное (tracking.motion_source_bytetrack.make_deferred_cascade — при нулевых буферах байт-в-байт равен DeferredBirthByteTrack, тест). Каждый — без сшивки и со сшивкой по геометрии (голова geometry, обученная на парах стенда 70 train-роликов, runs/fly_reid_ablation/models.pkl; жизни — непрерывные отрезки выхода трекера). Порог сшивки из {0,05; 0,1; 0,2; 0,3} — максимум IDF1 на dev для каждого трекера; отчёт — test (35) и val7; бутстреп по видео к стенду. scripts/eval_combined.py.

## Вопрос
Складываются ли эффекты каскада (HOTA/IDF1), отложенного рождения (IDSw) и сшивки (HOTA/IDF1/IDSw)?

## Гипотеза
Модули закрывают разные ошибки (ассоциация, ложные рождения, обрывы) и складываются почти независимо.

## Ожидалось (до прогона)
Каскад + отложенное + сшивка: HOTA test выше лучшего одиночного модуля (сшивка по геометрии, 59,283) и выше стенда с ДИ > 0; IDSw test ниже, чем у каждого модуля по отдельности (сшивка 1795).

## Что сделано
Карточка записана до прогона (30.09.2026, 19:3x MSK). Прогон 30.09 19:3x–22:04; первая версия сшивки брала жизни как любые непрерывные отрезки выхода (включая однокадровые) — втрое больше склеек, стенд + сшивка 59,029 (отчёт сохранён: runs/line_b_combined/report_minlen1.json); исправлено: минимальная длина жизни 2 кадра (как жизни глаза в build_fly_signatures), пересчёт сшивки 22:1x–22:3x.

## Получено
test (35): стенд: HOTA 58.469 / IDF1 66.965 / IDSw 2221 (val7 HOTA 54.573); стенд + сшивка: HOTA 59.125 / IDF1 68.743 / IDSw 1847 (val7 HOTA 55.449); каскад: HOTA 59.106 / IDF1 68.350 / IDSw 1968 (val7 HOTA 55.076); каскад + сшивка: HOTA 59.169 / IDF1 68.913 / IDSw 1795 (val7 HOTA 55.066); отложенное: HOTA 58.425 / IDF1 67.033 / IDSw 2072 (val7 HOTA 54.635); отложенное + сшивка: HOTA 59.144 / IDF1 68.873 / IDSw 1728 (val7 HOTA 55.365); каскад + отложенное: HOTA 59.139 / IDF1 68.524 / IDSw 1877 (val7 HOTA 55.217); каскад + отложенное + сшивка: HOTA 59.265 / IDF1 69.183 / IDSw 1718 (val7 HOTA 54.948). Бутстреп к стенду, каскад + отложенное + сшивка: ΔHOTA test +0.723 [+0.082; +1.443], ΔIDF1 +2.049 [+0.886; +3.360], ΔIDSw на ролик -14.37 [-18.51; -10.60] (лучше в 32 из 35); val7 ΔHOTA +0.030 [-0.753; +0.930].

## Интерпретация
По HOTA модули не складываются: все сборки дают +0,6…+0,8 к стенду, как каждый модуль по отдельности (потолок прироста HOTA при этом детекторе и этих модулях ≈ +0,8). По идентичности складываются: полная сборка даёт лучший IDF1 (+2,2 абс., +3,3 %) и наименьшие IDSw (−22,6 %, лучше в 32 из 35 роликов). Сшивка по жизням из выхода трекера немного слабее сшивки по жизням глаза (стенд + сшивка 59,125 против 59,283): определение жизни влияет на результат.

## Решение
Критерий 1 (HOTA test выше 59,283) не выполнен (59,265), критерий 2 (IDSw ниже 1795) выполнен (1718) → inconclusive. Итоговое ядро линии B для диссертации: каскад 0,3→0,6 + отложенное рождение N = 2 + сшивка по геометрии — HOTA +1,4 %, IDF1 +3,3 %, IDSw −22,6 % к стенду на test. Дальше HOTA упирается в детекцию (DetA), а не в ассоциацию.

## Ограничения
Голова сшивки обучена на выходах стенда, применяется к выходам других трекеров без переобучения; один сид.

## Нельзя заявлять
Новизну модулей: C-BIoU, AFLink-подобная сшивка — заимствования; своё — сборка, отложенное рождение по q_birth и протокол.

## Условия
- **baseline:** стенд XYWHByteTrack
- **variant:** каскад + отложенное рождение + сшивка по геометрии
- **dataset:** runs/train_cache (dev 35, test 35), runs/video_rate (val7)
- **split:** порог сшивки — dev; отчёт test и val7

## Критерии приёмки (записаны до прогона)
- ❌ каскад + отложенное + сшивка: HOTA test > 59,283 (сшивка по геометрии) и > стенда с ДИ > 0  
  *runs/line_b_combined/report.json → results.cascade_deferred.stitch.test.summary.HOTA = 59.265 < 59,283*
- ✅ IDSw test ниже 1795 (лучший одиночный модуль)  
  *runs/line_b_combined/report.json → results.cascade_deferred.stitch.test.summary.IDSw = 1718 < 1795*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 59.26548238685684 | 58.46862614225555 | test 35: каскад + отложенное + сшивка / стенд | `runs/line_b_combined/report.json` · results.cascade_deferred.stitch.test.summary.HOTA |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 69.18252317003973 | 66.965404625375 | test 35: каскад + отложенное + сшивка / стенд | `runs/line_b_combined/report.json` · results.cascade_deferred.stitch.test.summary.IDF1 |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 1718.0 | 2221.0 | test 35: каскад + отложенное + сшивка / стенд | `runs/line_b_combined/report.json` · results.cascade_deferred.stitch.test.summary.IDSw |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 54.94788368502137 | 54.57308976106422 | val7: каскад + отложенное + сшивка / стенд | `runs/line_b_combined/report.json` · results.cascade_deferred.stitch.val7.summary.HOTA |

## Доказательства
- 4 трекера × со сшивкой и без: `runs/line_b_combined/report.json` · results, bootstrap_vs_stand

## Код
- `scripts/eval_combined.py`
- `tracking/motion_source_bytetrack.py`
- `tests/test_motion_source_bytetrack.py`

## Данные
- `runs/line_b_combined`

