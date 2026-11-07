---
aliases:
- Отложенное рождение по q_birth на обучающих видео (130/35/35) и его совмещение с буфером IoU
tags:
- эксперимент
- исход/inconclusive
- направление/event_confidence
- метка/line-B
- метка/q_birth
- метка/deferred-birth
- метка/train-videos
- метка/end-to-end
- метка/c-biou
- проверка-гипотезы/завершена
исход: неопределённый
направление: Раздельная событийная уверенность (линия B)
начат: '2026-09-30'
завершён: '2026-09-30'
---

# 🟢 Отложенное рождение по q_birth на обучающих видео (130/35/35) и его совмещение с буфером IoU

↑ [[02 Эксперименты/Направления/Раздельная событийная уверенность (линия B)|Раздельная событийная уверенность (линия B)]] · **Исход:** неопределённый · `birth_deferred_train_v1`

Перенос положительного результата [birth_policy_deferred_v1↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbirth_policy_deferred_v1) (+0,37 HOTA на 3 отложенных val-видео, голова на 4 val-видео) на обучающие видео цели 1а: голова q_birth (StandardScaler + LogisticRegression, base + DIS) на 156 193 кандидатах 130 train-роликов (runs/event_dataset_train), τ = максимум F1 на train (0,329; на val-четвёрке было 0,327). N ∈ {2, 3, 5} выбирается на dev по HOTA; отчёт — test и val7; варианты без буфера и с буфером IoU 0,5. Эталоны — стенд и буфер 0,5 из [e2e_motion_source_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fe2e_motion_source_v1) (тот же трекер).

## Вопрос
Обобщается ли прирост HOTA от отложенного рождения с 3 видео на 35 отложенных обучающих роликов и складывается ли он с буфером IoU?

## Гипотеза
Отложенное рождение снижает ложные рождения и IDSw без потери HOTA и складывается с буфером.

## Ожидалось (до прогона)
Выбранный на dev вариант: HOTA test не ниже стенда и IDSw ниже; с буфером — HOTA test выше одного буфера. На dev (до теста) выбранный N = 2 дал HOTA 61,317 против 61,336 у стенда (−0,02) и IDSw 1865 против 1990; с буфером 61,736 против 61,766 у одного буфера.

## Что сделано
Карточка записана после выбора на dev, до прогона test/val7 (30.09.2026, 03:5x MSK). scripts/eval_birth_train.py, runs/event_dataset_train (165 роликов), runs/birth_train/report_dev.json.

## Получено
test: стенд HOTA 58.469 / IDF1 66.965 / IDSw 2221; deferred_n2 58.425 / 67.033 / 2072; буфер 0,5 59.121 / 68.307 / 2034; deferred_n2+буфер 59.144 / 68.402 / 1929. val7: стенд 54.573 / IDSw 526; deferred_n2 54.635 / 504; буфер 54.680 / 507; deferred_n2+буфер 54.802 / 475. Бутстреп по видео (runs/birth_train/bootstrap_final.json): deferred_n2 − стенд, test ΔHOTA -0.083 ДИ [-0.179; +0.008], ΔIDSw на видео -4.26 ДИ [-5.69; -2.97], лучше/хуже по IDSw 29/0; (deferred_n2+буфер) − буфер, test ΔHOTA ДИ [-0.151; +0.067], IDSw лучше/хуже 26/4; (deferred_n2+буфер) − стенд, test ΔHOTA +0.717 ДИ [+0.205; +1.464], ΔIDF1 ДИ [+0.697; +2.956].

## Интерпретация
На 35 отложенных обучающих роликах отложенное рождение не меняет HOTA (прирост +0,37 на трёх val-видео не обобщился), но значимо и почти без исключений снижает число переключений ID (29 из 35 роликов лучше, ни одного хуже). Эффекты буфера IoU (HOTA/IDF1) и отложенного рождения (IDSw) почти независимы: сборка даёт HOTA +0,68 и IDF1 +1,44 к стенду с ДИ > 0 и IDSw −13 %.

## Решение
Критерий 1 (HOTA ≥ стенда и IDSw ниже) не выполнен строго: HOTA −0,04 (ДИ через ноль), IDSw ниже значимо; критерий 2 выполнен формально (+0,02, ДИ через ноль) → inconclusive по HOTA, положительный по IDSw. Сборка «буфер 0,5 + отложенное рождение N = 2» — текущий лучший трекер линии B на обучающих видео по IDSw; по HOTA равна буферу.

## Ограничения
Один сид; признаки рождения без мухи; истина на опорных кадрах 5 Гц.

## Нельзя заявлять
Перенос на другие детекторы.

## Условия
- **baseline:** стенд XYWHByteTrack и буфер 0,5 (runs/e2e_motion_source)
- **variant:** deferred_n2; deferred_n2 + буфер 0,5
- **dataset:** runs/train_cache + runs/event_dataset_train; val7 — runs/video_rate
- **split:** голова — train 130; выбор N — dev 35; отчёт — test 35 и val7

## Критерии приёмки (записаны до прогона)
- ❌ deferred_n2: HOTA test ≥ стенда и IDSw test ниже стенда  
  *runs/birth_train/report_final.json → summary.deferred_n2.test.HOTA = 58.425 < стенд 58.469; IDSw 2072 < 2221*
- ✅ deferred_n2 + буфер 0,5: HOTA test выше буфера 0,5  
  *runs/birth_train/report_final.json → summary.deferred_n2+buf0.5.test.HOTA = 59.144 > буфер 59.121 (ДИ через ноль)*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.42513978145013 | 58.46862614225555 | test 35: deferred_n2 / стенд | `runs/birth_train/report_final.json` · summary.deferred_n2.test.HOTA |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 2072.0 | 2221.0 | test 35: deferred_n2 / стенд | `runs/birth_train/report_final.json` · summary.deferred_n2.test.IDSw |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 59.14396772548896 | 58.46862614225555 | test 35: deferred_n2 + буфер 0,5 / стенд | `runs/birth_train/report_final.json` · summary.deferred_n2+buf0.5.test.HOTA |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 68.40150386177785 | 66.965404625375 | test 35: deferred_n2 + буфер 0,5 / стенд | `runs/birth_train/report_final.json` · summary.deferred_n2+buf0.5.test.IDF1 |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 1929.0 | 2221.0 | test 35: deferred_n2 + буфер 0,5 / стенд | `runs/birth_train/report_final.json` · summary.deferred_n2+buf0.5.test.IDSw |

## Доказательства
- выбор N на dev: `runs/birth_train/report_dev.json` · summary

## Код
- `scripts/eval_birth_train.py`
- `tracking/motion_source_bytetrack.py`
- `tracking/birth_policy.py`

## Данные
- `runs/birth_train`
- `runs/event_dataset_train`

