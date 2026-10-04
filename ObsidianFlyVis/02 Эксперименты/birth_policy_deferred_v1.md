---
aliases:
- 'Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)'
tags:
- эксперимент
- исход/completed_positive
- направление/event_confidence
- метка/goal6
- метка/line-B
- метка/q_birth
- метка/birth-policy
- метка/tracker-level
- метка/deferred-confirmation
исход: положительный
направление: Раздельная событийная уверенность (линия B)
начат: '2026-09-17'
завершён: '2026-09-17'
---

# 🟢 Цель 6: отложенное подтверждение рождений с низким q_birth (N подтверждений вместо вето)

↑ [[02 Эксперименты/Направления/Раздельная событийная уверенность (линия B)|Раздельная событийная уверенность (линия B)]] · **Исход:** положительный · `birth_policy_deferred_v1`

Ответ на компромисс покрытия [[02 Эксперименты/birth_policy_qbirth_v1|birth_policy_qbirth_v1]]. Единственное изменение относительно стенда — правило подтверждения нового трека: рождение с q_birth ≥ τ подтверждается как в ByteTrack (первое повторное совпадение), рождение с q_birth < τ требует N последовательных повторных совпадений до публикации; любой пропуск до подтверждения удаляет трек (штатная семантика неподтверждённых). Ассоциация, пороги и буфер не тронуты.

## Вопрос
Сохраняется ли выигрыш вето по ложным рождениям, FP и IDSw при возврате покрытия (HOTA не ниже контроля − 0,1), если низкие рождения задерживать, а не запрещать?

## Гипотеза
Ложные рождения в основном короткоживущие (1–2 кадра), истинные объекты устойчивы; N-подтверждение отсекает первых почти без потери вторых ценой задержки ~N кадров (< 1 опорного кадра при N ≤ 5).

## Ожидалось (до прогона)
Ложные рождения и FP ниже контроля на каждом отложенном видео; IDSw не выше; HOTA и IDF1 не ниже контроля − 0,1; если диагностический контроль «N для всех» даёт тот же выигрыш, вклад головы не подтверждён.

## Что сделано
Карточка записана до кода (17.09.2026, 00:0x MSK). Реализован DeferredBirthByteTrack (tracking/birth_policy.py; 6 тестов: тождество при N=1, подавление двухкадрового мерцания), режим deferred в scripts/evaluate_birth_policy.py; рождение считается по первому опубликованному ID (одинаково для всех вариантов). Прогон пяти вариантов по 7 видео 17.09 00:1x–00:5x MSK (CPU). N выбран по обучающим видео: deferred_n3 (максимум HOTA на train при ложных рождениях ниже контроля).

## Получено
Отложенные видео, контроль: HOTA 52.675, AssA 48.310, IDF1 57.685, IDSw 310, FP 552, FN 2584; опубликованных ложных рождений 291. Выбранный на train N = 3: HOTA 53.041 (+0.366), AssA +0.735, IDF1 +0.846, IDSw 291 (-19), FP 455 (-97), FN 2660 (+76), Frag -9; ложные рождения 291 → 192, ниже на каждом видео; задержка первого правильного выхода +0,17–0,26 опорного кадра; объектов найдено 289 → 278 из 339. По видео HOTA: пересвет +1,27, темнота −0,19, сложный фон −0,28. N = 2: HOTA +0.395, IDF1 +0.895, IDSw -14, FP -63, FN +35. N = 5: HOTA +0,237. Диагностический контроль «N = 3 для всех рождений без головы»: HOTA -0.166, IDF1 +1.721, IDSw -120, FP -243, FN +308 — идентичность лучше, покрытие хуже, HOTA ниже контроля; на train HOTA −1,216.

## Интерпретация
Отложенное подтверждение, управляемое q_birth, даёт первый прирост на уровне трекера: HOTA +0,37, IDF1 +0,85, IDSw −19, FP −97 при FN +76. Безусловная задержка (диагностический контроль) выигрывает больше по идентичности (IDSw −120, IDF1 +1,72), но теряет покрытие (FN +308) и HOTA (−0,17): роль головы — селективность, то есть выбор, какие рождения задерживать. Прирост неоднороден: почти весь по HOTA — на видео с пересветом (+1,27); на темноте и сложном фоне HOTA в пределах −0,2…−0,3 при лучших FP/IDSw. Три видео — интервалы не считались; результат пилотный.

## Решение
Положительный по всем пяти записанным критериям (критерий 5 — по HOTA: селективная задержка выше безусловной; по IDF1/IDSw безусловная выше, что и ожидалось при потере покрытия). DeferredBirthByteTrack(τ_F1 = 0,327, N = 3) становится текущим лучшим вариантом линии B и контролем для следующих компонентов (q_match во втором проходе, q_alive для буфера). Число видео мало — независимая проверка остаётся обязательной.

## Ограничения
Три отложенных видео; метки рождений по опорным кадрам; τ и голова унаследованы. Прирост HOTA сосредоточен на видео с пересветом; бутстрап по видео не считался (три видео); голова и τ унаследованы без пересмотра.

## Нельзя заявлять
Новизну N-подтверждения (min_hits — стандартный механизм SORT/DeepSORT); собственный вклад — только условное применение по q_birth, если он подтвердится относительно диагностического контроля.

## Условия
- **baseline:** неизменённый XYWHByteTrack; диагностический контроль — задержка N для всех рождений
- **variant:** DeferredBirthByteTrack(τ_F1, N ∈ {2, 3, 5})
- **dataset:** 7 исходных val-видео
- **split:** N — по обучающим; метрики — отложенные
- **detector:** weights/baseline_seed42/best.pt
- **tracker:** XYWHByteTrack стенда + отложенное подтверждение

## Критерии приёмки (записаны до прогона)
- ✅ N = 1 воспроизводит стенд побайтово  
  *tests/test_birth_policy.py::test_deferred_with_one_confirmation_reproduces_frozen_tracker*
- ✅ ложные рождения и FP ниже контроля на каждом отложенном видео при выбранном N  
  *runs/birth_policy_deferred/report.json → summary.deferred_n3.acceptance.false_births_lower_every_video, fp_lower_every_video*
- ✅ IDSw не выше контроля на каждом отложенном видео  
  *runs/birth_policy_deferred/report.json → summary.deferred_n3.acceptance.idsw_not_higher_every_video*
- ✅ HOTA и IDF1 не ниже контроля − 0,1 на агрегате отложенных  
  *runs/birth_policy_deferred/report.json → summary.deferred_n3.delta_test: HOTA +0,366, IDF1 +0,846*
- ✅ выигрыш превосходит диагностический контроль «N для всех рождений» (вклад головы, не задержки)  
  *runs/birth_policy_deferred/report.json → variants.defer_all_n3.metrics_test.HOTA −0,166 против +0,366 у селективной; по IDF1/IDSw безусловная выше ценой FN +308 — селективность головы подтверждена по HOTA*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 53.04107492500015 | 52.67491393303034 | 3 отложенных видео, deferred N=3 (выбран на train) / контроль | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.HOTA |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 58.53156588584226 | 57.68539595064095 | то же | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.IDF1 |
| [AssA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAssA) | 49.04465899167915 | 48.30984716299449 | то же | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.AssA |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 291.0 | 310.0 | то же | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.IDSw |
| [FP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FFP) | 455 | 552 | то же | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.CLR_FP |
| [FN↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FFN) | 2660 | 2584 | то же (цена) | `runs/birth_policy_deferred/report.json` · variants.deferred_n3.metrics_test.CLR_FN |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 52.509117510893496 | 52.67491393303034 | диагностический контроль: N=3 для всех рождений без головы | `runs/birth_policy_deferred/report.json` · variants.defer_all_n3.metrics_test.HOTA |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 190.0 | 310.0 | диагностический контроль: N=3 для всех | `runs/birth_policy_deferred/report.json` · variants.defer_all_n3.metrics_test.IDSw |

## Доказательства
- один компонент за раз; парное сравнение; политика рождения не наследует порог продолжения: 3.7. Цель 6
- выбор N на train и разности к контролю: `runs/birth_policy_deferred/report.json` · summary, selected_on_train, train_hota
- 17.09 — отложенное подтверждение: Журнал изменений

## Код
- `tracking/birth_policy.py`
- `scripts/evaluate_birth_policy.py`
- `tests/test_birth_policy.py`

## Данные
- `runs/birth_policy_deferred`

