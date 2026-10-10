---
aliases:
- 'Н9/М1, ступень B: муха с сетчаткой-ламиной на обучающих видео — убирает ли ламина ночной провал мухи в голове надёжности (разбивка тёмные/светлые)'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/registry-H9
- метка/modernization-M1
- метка/retina
- метка/night
- метка/train-videos
- метка/registry-H2
- муха
исход: положительный
направление: FlyVis
начат: '2026-09-30'
завершён: '2026-09-30'
---

# 🟢 Н9/М1, ступень B: муха с сетчаткой-ламиной на обучающих видео — убирает ли ламина ночной провал мухи в голове надёжности (разбивка тёмные/светлые)

↑ [[01 Муха/Темы/Глаз и сетчатка-ламина|Глаз и сетчатка-ламина]] · **Исход:** положительный · `flyvis_retina_stageB_v1`

Фовеальный поток с ламиной (быстрая версия: гладкие поля на кадре ×1/4) на обучающих видео (runs/train_cache_retina_lamina4) и val7 (runs/flyvis_fovea_flow_retina_lamina4) — сторож scripts/run_fovea_jobs.py с межпроцессной очередью к GPU. Лестница Н2 (geometry → +DIS → +муха → +оба; перемешанный контроль) обучается на train-роликах, отчёт — test и val7 с разбивкой test по средней яркости ролика (< 60/255 — тёмные). Парное сравнение «без сетчатки / ламина» на одних и тех же роликах и наблюдениях (τ_e = 0,0903 из [motion_reliability_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fmotion_reliability_v1)).

## Вопрос
Устраняет ли сетчатка-ламина провал мухи на тёмных роликах, обнаруженный стратификацией [motion_reliability_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fmotion_reliability_v1) (муха без DIS хуже геометрии на тёмных)?

## Гипотеза
Провал вызван входом вне обучающего распределения; с ламиной голова «геометрия + муха» на тёмных не хуже геометрии, а «геометрия + DIS + муха» не хуже «геометрия + DIS».

## Ожидалось (до прогона)
На тёмных test-роликах: geometry_fly(ламина) ≥ geometry и > geometry_fly(без сетчатки); geometry_dis_fly(ламина) ≥ geometry_dis и > перемешанного контроля. Критерии следуют из ступени A и стратификации [motion_reliability_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fmotion_reliability_v1) (без сетчатки на тёмных dev+test: fly 0,447 < geometry 0,450; dis+fly 0,515 < dis 0,522).

## Что сделано
Карточка записана 30.09.2026, 09:1x MSK по предварительному расчёту (119/26 роликов, runs/retina_frontend/stageB_partial.json, scripts/evaluate_retina_stageB_partial.py); критерии не менялись. Ступень B закончена 30.09 11:25 (+3 ролика 14:08): 200/200 обучающих + val7. Полный пересчёт 14:1x: scripts/evaluate_motion_reliability.py --train-flow runs/train_cache_retina_lamina4 --val-flow runs/flyvis_fovea_flow_retina_lamina4 → runs/motion_reliability_lamina/report.json; эталон без сетчатки тем же скриптом → runs/motion_reliability_none_strata/report.json; бутстреп по видео внутри подвыборок → runs/motion_reliability_lamina/strata_bootstrap.json.

## Получено
dev+test, тёмные (27) / светлые (43): geometry 0.4497 / 0.4535; +муха без сетчатки 0.4471 / 0.4657; +муха-ламина 0.4659 / 0.4661; geometry+DIS 0.5215 / 0.5430; +DIS+муха без сетчатки 0.5148 / 0.5420; +DIS+муха-ламина 0.5242 / 0.5422; перемешанный (ламина) 0.5197 / 0.5422. Бутстреп по видео (тёмные): без сетчатки DIS+муха − DIS -0.0066 [-0.0109; -0.0023] (хуже в 20 из 27); ламина: муха − геометрия +0.0161 [+0.0036; +0.0294]; DIS+муха − DIS +0.0028 [-0.0007; +0.0064]; DIS+муха − перемешанный +0.0046 [+0.0011; +0.0082]; ламина − без сетчатки (голова DIS+муха) +0.0094 [+0.0049; +0.0142]; то же на светлых +0.0003 [-0.0010; +0.0015]. Агрегат test / val7 (ламина): geometry+DIS 0.5490 / 0.4818, +муха 0.5482 / 0.4840; geometry+муха 0.4692 / 0.3853 (без сетчатки 0.4674 / 0.3692).

## Интерпретация
Без сетчатки муха на тёмных роликах значимо портит голову надёжности поверх DIS (вход вне обучающего распределения: шум сжатия принимается за движение). Сетчатка-ламина (фоторецептор Вебера + центр-окружение) устраняет это: на тёмных роликах голова с мухой-ламиной значимо лучше головы с мухой без сетчатки (+0,009 AP) и значимо лучше перемешанного контроля; сама по себе муха-ламина на тёмных даёт +0,016 к геометрии. На светлых роликах сетчатка ничего не меняет. Сверх DIS муха-ламина даёт прибавку только в тёмных сценах (+0,003, ДИ задевает ноль); на агрегате test — ноль. Это первый результат, где у мухи есть свойство, зависящее от условий объекта исследования: её полезность ночью определяется биологической предобработкой входа.

## Решение
Оба заранее записанных критерия выполнены на полном наборе → completed_positive (узкая постановка: ламина устраняет ночной провал мухи). Прибавка сверх DIS на тёмных — на грани значимости; на агрегате и в трекере (e2e_motion_source_v1) — ноль. Дальше по мухе: функции без аналога у DIS (Н11) и сетчатка как обязательная часть любого следующего опыта с мухой. Контроль «тот же вход для DIS» (dis_retina_control_v1): DIS от сетчатки не выигрывает (+0,001 ночью, ДИ через ноль) — выигрыш специфичен для мухи.

## Ограничения
Тёмные ролики — 27 (порог яркости 60/255 задан заранее по [motion_reliability_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fmotion_reliability_v1)); прибавка сверх DIS на тёмных +0,003 с ДИ, задевающим ноль; один сид перестановки; порог события τ_e одинаков для обоих режимов.

## Нельзя заявлять
Прирост мухи сверх DIS; пользу для HOTA (см. [e2e_motion_source_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fe2e_motion_source_v1): ноль). Поправка 03.10 (F13): результат получен на одной модели ансамбля FlyVis (flow/0000/000); интервалы учитывают разброс видео, но не моделей ансамбля.

## Прежние версии (до исправлений)
- **Получено**, 2026-09-30 (c260d13) — восстановлено из истории git 10.10: текст заменён после итога; причина правки — в журнале плана за эту дату
  > Тёмные test (8) / светлые (18) / val7: geometry 0.4749 / 0.4709 / 0.3433; +муха без сетчатки 0.4624 / 0.4775 / 0.3694; +муха-ламина 0.4755 / 0.4782 / 0.3854; geometry+DIS 0.5208 / 0.5546 / 0.4819; +DIS+муха без сетчатки 0.5149 / 0.5519 / 0.4826; +DIS+муха-ламина 0.5219 / 0.5518 / 0.4838; перемешанный контроль (ламина) 0.5181 / 0.5539 / 0.4803.
- **Интерпретация**, 2026-09-30 (c260d13) — восстановлено из истории git 10.10: текст заменён после итога; причина правки — в журнале плана за эту дату
  > Ночной провал мухи устранён входом: без сетчатки добавление мухи ухудшало голову на тёмных роликах (−0,013 без DIS, −0,006 поверх DIS), с ламиной — не ухудшает (+0,001 и +0,001) и выше перемешанного контроля; на val7 прибавка мухи без DIS выросла с +0,026 до +0,042. Сверх DIS муха почти ничего не добавляет ни в тёмных, ни в светлых сценах: сигнал надёжности мухи с ламиной достоверен, но избыточен относительно рассогласования DIS.
- **Решение**, 2026-09-30 (c260d13) — восстановлено из истории git 10.10: текст заменён после итога; причина правки — в журнале плана за эту дату
  > Оба критерия выполнены (на предварительном наборе) → completed_positive в узкой постановке «ламина устраняет ночной провал». Прироста сверх DIS нет — Н2 остаётся закрытой; вклад мухи в трекер искать в функциях, где у DIS нет аналога (Н11: эго-движение, лоуминг, сигнатура состояния). Полный пересчёт после окончания ступени B.

## Условия
- **baseline:** та же лестница без сетчатки на тех же роликах
- **variant:** ламина
- **dataset:** runs/train_cache_retina_lamina4 (ламина), runs/train_cache (без сетчатки); val7 — runs/flyvis_fovea_flow_retina_lamina4 и runs/flyvis_fovea_flow
- **split:** train 130 → dev 35 + test 35 (27 тёмных, 43 светлых; порог средней яркости 60/255) и val7

## Критерии приёмки (записаны до прогона)
- ✅ тёмные test: geometry_fly(ламина) ≥ geometry и > geometry_fly(без сетчатки)  
  *runs/motion_reliability_lamina/report.json → strata.dark.geometry_fly = 0.4659 ≥ geometry 0.4497; runs/motion_reliability_none_strata/report.json → strata.dark.geometry_fly = 0.4471*
- ✅ тёмные test: geometry_dis_fly(ламина) ≥ geometry_dis и > перемешанного  
  *runs/motion_reliability_lamina/report.json → strata.dark.geometry_dis_fly = 0.5242 ≥ geometry_dis 0.5215 и > shuffled 0.5197; бутстреп strata_bootstrap.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.5242342972245125 | 0.5148270462426626 | тёмные dev+test (27): geometry+DIS+муха, ламина / без сетчатки | `runs/motion_reliability_lamina/report.json` · strata.dark.geometry_dis_fly |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.4658596834096219 | 0.4497132039440513 | тёмные dev+test (27): geometry+муха-ламина / geometry | `runs/motion_reliability_lamina/report.json` · strata.dark.geometry_fly |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.5242342972245125 | 0.5214502454533669 | тёмные dev+test (27): geometry+DIS+муха-ламина / geometry+DIS | `runs/motion_reliability_lamina/report.json` · strata.dark.geometry_dis_fly |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.5422218451185353 | 0.542968110857222 | светлые dev+test (43): geometry+DIS+муха-ламина / geometry+DIS | `runs/motion_reliability_lamina/report.json` · strata.bright.geometry_dis_fly |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.38533621625589337 | 0.36918534406515546 | val7: geometry+муха, ламина / без сетчатки | `runs/motion_reliability_lamina/report.json` · groups.geometry_fly.val7.average_precision |

## Доказательства
- лестница Н2 с ламиной: `runs/motion_reliability_lamina/report.json` · strata, groups
- эталон без сетчатки: `runs/motion_reliability_none_strata/report.json` · strata, groups
- парный бутстреп по тёмным роликам: `runs/motion_reliability_lamina/strata_bootstrap.json` · lamina-none|dark|geometry_dis_fly
- предварительный расчёт 09:1x: `runs/retina_frontend/stageB_partial.json` · none, lamina

## Код
- `scripts/evaluate_motion_reliability.py`
- `scripts/run_fovea_jobs.py`
- `tracking/retina.py`
- `tracking/gpu_lock.py`

## Данные
- `runs/flyvis_fovea_flow_retina_lamina4`
- `runs/motion_reliability_lamina`
- `runs/motion_reliability_none_strata`
- `runs/retina_frontend`
- `runs/train_cache_retina_lamina4`

