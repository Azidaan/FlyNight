---
aliases:
- 'Н2: голова достоверности движения — предсказывает ли остаточный поток фовеального глаза FlyVis большую ошибку предсказания фильтра'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/registry-H2
- метка/goal3
- метка/reliability
- метка/foveation
- метка/learned-head
- метка/train-videos
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-09-18'
завершён: '2026-09-18'
---

# 🟢 Н2: голова достоверности движения — предсказывает ли остаточный поток фовеального глаза FlyVis большую ошибку предсказания фильтра

↑ [[01 Муха/Темы/Движение мухи|Движение мухи]] · **Исход:** отрицательный · `motion_reliability_v1`

Первая обучаемая голова на обучающих видео. Событие: ошибка центра предсказания фильтра на опорном кадре, нормированная на sqrt(площади GT-рамки), выше порога τ_e; τ_e выбирается на train-сплите так, чтобы положительных было ≈25 % (правило, не подбор на тесте). Лестница логистических голов: geometry (размер и положение предсказанной рамки, сторона вырезки) → + DIS-рассогласование (|центр DIS-переноса − центр предсказания|) → + FlyVis (|δ|, δx, δy, средний |поток| в рамке, доля колонок внутри) → + оба; контроль shuffled FlyVis (перестановка внутри видео). Обучение — train (130), отбор ничего не требует (порог по F1 на train), отчёт — dev (35), test (35) и val-семёрка отдельно.

## Вопрос
Несёт ли остаточный поток фовеального глаза информацию о ненадёжности предсказания положения объекта сверх геометрии и DIS-рассогласования?

## Гипотеза
Сигнал ρ = 0,24 из Н1 обучаем и переносится на невиданные видео.

## Ожидалось (до прогона)
AP головы с FlyVis выше головы geometry + DIS на dev, test и val-семёрке; выше shuffled; Brier не хуже. Если вклад есть только на train — переобучение, результат отрицательный.

## Что сделано
Карточка записана до обучения (18.09.2026, 08:5x MSK). scripts/evaluate_motion_reliability.py; прогон 18.09 08:4x–08:5x MSK: train 201 301 наблюдений (130 видео), dev 51 612 (35), test 54 510 (35), val7 11 904; τ_e = 0,0903, доля положительных 0,25/0,23/0,26/0,23.

## Получено
AP test / val7: geometry 0.4478 / 0.3432; geometry+FlyVis 0.4674 / 0.3692; geometry+DIS 0.5490 / 0.4818; geometry+DIS+FlyVis 0.5482 / 0.4825; geometry+DIS+shuffled 0.5478 / 0.4797. Бутстрап по видео на test (FlyVis над geometry+DIS): ДИ через ноль. Brier с FlyVis не хуже; ECE на test хуже (0,0221 против 0,0212).

## Интерпретация
Остаточный поток фовеального глаза несёт обучаемый и переносимый сигнал надёжности (+0,02 AP над геометрией на test и val7), но DIS-рассогласование (расхождение классического потока с предсказанием фильтра) несёт тот же сигнал сильнее (+0,10), и поверх него FlyVis неотличим от перемешанного контроля. Это первый обучаемый результат на обучающих видео (201 тыс. наблюдений) — вывод больше не ограничен семью роликами.

## Решение
Отрицательный по критериям 1, 3 и по интервалу; выполнены 2 и «выше shuffled» на агрегате. Реестр: Н2 в постановке «надёжность через остаточный поток» закрыта — сигнал есть, но избыточен относительно DIS. Открытые пункты по мухе: Н7 обучаемое считывание на полной активности (кэши активности на train-видео ещё не строились), Н5 динамика, цель 5 собственное ядро. Линия B: DIS-рассогласование — новый сильный признак надёжности для q_alive/буфера потерянных (карточка отдельно).

## Ограничения
Истина только на опорных кадрах; событие определено через ошибку центра, не всей рамки; один сид логистической регрессии (детерминирована).

## Нельзя заявлять
Пользу для HOTA до встраивания в буфер/назначение.

## Условия
- **baseline:** geometry + DIS-рассогласование
- **variant:** + FlyVis остаточный поток
- **dataset:** runs/train_cache (200 роликов, fovea_flow + alignment) и runs/flyvis_fovea_flow (val-семёрка)
- **split:** train 130 / dev 35 / test 35 (цель 1а) + val 7

## Критерии приёмки (записаны до прогона)
- ❌ AP (+FlyVis) выше geometry+DIS на dev, на test и на val-семёрке; Brier не хуже  
  *runs/motion_reliability/report.json → acceptance.ap_up_dev_test_val7 = false (test −0,0008), brier_not_worse = true*
- ✅ прирост выше shuffled FlyVis на test и val-семёрке  
  *runs/motion_reliability/report.json → acceptance.beats_shuffled_test_val7 = true (на агрегате; ДИ через ноль)*
- ❌ калибровка (ECE) головы с FlyVis не хуже базы на test  
  *runs/motion_reliability/report.json → acceptance.ece_not_worse_test = false*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.548166823200043 | 0.5489989730539304 | test (35 видео): +FlyVis / geometry+DIS | `runs/motion_reliability/report.json` · groups.geometry_dis_fly.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.48245866061014603 | 0.4817980315745724 | val7: +FlyVis / geometry+DIS | `runs/motion_reliability/report.json` · groups.geometry_dis_fly.val7.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.46735389077424755 | 0.44780890376193105 | test: +FlyVis / geometry — сигнал есть | `runs/motion_reliability/report.json` · groups.geometry_fly.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.5489989730539304 | 0.44780890376193105 | test: +DIS / geometry | `runs/motion_reliability/report.json` · groups.geometry_dis.test.average_precision |

## Доказательства
- Н2 — обученная голова достоверности; Н1 дал ρ = 0,24: 1.7. Реестр
- лестница голов на train/dev/test/val7: `runs/motion_reliability/report.json` · groups, comparisons, acceptance

## Код
- `scripts/evaluate_motion_reliability.py`

## Данные
- `runs/motion_reliability`

