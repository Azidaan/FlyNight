---
aliases:
- 'Цель 2: фовеация — глаз FlyVis на каждый трек XYWHByteTrack, то же считывание T4/T5 и all34'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/goal2
- метка/foveation
- метка/readout
- метка/q_match
- метка/shuffled-control
- метка/stop-point-A1
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-09-16'
завершён: '2026-09-16'
---

# 🔴 Цель 2: фовеация — глаз FlyVis на каждый трек XYWHByteTrack, то же считывание T4/T5 и all34

↑ [[01 Муха/Темы/Считывание активности и события|Считывание активности и события]] · **Исход:** отрицательный · `flyvis_foveation_readout_v1`

Последняя карточка цели 2 по охвату. Считывание из [[01 Муха/Эксперименты/flyvis_internal_readout_v1|flyvis_internal_readout_v1]] не меняется; меняется только вход сети: вместо сетки 2x3 (шаг колонок 13,3 px) — вырезка вокруг предсказанной рамки каждого подтверждённого трека XYWHByteTrack (шаг колонок ~s/30 px при стороне вырезки s, т.е. 2–8 px для типичных объектов). Отрицательный результат по заранее записанным критериям означает точку остановки A1 (План целей 1.6).

## Вопрос
Несёт ли внутренняя активность FlyVis условную информацию о правильной паре, когда сеть смотрит на объект с разрешением, соответствующим его размеру, и с собственным рекуррентным состоянием на трек?

## Гипотеза
Провал tiled-считывания вызван грубым разрешением (13,3 px на колонку) и смешением объекта с фоном; фовеация устраняет обе причины. Проверяется, а не принимается.

## Ожидалось (до прогона)
Лестница на тех же парах: base+DIS+eye → +decoded tiled → +fovea T4T5 → +fovea all34 → +shuffled fovea T4T5 (перестановка строк внутри видео среди пар с глазом) → +tiled T4T5 (из [[01 Муха/Эксперименты/flyvis_internal_readout_v1|flyvis_internal_readout_v1]], для прямого сравнения охватов). Обучение и пороги на 4 обучающих видео, парный бутстрап по 3 отложенным.

## Что сделано
Карточка записана до кода (16.09.2026, 02:0x MSK). Глаза, кэш и гейт не реализованы; сеть не запускалась.

## Получено
Покрытие глазами: у положительных пар 0,78–0,94 по видео (сложный фон 0,779 < 0,8), у отрицательных 0,48–0,75. Отложенные видео, контроль base+DIS+has_eye AP 0.944843: +fovea T4T5 AP 0.945237 (Δ −0,000436, ДИ [−0,002667; +0,001196], по видео +0,0012/−0,0027/+0,0002), Brier 0.011516 против 0.011384 — хуже на каждом видео. Против shuffled fovea: ΔAP +0,002093, ДИ [+0,001009; +0,003450], положительно на каждом видео — признаки несут реальную информацию. Против tiled T4T5: +0,001308, ДИ [−0,000042; +0,002451] — фовеация лучше сетки. Против декодированного потока: −0,003430, ДИ [−0,005749; −0,000751]. all34: +0,001872, ДИ через ноль, Brier хуже. Нелинейная диагностика (HGB): base+DIS+eye AP 0,967989; все варианты FlyVis 0,9660–0,9671, decoded 0,967990 — ни один не выше базы.

## Интерпретация
Фовеальное считывание содержит информацию (отличимо от shuffled на каждом видео) и лучше сетки, но эта информация избыточна относительно геометрии пары и DIS и беднее декодированного потока; при сильной модели база насыщает задачу. Гипотеза о разрешении подтверждена лишь в части «фовеация > сетка», не в части «есть условный вклад». Лучший вариант охвата и считывания не даёт прироста над base + DIS, устойчивого по видео — условие точки A1 выполнено. Полнокадровое считывание не строилось: оно строго грубее сетки, которая уже проиграла.

## Решение
Отрицательный по критериям 2, 3, 5 (4 выполнен: лучше shuffled). ТОЧКА ОСТАНОВКИ A1 СРАБОТАЛА (План целей 1.6): линия A перестаёт быть кандидатом на основной метод; FlyVis закрывается как исследованная ветвь с отрицательным результатом; основной метод — линия B (XYWHByteTrack + раздельная событийная уверенность); цель 3, дообучение FlyVis и сборка линии A не начинаются. Изменение маршрута согласуется с пользователем (6.5).

## Ограничения
Глаза существуют только у подтверждённых треков: пары «новая детекция → новая детекция» без глаза оцениваются только контролем. Псевдометки 30 Гц, три отложенных видео. Если критерии 3–5 не выполнены — точка остановки A1: FlyVis закрывается как исследованная ветвь, основной становится линия B.

## Нельзя заявлять
Что FlyVis бесполезен для событий q_alive/q_birth или как источник достоверности движения (не проверялось); что иной дизайн глаза (контекст, fade-in, размер) дал бы иной знак; что 5-Гц режим повторил бы результат (не запускался для считывания).

## Условия
- **baseline:** base + DIS + has_eye на тех же парах
- **variant:** base + DIS + has_eye + fovea T4T5 (основной); + fovea all34 (широкий)
- **dataset:** 7 исходных val-видео ~30 Гц; пары runs/qmatch_readout_gate/pairs_30hz
- **split:** как в [[01 Муха/Эксперименты/qmatch_30hz_pseudo|qmatch_30hz_pseudo]]: 4 обучающих / 3 отложенных
- **detector:** weights/baseline_seed42/best.pt (манифест [bench_freeze_v1↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbench_freeze_v1))
- **tracker:** XYWHByteTrack h0.25/l0.1/b30/m0.8 — только как источник треков для глаз; его выходы совпадают с [bench_freeze_v1↗](obsidian://open?vault=ObsidianFlyVis&file=02%20%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fbench_freeze_v1)

## Критерии приёмки (записаны до прогона)
- ⏳ глаза детерминированы: повторный прогон одного видео даёт те же строки признаков; состояние глаза зависит только от кадров ≤ t; трекер не читает разметку
- ❌ доля пар с глазом (has_eye) отчётливо указана по видео и по классу пары; положительные пары покрыты не менее чем на 80%  
  *runs/qmatch_readout_gate/fovea_30hz/report.json → eye_coverage: положительные с глазом 0,779 на b251064f (< 0,8), остальные 0,85–0,94*
- ❌ основной: base+DIS+eye+fovea T4T5 против base+DIS+eye — AP выше и Brier не хуже на каждом из трёх отложенных видео  
  *runs/qmatch_readout_gate/fovea_30hz/report.json → acceptance.primary_ap_up_every_video = false, primary_brier_not_worse_every_video = false*
- ✅ прирост fovea T4T5 превосходит прирост shuffled fovea T4T5 на каждом отложенном видео  
  *runs/qmatch_readout_gate/fovea_30hz/report.json → acceptance.fovea_beats_shuffled_every_video = true; comparisons.fovea_vs_shuffled ДИ положительный*
- ❌ fovea T4T5 не хуже tiled T4T5 и декодированного потока по AP на отложенных  
  *runs/qmatch_readout_gate/fovea_30hz/report.json → fovea_not_worse_than_tiled_ap = true, fovea_not_worse_than_decoded_ap = false*
- ⏳ подтверждение на уровне трекера: HOTA, AssA, IDSw

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9452374402827249 | 0.9448432390892452 | 30 Гц, отложенные: base+DIS+eye+fovea T4T5 / base+DIS+eye | `runs/qmatch_readout_gate/fovea_30hz/report.json` · feature_groups.base_dis_eye_fovea_t4t5.test.average_precision |
| [Brier↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FBrier%20score) | 0.011515590404385153 | 0.011384469628060888 | то же | `runs/qmatch_readout_gate/fovea_30hz/report.json` · feature_groups.base_dis_eye_fovea_t4t5.test.brier |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9428629741037131 | 0.9452374402827249 | shuffled fovea / fovea — фовеа отличима от шума | `runs/qmatch_readout_gate/fovea_30hz/report.json` · feature_groups.base_dis_eye_shuffled_fovea.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9440523480853691 | 0.9452374402827249 | tiled T4T5 / fovea T4T5 — фовеа лучше сетки | `runs/qmatch_readout_gate/fovea_30hz/report.json` · feature_groups.base_dis_eye_tiled_t4t5.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9497669421022101 | 0.9452374402827249 | decoded / fovea — декодер богаче | `runs/qmatch_readout_gate/fovea_30hz/report.json` · feature_groups.base_dis_eye_decoded.test.average_precision |

## Доказательства
- фовеация — обязательная абляция охвата; размер вырезки фиксируется до опыта: 3.3. Цель 2
- A1 после лучшего варианта считывания и охвата: 1.6. Точки остановки линии A
- пять парных бутстрапов: `runs/qmatch_readout_gate/fovea_30hz/report.json` · comparisons
- нелинейная диагностика: `runs/qmatch_readout_gate/nonlinear_diag/report.json` · feature_groups
- кэш фовеальных считываний: `runs/flyvis_fovea/summary.json` · *
- 16.09 — точка A1: Журнал изменений

## Код
- `tracking/foveated_eyes.py`
- `scripts/build_foveated_readout_cache.py`
- `scripts/evaluate_flyvis_readout_gate.py`
- `tests/test_foveated_eyes.py`

## Данные
- `runs/flyvis_fovea`
- `runs/qmatch_readout_gate`

