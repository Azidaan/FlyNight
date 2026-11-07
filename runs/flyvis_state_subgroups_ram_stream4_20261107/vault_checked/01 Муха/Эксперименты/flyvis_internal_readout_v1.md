---
aliases:
- 'Цель 2: считывание внутренней активности FlyVis (T4/T5, 34 типа) для q_match против декодированного потока и shuffled'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/goal2
- метка/internal-activity
- метка/readout
- метка/q_match
- метка/shuffled-control
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-09-16'
завершён: '2026-09-16'
---

# 🟢 Цель 2: считывание внутренней активности FlyVis (T4/T5, 34 типа) для q_match против декодированного потока и shuffled

↑ [[01 Муха/Темы/Считывание активности и события|Считывание активности и события]] · **Исход:** отрицательный · `flyvis_internal_readout_v1`

Первый опыт линии A по новой постановке (План целей 1.2, 3.3): добавляют ли объектно-центрированные признаки внутренней активности замороженного FlyVis (tiled 2x3, коэффициент масштаба не нужен) информацию о правильной паре сверх base + DIS и сверх декодированного потока, и превосходит ли их вклад контроль shuffled. Кэш активности строится новым проходом сети по 7 видео (GPU); пары и разбиение — те же, что у [qmatch_30hz_pseudo↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo) и [qmatch_gate↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_gate) all_current.

## Вопрос
Содержит ли внутренняя активность FlyVis (а не её декодер потока) условную информацию о правильной паре «прошлая детекция — текущая детекция» на 30 Гц и 5 Гц сверх геометрии, confidence YOLO26 и DIS?

## Гипотеза
Декодер потока — узкое место переноса (обучен на Sintel); популяционный код направлений T4/T5 и активность Tm/TmY несут признак неоднозначности и надёжности движения, который декодер теряет. Проверяется, а не принимается.

## Ожидалось (до прогона)
Лестница вложенных моделей на одних и тех же парах и метках (adjacent_pseudo + all_current; отдельно reference + all_current как воспроизводимость 5 Гц): base + DIS; + FlyVis decoded tiled; + FlyVis T4T5; + FlyVis all34; + shuffled T4T5 (перестановка признаков считывания между парами внутри видео, фиксированный сид); + DIS + decoded + T4T5. Обучение и пороги только на 4 обучающих видео; парный бутстрап по 3 отложенным видео.

## Что сделано
Карточка и определение считывания записаны до кода (16.09.2026, 01:2x MSK). 16.09, 01:3x–02:1x: tracking/flyvis_activity.py (считывание, 11 тестов), хук last_activity/baseline_activity в tracking/flyvis_motion.py (38 регрессионных тестов зелёные), scripts/build_flyvis_activity_cache.py — кэш построен по 7 видео (1,9 ГБ, ~19 мс/кадр, GPU). Гейт scripts/evaluate_flyvis_readout_gate.py (4 теста) и аддитивное расширение evaluate_flyvis_qmatch_gate.py (рамки/кадры пар в pairs.npz, --pairs-only) готовы; Пары 30 Гц пересобраны с рамками (runs/qmatch_readout_gate/pairs_30hz, хэш признаков/меток/разбиения тождествен [qmatch_30hz_pseudo↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo)). Гейт считывания выполнен 16.09, 01:5x MSK (CPU): 250 275 пар, 7 вложенных логистических моделей, 5 парных бутстрапов по видео. Шаг сетки колонок 13,3 px.

## Получено
Отложенные видео. Основное сравнение base+DIS+T4T5 против base+DIS: AP 0.943199 против 0.945423 — ниже на каждом из трёх видео (средняя разность −0,002381, 95%-й интервал [−0,003868; −0,001499] целиком отрицательный, доля положительных ресемплингов 0,000); Brier 0.011526 против 0.011380 — хуже на каждом видео. T4T5 против shuffled T4T5: средняя разность AP +0,001022, интервал [−0,004194; +0,006119], доля положительных 0,62 — неотличимо от перемешанных признаков. T4T5 против декодированного потока: AP ниже на 0,004538, интервал целиком отрицательный — декодер не является узким местом, он извлекает больше, чем это считывание. all34 (154 признака) против base+DIS: ΔAP +0,000144, интервал через ноль, Brier хуже. base+T4T5 без DIS: AP 0.948671 — на уровне чистого base (0,949014 в [qmatch_30hz_pseudo↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo)), Brier худший. При порогах с обучающих видео T4T5 добавляет FP пар 1 713 → 1 960 при FN 755 → 620 — сдвиг рабочей точки, не селективность.

## Интерпретация
Объектно-центрированное считывание внутренней активности FlyVis (энергии направлений T4/T5, энтропия, доминирование, ON/OFF, кольцо; и широкое считывание 34 типов) при охвате tiled 2x3 на соседних кадрах 30 Гц не несёт условной информации о правильной паре сверх геометрии и DIS: результат неотличим от shuffled и хуже декодированного потока. Гипотеза «декодер — узкое место переноса» для этого охвата опровергнута. Не исключено, что колонки шагом 13,3 px слишком грубы для объектов BDD100K — это проверяет фовеация (следующая карточка); до неё точка A1 формально не срабатывает.

## Решение
Отрицательный по заранее записанным критериям 3–5. Считывание T4T5/all34 при охвате tiled в трекер не встраивается. Следующая и последняя карточка цели 2 по охвату — фовеация (то же считывание, вырезки вокруг рамок); её отрицательный результат означает точку остановки A1.

## Ограничения
Только охват tiled 2x3; фовеация — отдельная карточка (то же считывание, другой охват), без неё точка A1 не срабатывает формально. Псевдометки на 30 Гц, три отложенных видео. Стоимость: новый проход сети FlyVis по 7 857 кадрам на GPU и ~2,3 ГБ кэша активности в float16. Только линейная модель по протоколу; 5-Гц режим reference с этим считыванием не выполнен.

## Нельзя заявлять
Что внутренняя активность FlyVis бесполезна при любом охвате (фовеация не проверена) и для других событий (q_alive, q_birth, достоверность движения); что нелинейная модель не нашла бы сигнал (логистическая регрессия по протоколу карточки).

## Условия
- **baseline:** base + DIS на тех же парах; вторичный контроль — base + DIS + FlyVis decoded tiled
- **variant:** base + DIS + FlyVis T4T5 (основной); base + DIS + FlyVis all34 (широкий)
- **dataset:** 7 исходных val-видео ~30 Гц; тот же манифест, что bench_freeze_v1
- **split:** обучение: b1c66a42, b1c81faa, b1cac6a7, b1cebfb7; отложенные: b232c7c9 (пересвет), b1d22449 (темнота), b251064f (сложный фон) — как в [qmatch_30hz_pseudo↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo)
- **detector:** weights/baseline_seed42/best.pt (из манифеста bench_freeze_v1)

## Критерии приёмки (записаны до прогона)
- ✅ кэш активности построен для 7 видео в режиме tiled с тем же числом кадров, что кэш потока; признак зависит только от текущего и прошлых кадров (тест на отсутствие будущего)  
  *runs/flyvis_activity/summary.json: 7 видео, frame_count совпадает с кэшем потока (проверка в build_flyvis_activity_cache.py), декодированный поток нового прохода воспроизводит сохранённый с max |Δ| ≤ 0,0017 px; признак строится из estimator.last_activity текущего кадра (тест test_estimator_exposes_internal_activity_of_current_frame_only)*
- ✅ множество пар, метки и разбиение совпадают с [qmatch_30hz_pseudo↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fqmatch_30hz_pseudo) (хэш pairs)  
  *runs/qmatch_readout_gate/readout_30hz/report.json → identity.same_pairs_and_labels = true*
- ❌ основной: base+DIS+T4T5 против base+DIS — AP выше и Brier не хуже на каждом из трёх отложенных видео  
  *runs/qmatch_readout_gate/readout_30hz/report.json → acceptance.primary_ap_up_every_video = false, primary_brier_not_worse_every_video = false; comparisons.primary_t4t5_vs_base_dis*
- ❌ прирост T4T5 превосходит прирост shuffled T4T5 на каждом отложенном видео  
  *runs/qmatch_readout_gate/readout_30hz/report.json → acceptance.t4t5_beats_shuffled_every_video = false; comparisons.t4t5_vs_shuffled ДИ через ноль*
- ❌ T4T5 не хуже decoded tiled по AP на отложенных (иначе декодер не является узким местом)  
  *runs/qmatch_readout_gate/readout_30hz/report.json → acceptance.t4t5_not_worse_than_decoded_ap = false; comparisons.t4t5_vs_decoded*
- ⏳ подтверждение на уровне трекера: HOTA, AssA, IDSw

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9431989524551653 | 0.9454231150548597 | adjacent_pseudo + all_current, отложенные: base+DIS+T4T5 / base+DIS | `runs/qmatch_readout_gate/readout_30hz/report.json` · feature_groups.base_dis_t4t5.test.average_precision |
| [Brier↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FBrier%20score) | 0.011525959042269702 | 0.011379555146402502 | то же | `runs/qmatch_readout_gate/readout_30hz/report.json` · feature_groups.base_dis_t4t5.test.brier |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9408544820858769 | 0.9454231150548597 | контроль shuffled T4T5 / base+DIS | `runs/qmatch_readout_gate/readout_30hz/report.json` · feature_groups.base_dis_shuffled_t4t5.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9465781445169114 | 0.9454231150548597 | широкое считывание 34 типов / base+DIS | `runs/qmatch_readout_gate/readout_30hz/report.json` · feature_groups.base_dis_all34.test.average_precision |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.9483764694441175 | 0.9431989524551653 | декодированный поток / T4T5 (декодер не узкое место) | `runs/qmatch_readout_gate/readout_30hz/report.json` · feature_groups.base_dis_decoded_tiled.test.average_precision |

## Доказательства
- считывание определяется до опыта; внутренняя активность обязательна; контроль shuffled: 3.3. Цель 2
- точка A1 после цели 2 требует лучший вариант считывания и охвата: 1.6. Точки остановки линии A
- пять парных бутстрапов по видео: `runs/qmatch_readout_gate/readout_30hz/report.json` · comparisons
- кэш активности воспроизводит поток: `runs/flyvis_activity/summary.json` · *.flow_reproduction
- 16.09 — отрицательный гейт считывания: Журнал изменений

## Код
- `tracking/flyvis_activity.py`
- `tracking/flyvis_motion.py`
- `scripts/build_flyvis_activity_cache.py`
- `scripts/evaluate_flyvis_readout_gate.py`
- `scripts/evaluate_flyvis_qmatch_gate.py`
- `tests/test_flyvis_activity_readout.py`
- `tests/test_flyvis_readout_gate.py`

## Данные
- `runs/flyvis_activity`
- `runs/qmatch_readout_gate`

