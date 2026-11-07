---
aliases:
- ByteTrack против BoT-SORT и зоопарк из одиннадцати трекеров
tags:
- эксперимент
- исход/completed_positive
- направление/zoo
- метка/zoo
- метка/boxmot
- метка/hybridsort
- метка/botsort
- метка/oracle
исход: положительный
направление: ByteTrack, BoT-SORT и зоопарк трекеров
начат: '2026-09-07'
завершён: '2026-09-10'
---

# 🟢 ByteTrack против BoT-SORT и зоопарк из одиннадцати трекеров

↑ [[02 Эксперименты/Направления/ByteTrack, BoT-SORT и зоопарк трекеров|ByteTrack, BoT-SORT и зоопарк трекеров]] · **Исход:** положительный · `tracker_zoo`

Готовые трекеры на одном кэше детекций без ReID; точка отсчёта сменилась дважды; цена детектора отделена от цены ассоциации.

## Вопрос
Какой готовый трекер — честная база, и сколько стоит ассоциация по сравнению с детектором?

## Ожидалось (до прогона)
Ранжирование трекеров при равном входе; запас ассоциации по оракулу; вклад детектора и подмножества по отдельности.

## Что сделано
Гейт 0б — оракул ассоциации; гейт 0в — 6 трекеров ultralytics и 5 из BoxMOT на 50 видео (640); зоопарк на полном val 960 дважды (авторские настройки и единый порог 0.25); проверка доменных настроек BoostTrack.

## Получено
Потолок ассоциации +23.0 IDF1 (AssA 49.79 → 73.40). На 640: HybridSORT +11.7% HOTA к ByteTrack; BoostTrack без фильтра формы 43.20 HOTA против 37.34 с ним (фильтр отбрасывал 21.6% объектов). На 960 и полном val впереди BoT-SORT (mMOTA 26.78); TrackTrack при едином пороге переезжает с последнего места на второе. Замена детектора стоит 24.13 mMOTA, разброс алгоритмов при равных точках — 5.78.

## Интерпретация
Детектор дороже всей ассоциации в 2.9–4.2 раза (положение П5). Точка отсчёта требует переизмерения: HybridSORT на 960 не прогонялся.

## Решение
До переизмерения точкой отсчёта считать BoT-SORT; правило 6.11 — у чужого метода сначала проверять доменные предположения.

## Ограничения
Прирост IDF1 у оракула частично объясняется тем, что он публикует все детекции; чистая величина — по AssA. Числа с 640/50 видео и с 960/полного val нельзя ставить в одну строку.

## Нельзя заявлять
«Впервые разделён вклад детектора и ассоциации» — снято (Detection-aware MOT evaluation, AVSS 2022). Наше — отношение цены детектора к разбросу алгоритмов на замороженном кэше.

## Условия
- **baseline:** ByteTrack (ultralytics)
- **variant:** HybridSORT, BoT-SORT, SFSORT, TrackTrack, BoostTrack(+), OC-SORT, Deep OC-SORT, FastTracker, OccluBoost
- **dataset:** BDD100K
- **split:** val 50 видео (640); полный val (960)
- **detector:** YOLO26s seed42
- **tracker:** 11 трекеров (ultralytics 8.4.84, BoxMOT 23.0.0)

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [AssA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAssA) | 73.3959 | 49.789 | оракул ассоциации / ByteTrack, TrackEval | `experiments/results/assoc_oracle.csv` · источник=trackeval;метрика=AssA;col=значение |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 74.0239 | 51.014 | оракул ассоциации / ByteTrack, bdd100k | `experiments/results/assoc_oracle.csv` · источник=bdd100k;метрика=IDF1;col=значение |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 42.7737 | None | ByteTrack, 50 видео 640 | `experiments/results/tracker_zoo.csv` · трекер=bytetrack;метрика=HOTA;col=значение |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 43.2012 | 37.344 | BoostTrack без фильтра формы / с фильтром | `experiments/results/boxmot_zoo.csv` · трекер=boosttrack;метрика=HOTA;col=значение |
| [mMOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FmMOTA) | 26.78 | 21.07 | BoT-SORT / ByteTrack, полный val 960 | `dissertation/ЕДИНЫЙ_ПЛАН.md` ·  |

## Доказательства
- гейты 0б и 0в, находка о фильтре формы: [§2.8↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F2.8.%2007%E2%80%9308.09%20%D0%B7%D0%B0%D0%BF%D0%B0%D1%81%20%D0%B0%D1%81%D1%81%D0%BE%D1%86%D0%B8%D0%B0%D1%86%D0%B8%D0%B8%2C%20%D0%B7%D0%BE%D0%BE%D0%BF%D0%B0%D1%80%D0%BA%20%D1%82%D1%80%D0%B5%D0%BA%D0%B5%D1%80%D0%BE%D0%B2%2C%20%D0%BF%D0%BE%D0%BB%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8)
- таблица одиннадцати трекеров: [§3.6↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F3.6.%20%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20%D1%82%D1%80%D0%B5%D0%BA%D0%B5%D1%80%D0%BE%D0%B2%20%D0%BD%D0%B0%20%D0%BE%D0%B4%D0%BD%D0%BE%D0%BC%20%D0%BA%D1%8D%D1%88%D0%B5%20%D0%B4%D0%B5%D1%82%D0%B5%D0%BA%D1%86%D0%B8%D0%B9%20%2808.09%29)
- зоопарк на полном val, разделение вкладов: [§3.9↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F3.9.%20%D0%98%D0%B7%D0%BC%D0%B5%D1%80%D0%B5%D0%BD%D0%B8%D1%8F%2009%E2%80%9311.09%20%28%D1%81%D1%82%D0%B5%D0%BD%D0%B4%20960%2C%20%D0%BF%D0%BE%D0%BB%D0%BD%D1%8B%D0%B9%20val%20%D0%B3%D0%B4%D0%B5%20%D1%83%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%BE%29)
- метрики BoxMOT: `experiments/results/boxmot_zoo.csv` · 
- метрики ultralytics: `experiments/results/tracker_zoo.csv` · 

## Код
- `scripts/tracker_zoo.py`
- `scripts/boxmot_zoo.py`
- `scripts/assoc_oracle.py`
- `evaluation/hota_eval.py`

## Данные
- `runs/tracker_zoo_960_full`
- `runs/tracker_zoo_960_tau025`
- `runs/boxmot_zoo`
- `runs/assoc_oracle`

## Коммиты
- `932adf6`
- `4b673d4`
- `7c433a7`
- `e485dd8`

