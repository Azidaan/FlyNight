---
aliases:
- 'Каркас: BDD100K, кэш детекций, три baseline-детектора'
tags:
- эксперимент
- исход/infrastructure
- направление/infra
- метка/bdd100k
- метка/baseline
- метка/cache
исход: инфраструктура
направление: Инфраструктура и данные
начат: '2026-06-01'
завершён: '2026-08-16'
---

# 🔧 Каркас: BDD100K, кэш детекций, три baseline-детектора

↑ [[02 Эксперименты/Направления/Инфраструктура и данные|Инфраструктура и данные]] · **Исход:** инфраструктура · `infra_bdd_baseline`

Собраны данные (1400 обучающих и 200 проверочных видео), обучены три базовых детектора, построены кэширование детекций и офлайн-ассоциация — инфраструктура, пережившая всё последующее.

## Ожидалось (до прогона)
Рабочий стенд: данные, обученный контроль, воспроизводимая офлайн-оценка трекинга.

## Что сделано
Импорт кода, разметка box_track_20, обучение baseline seed 42/123/2024 по 100 эпох, кэш детекций, автономный прогон конвейера с проверками.

## Получено
Baseline на 50 видео val (τ=0.25, буфер 30): IDF1 51.014, MOTA 40.024, IDSw 4393. Ночная mAP50 контроля 0.3772 ± 0.0031 по трём сидам.

## Решение
Стенд принят; baseline — единственная точка отсчёта работы, убирать нельзя.

## Ограничения
Дата начала — по [разделу 2.1↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F2.1.%20%D0%98%D1%8E%D0%BD%D1%8C%20%E2%80%94%20%D0%B8%D1%8E%D0%BB%D1%8C%20%D0%BA%D0%B0%D1%80%D0%BA%D0%B0%D1%81) плана («июнь — июль»), точнее не зафиксирована; первый коммит репозитория 25.07.

## Условия
- **baseline:** —
- **variant:** baseline seed42
- **dataset:** BDD100K MOT
- **split:** val, 50 видео
- **detector:** YOLO26s
- **tracker:** ByteTrack (ultralytics)
- **seeds:** [42, 123, 2024]
- **checkpoint:** weights/baseline_seed42/best.pt

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 51.0137 | None | 50 видео val, bdd100k | `experiments/results/gate_a_oracle.csv` · режим=baseline;метрика=IDF1;col=значение |
| [MOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FMOTA) | 40.0237 | None | 50 видео val | `experiments/results/gate_a_oracle.csv` · режим=baseline;метрика=MOTA;col=значение |
| [mAP50↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FmAP50) | 0.375729 | None | ночь, seed42 | `experiments/results/target_full_night.csv` · модель=baseline_seed42;поднабор=night;метрика=mAP50;col=значение |
| [mAP50↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FmAP50) | 0.375121 | None | ночь, seed123 | `experiments/results/target_full_night.csv` · модель=baseline_seed123;поднабор=night;метрика=mAP50;col=значение |
| [mAP50↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FmAP50) | 0.380805 | None | ночь, seed2024 | `experiments/results/target_full_night.csv` · модель=baseline_seed2024;поднабор=night;метрика=mAP50;col=значение |

## Доказательства
- каркас июнь–июль: [§2.1↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F2.1.%20%D0%98%D1%8E%D0%BD%D1%8C%20%E2%80%94%20%D0%B8%D1%8E%D0%BB%D1%8C%20%D0%BA%D0%B0%D1%80%D0%BA%D0%B0%D1%81)
- baseline IDF1 51.014, MOTA 40.024: [§3↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F3.%20%D0%92%D1%81%D0%B5%20%D0%B8%D0%B7%D0%BC%D0%B5%D1%80%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%87%D0%B8%D1%81%D0%BB%D0%B0)
- оценка baseline по поднаборам: `experiments/results/baseline_summary.csv` · 

## Код
- `tracking/offline_association.py`
- `evaluation/tracking_metrics.py`
- `training/train_baseline.py`
- `Запусти меня.py`

## Данные
- `runs/baseline`
- `runs/det_cache_ctl`

## Коммиты
- `5dab94b`
- `b1bc875`
- `27e1281`
- `f5d4a38`

