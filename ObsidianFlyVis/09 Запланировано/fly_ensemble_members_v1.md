---
aliases:
- 'Д5: держатся ли выводы о мухе на других моделях ансамбля FlyVis (012, 025, 037 против 000)'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fundamental-D5
- метка/flyvis
- метка/ensemble
- метка/robustness-of-claims
- метка/registry-H12
исход: идёт
направление: FlyVis
начат: '2026-10-04'
---

# 🔵 Д5: держатся ли выводы о мухе на других моделях ансамбля FlyVis (012, 025, 037 против 000)

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `fly_ensemble_members_v1`

Из research_fundamental_problems (F13): все выводы о мухе получены на одной модели ансамбля flow/0000/000 (лучшая по Sintel, 5,137). Повтор цепочки сигнатур повторного опознания (scripts/build_fly_signatures.py, сетчатка-ламина, те же 147 роликов, трекер и пары, что у [fly_signature_reid_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_signature_reid_v1) / [fly_connectome_null_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_connectome_null_v1)) на моделях 012 (5,278), 025 (5,301, медиана), 037 (5,333); лестница голов scripts/eval_fly_reid.py --ladder-only. Главная проверяемая заявка — вклад мухи поверх её входа (geometry + вход + муха − geometry + вход) и муха к геометрии.

## Вопрос
Воспроизводятся ли «муха к геометрии +0,14 AP» и «муха поверх входа +0,006» на других моделях ансамбля?

## Гипотеза
Знак обоих эффектов одинаков у всех 4 моделей; величина в пределах разброса между моделями.

## Ожидалось (до прогона)
Знак «geometry + вход + муха − geometry + вход» на test тот же у ≥ 3 из 3 новых моделей (с 000 — 4 из 4); «geometry + муха − geometry» > 0 у всех. Если знак расходится — заявки о мухе относятся к модели 000, а не к FlyVis.

## Условия
- **baseline:** модель 000 (fly_signature_reid_v1)
- **variant:** модели 012, 025, 037
- **dataset:** те же 147 роликов (train 130 → dev/test 35, val7), runs/fly_signatures_m{012,025,037}[_val]
- **split:** train / dev / test / val7

## Критерии приёмки (записаны до прогона)
- ⏳ test: знак (вход + муха) − вход совпадает у 012, 025, 037 с моделью 000 (> 0)
- ⏳ test: (геометрия + муха) − геометрия > 0 у всех трёх

## Доказательства
- постановка: [§7.7↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F7.7.%20%D0%A4%D1%83%D0%BD%D0%B4%D0%B0%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D0%BF%D1%80%D0%BE%D0%B1%D0%BB%D0%B5%D0%BC%D1%8B%20%D0%B8%20%D0%BF%D0%BE%D1%80%D1%8F%D0%B4%D0%BE%D0%BA%20%D0%B1%D0%BE%D1%80%D1%8C%D0%B1%D1%8B%2C%2003.10), Д5

## Код
- `scripts/build_fly_signatures.py`
- `scripts/run_bdd_video_rate_experiment.py`
- `scripts/eval_fly_reid.py`

## Данные
- `runs/fly_signatures_m012`
- `runs/fly_signatures_m025`
- `runs/fly_signatures_m037`

