---
aliases:
- 'Муха с нормализацией контраста как источник движения в трекере: даёт ли HOTA'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P2
- метка/motion
- метка/front-end
- метка/hota
исход: идёт
направление: FlyVis
начат: '2026-10-06'
---

# 🔵 Муха с нормализацией контраста как источник движения в трекере: даёт ли HOTA

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `fly_input_norm_hota_v1`

Продолжение [fly_input_norm_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_input_norm_v1): поле мухи + lcn24 (усиление 3,0, выбрано по train, на val не подбирается) подмешивается в ByteTrack так же, как в [fly_gain_tracker_recheck_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_gain_tracker_recheck_v1) (scripts/evaluate_bytetrack_motion_injection.py, вес 1, все треки / только потерянные). Сравнение с ByteTrack без движения и с DIS на 7 видео val7 (детектор их не видел).

## Вопрос
Переходит ли выигрыш мухи + lcn24 в переносе рамки в HOTA трекера?

## Гипотеза
Поле, лучше нулевого движения по переносу рамки, не вредит трекеру и даёт прирост HOTA.

## Ожидалось (до прогона)
Среднее HOTA по val7: муха + lcn24 (все треки или только потерянные — выбор по val7 запрещён, сообщаются оба) не ниже ByteTrack без движения; прирост с ДИ > 0 (бутстрэп по видео) — положительный исход. Ниже без движения при ДИ < 0 — отрицательный.

## Условия
- **baseline:** ByteTrack без движения; DIS — потолок; прежняя муха (усиление 1,6) −0,24 HOTA
- **variant:** ByteTrack + поле мухи с нормализацией контраста
- **dataset:** BDD100K MOT val (val7)
- **split:** val7

## Критерии приёмки (записаны до прогона)
- ⏳ среднее HOTA по val7 (все треки) не ниже ByteTrack без движения
- ⏳ прирост HOTA (все треки или только потерянные) с ДИ > 0

## Доказательства
- выигрыш в переносе рамки и масштаб 3,0 по train: `runs/fly_scale_probe/real.json` · 2x3s1_lcn24

## Код
- `scripts/build_fly_lcn_motion.py`
- `scripts/evaluate_bytetrack_motion_injection.py`

## Данные
- `runs/video_rate/<id>/motion/flyvis_tiled_lcn24.npz`
- `runs/video_rate/<id>/bytetrack_motion_injection_gated.json`

