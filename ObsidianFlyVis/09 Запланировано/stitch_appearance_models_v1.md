---
aliases:
- 'Сшивка обрывков без утечки разметки в кандидаты: внешность DINOv3-S против MASA-R50 и вклад мухи'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P1
- метка/tracker
- метка/association
исход: идёт
направление: FlyVis
начат: '2026-10-08'
---

# 🔵 Сшивка обрывков без утечки разметки в кандидаты: внешность DINOv3-S против MASA-R50 и вклад мухи

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `stitch_appearance_models_v1`

Решение автора 08.10: завершить опыт окна; исправить зависимость списка кандидатов от разметки (в eval_fly_reid.video_pairs пара допускалась, только если оба конца сопоставлены с эталоном — сшиватель видел только треки настоящих объектов, прежние числа сшивки завышены); пилот DINOv3-S и MASA-R50 — сравнить, что лучше (замена узла «внешность» по правилу 6.15). DINOv3 ViT-S/16 (timm vit_small_patch16_dinov3.lvd1689m, открытое зеркало, лицензия DINOv3) — вырез рамки 128², средний токен 384 (scripts/build_dino_embeddings.py). MASA-R50 (Li et al., CVPR 2024; веса HF dereksiyuanli/masa, Apache-2.0) — собран без mmdetection в tracking/masa_r50.py (ResNet-50 caffe + FPN + DeformFusion с чтением смещений как в mmcv + RoIAlign + голова QuasiDense 256); проверка: тот же трек через 5 кадров — сходство 0,955, разные — 0,103 (scripts/build_masa_embeddings.py). Оценка — scripts/stitch_eval_v2.py: все кандидаты окна (150; 5) без фильтра по эталону; головы на train 130, порог — по HOTA на dev 35, проверка test 35 и val7.

## Вопрос
Какая внешность лучше для сшивки обрывков — DINOv3-S или MASA-R50, сколько HOTA она даёт без утечки разметки, и добавляет ли муха сверх неё?

## Гипотеза
MASA-R50 (обучена сопоставлять объекты) лучше DINOv3-S (общие признаки) и обе лучше простой внешности; муха сверх MASA добавляет мало.

## Ожидалось (до прогона)
test: (1) лучшая из голов с DINO/MASA − голова геометрии > 0, ДИ > 0 (по HOTA сшивки); (2) MASA − DINO: знак и ДИ — что лучше; (3) муха сверх MASA (masa_fly − masa) > 0, ДИ > 0 — вклад мухи; val7 — тот же знак.

## Условия
- **baseline:** без сшивки; сшивка по геометрии; нынешняя голова (геометрия + миниатюра + вход глаза)
- **variant:** + DINOv3-S / + MASA-R50 / + муха
- **dataset:** BDD100K MOT train_cache (train 130 / dev 35 / test 35) + val7

## Критерии приёмки (записаны до прогона)
- ⏳ test: лучшая голова с DINO/MASA − геометрия > 0, ДИ > 0 (HOTA сшивки)
- ⏳ test: masa − dino — знак с ДИ (что лучше)
- ⏳ test: masa_fly − masa > 0, ДИ > 0

## Доказательства
- идеальная сшивка в окнах: `runs/fly_stitch_oracle/summary.json` · windows

## Код
- `scripts/stitch_eval_v2.py`
- `scripts/build_dino_embeddings.py`
- `scripts/build_masa_embeddings.py`
- `tracking/masa_r50.py`

## Данные
- `runs/stitch_v2`
- `runs/dino_embeddings`
- `runs/masa_embeddings`

