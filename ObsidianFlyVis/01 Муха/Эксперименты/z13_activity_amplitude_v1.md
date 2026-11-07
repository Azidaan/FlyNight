---
aliases:
- Амплитуда активности внутри регионального считывателя FlyVis
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Амплитуда активности внутри регионального считывателя FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_activity_amplitude_v1`

Сохранение амплитуды: критерий не пройден.

## Вопрос
Мешает ли отдельная нормировка каждого вектора68признаков обучаемому выбору и отказу?

## Гипотеза
Source tracking/z13_filter_descriptor.py applies per-candidate LayerNorm before query embedding; this discards common shifts and approximately positive rescaling (finite epsilon) of its68channels. Current heads may lose response magnitude needed for refusal. This is a hypothesis about utility, not an established cause. Previous map-standard-deviation repair acted on a scalar response map, not raw68-vector normalization inside learned embedding.

## Ожидалось (до прогона)
Change only candidate activity embedding LayerNorm68 to a per-channel learnable affine transform (68gain+68bias, same initialization/capacity). Leave target-coefficient LayerNorm and fixed dataset-channel normalization intact. Same actual regional categorical action and signed good-harm labels as prior context readout. 4representations x2norm/affine x3seeds=24models, same initial parameter tensors, optimizer,1150steps,batch128,UID batches, old6observed assignment-context features, same84train/20fit/26evaluation, original frozen fit20 threshold grid. Geometry zeros activity. No GPU, new video replay, features, full-box labels or core training.

## Что сделано
2026-10-25

## Получено
runs/z13_activity_amplitude_20261025/training/report.json → paired: {'Daffine-baseline': {'mean_per_video': -0.003491223018656633, 'ci95': [-0.012578521010528889, 0.001819947023518558], 'better': 13, 'n': 26}, 'Daffine-D_norm': {'mean_per_video': -0.0015619521565209335, 'ci95': [-0.007228524458251436, 0.001839425119657753], 'better': 13, 'n': 26}, 'Daffine-C_affine': {'mean_per_video': -0.003785495816930639, 'ci95': [-0.012787944038714437, 0.0014824769210418944], 'better': 11, 'n': 26}, 'Daffine-R1_affine': {'mean_per_video': -0.0035257817259811966, 'ci95': [-0.012488720828683379, 0.0018193839243145441], 'better': 12, 'n': 26}, 'Daffine-geometry_affine': {'mean_per_video': -0.003491223018656633, 'ci95': [-0.012578521010528889, 0.001819947023518558], 'better': 13, 'n': 26}}

## Интерпретация
Проверено одно преобразование68входов при равной ёмкости, одинаковых начальных весах и точном повторе12старых моделей. Это действие регионального считывателя на повторных26видео, не HOTA и не полное объяснение отказа.

## Решение
Pass -> separate frozen closed-loop gate, including native/C/R1/no-FlyVis. Fail -> preserve this exact normalization ablation and do not claim input amplitude is universally useless; no epsilon/gain/normalizer sweep on26. Основной критерий: False

## Нельзя заявлять
Normalization may be essential for stable cross-channel matching. Dataset-level scales and relative channel patterns may already contain enough evidence; retaining magnitude can saturate Tanh and make generalization worse. A useful effect may occur equally in eye/geometry controls. Reused26 is exploratory, not independent confirmation.

## Критерии приёмки (записаны до прогона)
- ✅ All4normalization controls reproduce their prior context models tensor-by-tensor, selected tau and every8326saved row exactly. Equal parameter counts and initial tensors. All feature/context hashes and24model outputs audited against unchanged action labels; real CPU gradient test finite, CUDA uninitialized. First completed model2*time*23+60 fits remaining1800s; otherwise preserve and stop.  
  *runs/z13_activity_amplitude_20261025/independent_audit.json*
- ❌ Daffine-baseline and Daffine-Dnorm eachmean>=2pp; all5paired95%video lower>0: baseline,Dnorm,Caffine,R1affine,geometryaffine. Every seed/count/threshold retained. Daffine fixed primary, no choosing seed/mode on26.  
  *runs/z13_activity_amplitude_20261025/training/report.json*

## Доказательства
- Один внутренний преобразователь и равные контроли до расчёта: `runs/z13_activity_amplitude_20261025/protocol.json` · 

