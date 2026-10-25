---
aliases:
- 'З13: внутреннее межкадровое считывание с метками личности реальных областей'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🔴 З13: внутреннее межкадровое считывание с метками личности реальных областей

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_temporal_identity_readout_v1`

Межкадровый считыватель с метками личности не прошёл критерии этой реализации.

## Вопрос
Даёт ли обучаемое соответствие реальных областей пользу выбора действий сверх истории без меток, текущего чтения, штатнойC,входаR1 игеометрии?

## Гипотеза
Явное обучение известных соответствий может помочь сохранять личность и выбирать полезную старую рамку; это рабочая гипотеза,а не установленная причина неудач.

## Ожидалось (до прогона)
Replace internal full-box action readout by target-conditioned cross-frame real-region association; residual action scores can change the selected old full-box/NULL action. Keep old current features, labels and action semantics. Latest supported strictly earlier stored observation on the fixed alignment reference clock, same episode; exact same reference selection at training/application. Labels never choose reference. Missing prior history gives zero temporal residual; all original queries remain evaluated.

## Что сделано
2026-10-25

## Получено
runs/z13_temporal_identity_20261025/training/internal_report.json → paired: {'D-baseline': {'mean_per_video': -0.0003002062571798735, 'ci95': [-0.0009200805189374642, 0.00034795038638010703], 'better': 23, 'n': 104}, 'D-D_blank': {'mean_per_video': -0.00010676456150873848, 'ci95': [-0.000951386550063532, 0.0009306469313944547], 'better': 23, 'n': 104}, 'D-C': {'mean_per_video': 6.910289860455686e-05, 'ci95': [-0.0009300072566417491, 0.001099053804324879], 'better': 34, 'n': 104}, 'D-R1': {'mean_per_video': -0.00011198368793115165, 'ci95': [-0.0007344736806070753, 0.0005242285653164296], 'better': 25, 'n': 104}, 'D-geometry': {'mean_per_video': -0.0003104136718458869, 'ci95': [-0.000924568481044985, 0.00033811819954452955], 'better': 22, 'n': 104}, 'D-D_no_identity': {'mean_per_video': 0.00010451017886935562, 'ci95': [-0.0007553043375120709, 0.0010525560920911612], 'better': 32, 'n': 104}}; runs/z13_temporal_identity_20261025/training/report.json:models=90. independent_audit.json:всеисходы,веса,сохранённыеповторыпредсказанийиинтервалыпроверены.

## Интерпретация
Проверено внутреннее считывание реальных областей,одинаковаяопорнаясетка иявноеразмеченноесопоставление прификсированномядре. Одна эта реализация не реализовала заранее заданную чистую пользу и полный набор преимуществ над контролями. Это 104видео вложеннойпроверки; условные интервалы и одноступенчатые назначения,неHOTA. Диагностика соответствий сама по себе не доказывает пользу вмешательства или причину неудачи.

## Решение
Эту реализацию не переносить в стенд,не подбирать порог/сид/длину по её исходу. Сохранить узел и полный отрицательный результат. Следующее изменение выбирать по различию обучения соответствий и полезного выбора,не объявляя линиюFlyVis закрытой. Новая26оценка не запускалась после внутренней остановки.

## Нельзя заявлять
Not fullKeepTrack,not a retained identity bank or full spatial exemplar matching. No HOTA/core-training/independentconfirmation/calibrated absence. Finite regional mean68/old target68 and fixed clock limit mechanism. Reused104and26 exploratory. GT-known associations may be insufficient or unrelated to beneficial intervention; neighboring objects and sparse clock can still confuse FlyVis. Improved pairAUROC alone is not benefit to the stand. The same architecture may perform equally well using geometry or retinal input.

## Критерии приёмки (записаны до прогона)
- ✅ All12previous contextual heads4representations*3seeds on8326queries: restored saved rows/probabilities/outcomes exact AND new zero residual exact same logits with prior weights. Pure CPU regression while GPU feature extraction runs. New trained predictions exactly reproduced from saved weights on their validation inputs.  
  *runs/z13_temporal_identity_20261025/training/original_regression.json; independent_audit.json*
- ✅ Same6architectures/parameter allocation/init per seed,UID minibatch draws,1150updates,batch128,Adam.001 and all original outcome labels. D/C/R1 share actual frames/regions/masks. No architecture/loss/threshold/seed choice after outcomes.  
  *runs/z13_temporal_identity_20261025/features/independent_audit.json;training/independent_audit.json*
- ❌ On full104out-of-fold videos: D mean-video net>=.02 AND every paired-video95%lower>0 versus0,D_blank,C,R1,geometry,D_no_identity. All required; otherwise stop before any new26evaluation. Intervals conditional on reused104/fixed folds.  
  *runs/z13_temporal_identity_20261025/training/internal_report.json*
- ⏳ Only after internal pass:18newmodels fit84/cal20,one evaluation on26reusedvideos with same6contrasts and frozen previousD_context contrast; alllower>0,net>=.02. No automatic HOTA. Diagnostic pairAUROC is secondary and cannot rescue action gate.  
  *runs/z13_temporal_identity_20261025/training/internal_report.json; final stage not run*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [temporal_readout_net_assignment_effect↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%A7%D0%B8%D1%81%D1%82%D0%B0%D1%8F%20%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%B0%20%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%B0%20%D1%80%D0%B0%D0%BC%D0%BE%D1%87%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%B4%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D0%B8%D1%8F) | -0.0003002062571798735 | 0 | 104видео вложеннойпроверки | `runs/z13_temporal_identity_20261025/training/internal_report.json` · paired.D-baseline.mean_per_video |

## Доказательства
- Архитектура,одинаковое прошлое,отличия,контроли,цена икритерии дообучения: `runs/z13_temporal_identity_20261025/training/protocol.json` · 
- Независимый полный пересчёт исходов и интервалов: `runs/z13_temporal_identity_20261025/training/independent_audit.json` · 
- Число моделей,ступени,отсутствиеHOTA: `runs/z13_temporal_identity_20261025/training/report.json` · 
- Точное воспроизведение12исходныхголов: `runs/z13_temporal_identity_20261025/training/original_regression.json` · 
- Успешный контроллер и полное завершение дерева; прежний неудачный замер цены сохранён отдельно: `runs/z13_temporal_identity_20261025/training/recovery2/controller_result.json` · 
- Описательные числа полезных, вредных и неизвестных выбранных действий; повторные наблюдения по трём начальным состояниям, не уникальные запросы.: `runs/z13_temporal_identity_20261025/training/action_diagnostic.json` · 
- Проверка пакетного исполнения независимых моделей: градиенты, потери, шаг Adam и неизменные правила выбора.: `runs/z13_temporal_identity_20261025/training/recovery2/numerical_parity.json` · 
- Исправление технического ограничения классов без изменения научного протокола.: `runs/z13_temporal_identity_20261025/training/recovery2/structural_stack_recovery3.json` · 

