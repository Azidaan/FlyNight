---
aliases:
- 'Внутреннее считывание FlyVis: обучение последствиям полной рамки'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Внутреннее считывание FlyVis: обучение последствиям полной рамки

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_full_box_supervision_v1`

Расчёт меток остановлен проверкой стоимости; эффективность не измерена.

## Вопрос
Сможет ли внутренний выбор/отказ использовать запас полной рамки при обучении именно этому действию?

## Гипотеза
Рабочая гипотеза: метки прежнего категориального изменения стоимости не переносятся автоматически на другую геометрию действия. Обучение полной рамке при равных признаках может изменить полезность.

## Ожидалось (до прогона)
Replace the internal regional readout action semantics and corresponding supervision: selected real region now transfers complete XYWH and uses unchanged native get_dists, instead of a categorical cost row. Explicit unmatched still sets this row costs to1; fixed unchanged logit still means no action. No row or target identity is supplied at runtime. Reuse all D/C/R1 raw68 regional activity, observed target coefficients, original assignment context6, eight geometry inputs and query UID order. Same24heads (4representations x context/blank x3seeds),1150steps,batch128,Adam0.001,signed good-harm loss, fixed fit20 threshold grid. Train84; calibrate20; evaluate26. Only outcome labels and applied candidate-box action change together. No new feature/core computation or GPU.

## Что сделано
2026-10-25

## Получено
runs/z13_full_box_supervision_20261025/cache/price.json:1544,032с против1170,337с; cache/recovery1/price.json:1112,160с против909,670с;4из130видео,0из24моделей.

## Интерпретация
Численная пакетная версия уменьшила цену, но с исходным консервативным запасом всё ещё не проходит остаток1200с. Это остановка исполнения по стоимости, не отрицательная научная проверка согласованных меток.

## Решение
Сохранены4видео,осталось126и24модели. Текущее окно cache не продлевать, третий ценовой повтор не запускать. Семья двух ручных защит отрицательна; обучение полной рамке остаётся не проверенным. Продолжить отдельную проверку другого внутреннего узла на уже готовых признаках без нового прогона видео.

## Нельзя заявлять
The old representations may not identify absent targets even with correct action labels. Severe imbalance may yield no action; the same geometry controls may suffice. Action reserve does not imply learnability. Both20fit and26evaluation repeatedly used, so this is exploratory, not independent confirmation. No manual guards or threshold edits after evaluation.

## Критерии приёмки (записаны до прогона)
- ❌ All130 baseline raw outputs exactly reproduce original. All37296query bindings/region orders match original labels and saved context; same observed features reused. Label arrays contain actual full assignment outcomes for each full-box/null action, not IoU or identity surrogate. Complete means restored; unchanged action exact. All input hashes; every model/hash/row/interval independently audited. First2label replays2*mean*128+60 must fit1200s. Firsttraining model2*time*23+60 must fit1800s. No completed video/model repeated.  
  *runs/z13_full_box_supervision_20261025/cache/recovery1/price.json*
- ⏳ Preregister D_context as primary, no choosing blank after evaluation: net baseline effect mean>=2pp and all6paired95%video lower>0 versus baseline,D_blank,C_context,R1_context,geometry_context,previous Dcontext categorical action. Report all24models and frozen fit20 thresholds, including zero-action and unfavorable outcomes.

## Доказательства
- Связанный выход/метки, равные признаки и критерии до обучения: `runs/z13_full_box_supervision_20261025/protocol.json` · 

