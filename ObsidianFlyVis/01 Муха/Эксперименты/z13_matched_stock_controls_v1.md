---
aliases:
- 'З13: штатная FlyVis C и контроли без мухи в общем сравнении7val'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- метка/benchmark
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: штатная FlyVis C и контроли без мухи в общем сравнении7val

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_matched_stock_controls_v1`

Частичный результат:5из35заданий,цена полного остатка не помещается в исходный бюджет;вывода об эффективности нет.

## Вопрос
Каковы результаты штатного ядра C и обученной D при одинаковом окружении и способе считывания; отличается ли добавка мухи от контроля без неё?

## Гипотеза
Знак отличий заранее неизвестен; прежние35dev не заменяют новое общее сравнение.

## Ожидалось (до прогона)
Same original video builder, four observed frames followed by recurrent search history, max4eyes, retina/lamina, existing frozen per-arm weights/normalization/lambda/tau. Legacy C/B use identical spatial architecture and their saved trained readouts; center C/R1 use identical ridge algorithm with separately preselected training-only configurations. C is pretrained frozen core, not an unadapted complete tracker. B is learned smallGRU, R1 temporal receptor control: neither equals parameter-matched FlyVis. C/D differences include individually trained readout/policy weights; this is not an isolated backbone-only causal effect.

## Что сделано
2026-10-24

## Получено
runs/z13_matched_trackers_20261024/stock_recovery1/price.json: conservative_remaining_seconds=10509,18;available_seconds=8938,789432;proceed=false. controller_result.json:exit1,all_owned_ended=true;готовы5маркеров/jobs,исходыC/B/R1по7видео не измерены.

## Интерпретация
АдаптерR1восстановлен и полный первый ролик сохранён;технический критерий цены не пройден. Это не отрицательный научный итог контролей.

## Решение
Сохранить5готовыхзаданий и остаток30,не продлевать исходный бюджет10800с. Не строить неполную таблицу10систем. Новая дешёваяCPUгипотеза регистрируется отдельной карточкой,не как продолжение бюджетаstock.

## Нельзя заявлять
Only7previously used val videos; not independent final test and no75heldout. Detector validation set used in its training monitoring, but these videos are absent from gradient-training list. Settings not optimized per tracker; cannot infer universal ranking or compare directly with published benchmarks. The old MASA head was trained on XYWH life pairs and can transfer differently to other trackers; report raw/MASA separately. Same original video builder, four observed frames followed by recurrent search history, max4eyes, retina/lamina, existing frozen per-arm weights/normalization/lambda/tau. Legacy C/B use identical spatial architecture and their saved trained readouts; center C/R1 use identical ridge algorithm with separately preselected training-only configurations. C is pretrained frozen core, not an unadapted complete tracker. B is learned smallGRU, R1 temporal receptor control: neither equals parameter-matched FlyVis. C/D differences include individually trained readout/policy weights; this is not an isolated backbone-only causal effect.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (3f903b040856042f18d65e571c205cd3434191ba) — Прежний бюджет сохранён;неполный запуск остановлен технической ценой.
  > Отдельное расширение до измерения C/B/R1; основной протокол и report не меняются.

## Критерии приёмки (записаны до прогона)
- ❌ Parent six-system run complete exit0; sources,weights,inputs frozen; all35controljobs and7videos complete.  
  *runs/z13_matched_trackers_20261024/stock_recovery1/price.json;controller_result.json*
- ⏳ Baseline MASA features exactly reproduced for every video; same scorer and annotation projection.
- ⏳ For each particular positive claim paired-video95%CI lower>0; pooled and mean-video differences reported separately. No tuning based on these seven videos; all outcomes shown.

## Доказательства
- Условия и сравнения до новых результатов: `runs/z13_matched_trackers_20261024/stock_controls/protocol.json` · 
- Служебное восстановление известной формыR1,прежние3готовыхзадания и исходный срок сохранены: `runs/z13_matched_trackers_20261024/stock_recovery1/protocol.json` · 

## Код
- `scripts/z13_matched_stock_controls_20261024.py`
- `scripts/z13_matched_stock_queue_20261024.py`
- `scripts/z13_matched_stock_console.ps1`

