---
aliases:
- 'Общие20val: десять систем с одинаковой MASA и контролями C/B/R1'
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/benchmark
- метка/fly-problem-P8
исход: идёт
направление: FlyVis
начат: '2026-10-24'
---

# 🔵 Общие20val: десять систем с одинаковой MASA и контролями C/B/R1

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `z13_val20_common_v1`

181задание проверено;незавершённый R1 прошёл инструментальный повтор,продолжается остаток в исходном окне.

## Вопрос
Каковы FlyVis и внешние трекеры при одинаковых исходных кадрах,детекциях,оценке и сшивке на фиксированных20val?

## Гипотеза
Знак добавки D/штатной C и общий рейтинг заранее неизвестны;проверка устраняет смешение условий.

## Ожидалось (до прогона)
Complete a wider fixed matched evaluation of existing frozen readouts/cores with native competitors. No new learned candidate, descriptor, loss or policy is promoted. Four baseline systems plus legacyD/centerD/legacyC/centerC/legacyB/centerR1. Each raw and with identical frozen base+MASA. This is evaluation, not a newly improved FlyVis version. C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.

## Что сделано
2026-10-24

## Получено
runs/z13_val20_matched_20261024/common/controller_result.json:exit1,all_owned_ended=true;common_replay_diagnostic/report.json:старыйпроизводительтакженеповторилпобитово,максимум0,000030517578125px,числастрок/кадры/идентификаторы/классы/уверенностиравны. Сохранены4точныхконтроляобычныхтрекеров;ниодногоиз20основныхроликов10системнеобработано.

## Интерпретация
Строгий технический критерий не выполнен;это не научный отрицательный итог D. Конкретный источник малого межпроходного разброса не установлен.

## Решение
Использовать проверенный полный R1выход диагностического повтора;не вычислять готовое снова. Продолжить только остаток,с предварительными снимками собственного стека при новом простое. Причина прежних зависаний не установлена. Исходный конец25.10 06:00:27МСК не продлевать;до полного20valитога эффективности не оценивать.

## Нельзя заявлять
All200val videos were used in older low-frame-rate tracker research; detector val monitored during weight selection. New20 are additional original-frame caches, not untouched independent confirmation or final test. No outcome-dependent exclusion; ambiguous alignment stops preparation, no replacement video. Additional20 were selected before outcomes, but are not untouched final test. Native algorithms not optimally tuned. MASA scorer trained on XYWH pairs may transfer unequally;show raw and MASA. One saved D is not a seed-independence result. Prior7 subset and these20 remain separate tables;do not compare their raw numbers as changes. C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (16f908cea316bfd29863c62e5dda42ec4007680f) — Строгий контроль старой D повторно не выполнен;допуск без автора не ослаблен.
  > Зарегистрировано до общего сравнения;числа CPUraw20 не подменяют общий итог с FlyVis/MASA.
- **Суть**, 2026-10-24 (38bcf133a4b4a39db35ed1e9b8958aae9550e86b) — Явное разрешение автора на отдельный координатный служебный допуск.
  > Общее10-системное сравнение остановлено на точном контроле D;готовы4контроля,эффективность не измерена.
- **Решение**, 2026-10-24 (38bcf133a4b4a39db35ed1e9b8958aae9550e86b) — Явное разрешение автора на отдельный координатный служебный допуск.
  > Не продолжать ветвь FlyVis с изменённым допуском без решения автора;вопрос задан. Независимо выполняется отдельная общая MASA четырёхтрекеров с прежними точнымиконтролями. Старые ошибка/источники/код/журналы сохранены;бесконечногоповтораDнет.
- **Суть**, 2026-10-24 (c36ddbe123589b0f34284bb42acd5306faf3a0cd) — Сохранены результаты после600с без прогресса MASA;ресурсная очередь без изменения научных настроек.
  > Автор разрешил координатный служебный допуск;общее сравнение продолжается с86проверенных заданий.
- **Решение**, 2026-10-24 (c36ddbe123589b0f34284bb42acd5306faf3a0cd) — Сохранены результаты после600с без прогресса MASA;ресурсная очередь без изменения научных настроек.
  > Продолжить только недостающие6вариантов на техже20val;готовые4трекера/MASA не повторять. Исходный конец25.10 06:00:27МСК не продлён. Первый новый полный ролик проверяет цену остатка. Старый точный контроль остаётся непройденным и сохранённым.
- **Суть**, 2026-10-24 (e51214b01b68d732b6f4df7817bb8de0a3aa37d4) — Полный инструментальный повтор незавершённого R1 и проверка181результата;старые сбои сохранены.
  > 118заданий сохранены;после остановки MASA ожидается свободная GPU для единственного служебного возобновления.
- **Решение**, 2026-10-24 (e51214b01b68d732b6f4df7817bb8de0a3aa37d4) — Полный инструментальный повтор незавершённого R1 и проверка181результата;старые сбои сохранены.
  > Продолжить только незавершённые задания в прежнем окне до25.10 06:00:27МСК после освобождения ресурсов. Не останавливать чужую игру. Повтор того же зависания заново диагностировать,не перезапускать бесконечно. До полного20valитога вывода об эффективности нет.

## Критерии приёмки (записаны до прогона)
- ❌ First old fixed control video: all10 raw output arrays exactly reproduce stored previous signatures;XYWH feature arrays and MASA baseline embeddings exactly repeat. Fresh process per signature/MASA,one namedGPUmutex. Then first fixed new video all10+MASA retained;2x its full walltime*remaining19+900seconds must fit remaining stage86400s and author deadline. Scientific thresholds unchanged. Stop after600s without frame/stage progress,retain completed jobs,no automatic repeated failed child.  
  *runs/z13_val20_matched_20261024/common_replay_diagnostic/report.json*
- ⏳ Все20×10систем raw/MASA,231задание,первичные строки/хеши и независимая сверка;своё дерево закончено в бюджете.
- ⏳ Report all10 raw/MASA pooled HOTA and all per-video values. For each concrete benefit claim require paired-video95%CI strict lower>0 for the exact matched contrast. FlyVis-to-stand requires each D versus xywh; modernization versus stock additionally D versus corresponding C, and matched receptor/smallGRU contrast. No secondary comparison rescues a failed primary. Complete20 and all231jobs required; otherwise only technical remainder, no efficacy verdict.
- ⏳ Дописан после разрешения автора,до основного20val-прогона: Only FlyVis/control-model old outputs: absolute coordinate error<=0.0001px;count/frame/ID/class/confidence exact. Native raw and previous baseline MASA remain exact. Existing86verified markers retained. No output rounded,altered or substituted. Old failed bit-exact criterion retained. Missing6arms only;native4raw/MASA/evaluations reused unchanged. Missing6 MASA uses identical per-frame backbone and per-arm ROI operations. Same-frame fp16 guard remains exact.

## Доказательства
- Фиксированные условия,критерии,контроли и бюджет до запуска: `runs/z13_val20_matched_20261024/common/protocol.json` · 
- Готовая исходная ступень20val без FlyVis/MASA: `runs/z13_val20_matched_20261024/raw/report.json` · 
- Явное разрешение,новый служебный контроль,86готовых результатов и исходный абсолютный срок.: `runs/z13_val20_matched_20261024/common_authorized/protocol.json` · 
- 118готовых заданий,прежняя цена/срок и ожидание ресурсов;старые логи/ошибка сохранены.: `runs/z13_val20_matched_20261024/common_authorized_recovery1/protocol.json` · 
- Диагностический повтор,181готовый результат и наблюдение стека без изменения научных настроек.: `runs/z13_val20_matched_20261024/common_authorized_recovery2/protocol.json` · 

## Код
- `scripts/z13_val20_common_20261024.py`
- `scripts/z13_val20_common_console.ps1`
- `scripts/z13_val20_common_finish_20261024.py`

