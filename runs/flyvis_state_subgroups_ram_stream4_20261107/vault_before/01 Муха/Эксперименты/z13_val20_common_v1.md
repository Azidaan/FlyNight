---
aliases:
- 'Общие20val: десять систем с одинаковой MASA и контролями C/B/R1'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/benchmark
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: положительный
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 Общие20val: десять систем с одинаковой MASA и контролями C/B/R1

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_val20_common_v1`

Общее сравнение10систем на20val завершено и проверено;положительный статус означает полноту оценки,а не автоматический выигрыш FlyVis.

## Вопрос
Каковы FlyVis и внешние трекеры при одинаковых исходных кадрах,детекциях,оценке и сшивке на фиксированных20val?

## Гипотеза
Знак добавки D/штатной C и общий рейтинг заранее неизвестны;проверка устраняет смешение условий.

## Ожидалось (до прогона)
Complete a wider fixed matched evaluation of existing frozen readouts/cores with native competitors. No new learned candidate, descriptor, loss or policy is promoted. Four baseline systems plus legacyD/centerD/legacyC/centerC/legacyB/centerR1. Each raw and with identical frozen base+MASA. This is evaluation, not a newly improved FlyVis version. C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.

## Что сделано
2026-10-24

## Получено
runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json → pooled/per_video/paired; verified_summary.json: {"xywh": {"raw_hota": 57.900829184147604, "same_masa_hota": 58.48578687561312, "relative_percent_from_xywh_masa": 0.0}, "bytetrack": {"raw_hota": 55.72093730404129, "same_masa_hota": 56.30107747900368, "relative_percent_from_xywh_masa": -3.735453540628351}, "botsort": {"raw_hota": 58.49031414305054, "same_masa_hota": 58.87367976141423, "relative_percent_from_xywh_masa": 0.6632258990138474}, "tracktrack": {"raw_hota": 59.264750081726845, "same_masa_hota": 59.62236652191588, "relative_percent_from_xywh_masa": 1.9433433437768743}, "legacyD": {"raw_hota": 57.85354406342428, "same_masa_hota": 58.48933835257775, "relative_percent_from_xywh_masa": 0.006072376135038312}, "centerD": {"raw_hota": 57.89437488017016, "same_masa_hota": 58.48071262037314, "relative_percent_from_xywh_masa": -0.008676048508630603}, "legacyC": {"raw_hota": 57.839656810761944, "same_masa_hota": 58.37382311091559, "relative_percent_from_xywh_masa": -0.19143756231861614}, "centerC": {"raw_hota": 57.89546052256709, "same_masa_hota": 58.4814674753895, "relative_percent_from_xywh_masa": -0.0073853844743654255}, "legacyB": {"raw_hota": 57.88527381009491, "same_masa_hota": 58.43656103421079, "relative_percent_from_xywh_masa": -0.0841671866483142}, "centerR1": {"raw_hota": 57.50531352980429, "same_masa_hota": 58.376266431341946, "relative_percent_from_xywh_masa": -0.18725993121047155}}

## Интерпретация
Полная оценка не подтвердила пользу добавления D к XYWH+MASA: прежняя D−XYWH средняя по видео −0,00806456,95%ДИ[−0,05454689;0,02558647];центрированная D−XYWH −0,00990808[−0,02941177;0,00023406]. Отдельные положительные сравнения: прежняя D−C +0,12010830[0,06161887;0,18657257],17/20;центрированная D−R1 +0,10853790[0,00447228;0,22112094],13/20. Источник runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json → paired. Эти вторичные сравнения не спасают отсутствие пользы стенду;D−C не выделяет ядро,поскольку различаются обученные считыватели/политики. Положительный статус карточки означает полноту сравнения.

## Решение
Использовать общую таблицу с n=20 и ограничениями. Не подбирать порог/сид и не открывать75. Только положительные конкретные контрасты могут попасть в лучшие результаты;нового обучения этот опыт не выполняет.

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
- **Суть**, 2026-10-24 (000cec7a6a2611654e22d196198ff2968d6d1284) — Полное независимо проверенное сравнение20val.
  > 181задание проверено;незавершённый R1 прошёл инструментальный повтор,продолжается остаток в исходном окне.
- **Получено**, 2026-10-24 (000cec7a6a2611654e22d196198ff2968d6d1284) — Полное независимо проверенное сравнение20val.
  > runs/z13_val20_matched_20261024/common/controller_result.json:exit1,all_owned_ended=true;common_replay_diagnostic/report.json:старыйпроизводительтакженеповторилпобитово,максимум0,000030517578125px,числастрок/кадры/идентификаторы/классы/уверенностиравны. Сохранены4точныхконтроляобычныхтрекеров;ниодногоиз20основныхроликов10системнеобработано.
- **Интерпретация**, 2026-10-24 (000cec7a6a2611654e22d196198ff2968d6d1284) — Полное независимо проверенное сравнение20val.
  > Строгий технический критерий не выполнен;это не научный отрицательный итог D. Конкретный источник малого межпроходного разброса не установлен.
- **Решение**, 2026-10-24 (000cec7a6a2611654e22d196198ff2968d6d1284) — Полное независимо проверенное сравнение20val.
  > Использовать проверенный полный R1выход диагностического повтора;не вычислять готовое снова. Продолжить только остаток,с предварительными снимками собственного стека при новом простое. Причина прежних зависаний не установлена. Исходный конец25.10 06:00:27МСК не продлевать;до полного20valитога эффективности не оценивать.
- **Интерпретация**, 2026-10-24 (000cec7a6a2611654e22d196198ff2968d6d1284) — Явная интерпретация проверенных первичных интервалов после полного итога.
  > Для каждой добавки D,сравнения со штатной C и контроля без мухи действует свой парный интервал;положительный нижний предел необходим. Общую HOTA и среднюю разность по видео не смешивать. Все исходы показаны;настройки по20val не выбирались.

## Критерии приёмки (записаны до прогона)
- ❌ First old fixed control video: all10 raw output arrays exactly reproduce stored previous signatures;XYWH feature arrays and MASA baseline embeddings exactly repeat. Fresh process per signature/MASA,one namedGPUmutex. Then first fixed new video all10+MASA retained;2x its full walltime*remaining19+900seconds must fit remaining stage86400s and author deadline. Scientific thresholds unchanged. Stop after600s without frame/stage progress,retain completed jobs,no automatic repeated failed child.  
  *runs/z13_val20_matched_20261024/common_replay_diagnostic/report.json*
- ✅ Все20×10систем raw/MASA,231задание,первичные строки/хеши и независимая сверка;своё дерево закончено в бюджете.  
  *runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json;verified_summary.json;controller_result.json;231jobs*
- ✅ Report all10 raw/MASA pooled HOTA and all per-video values. For each concrete benefit claim require paired-video95%CI strict lower>0 for the exact matched contrast. FlyVis-to-stand requires each D versus xywh; modernization versus stock additionally D versus corresponding C, and matched receptor/smallGRU contrast. No secondary comparison rescues a failed primary. Complete20 and all231jobs required; otherwise only technical remainder, no efficacy verdict.  
  *runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json;verified_summary.json;controller_result.json;231jobs*
- ✅ Дописан после разрешения автора,до основного20val-прогона: Only FlyVis/control-model old outputs: absolute coordinate error<=0.0001px;count/frame/ID/class/confidence exact. Native raw and previous baseline MASA remain exact. Existing86verified markers retained. No output rounded,altered or substituted. Old failed bit-exact criterion retained. Missing6arms only;native4raw/MASA/evaluations reused unchanged. Missing6 MASA uses identical per-frame backbone and per-arm ROI operations. Same-frame fp16 guard remains exact.  
  *runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json;verified_summary.json;controller_result.json;231jobs*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.48578687561312 | 58.48578687561312 | 20 common val xywh sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.xywh_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 56.30107747900368 | 58.48578687561312 | 20 common val bytetrack sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.bytetrack_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.87367976141423 | 58.48578687561312 | 20 common val botsort sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.botsort_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 59.62236652191588 | 58.48578687561312 | 20 common val tracktrack sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.tracktrack_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.48933835257775 | 58.48578687561312 | 20 common val legacyD sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.legacyD_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.48071262037314 | 58.48578687561312 | 20 common val centerD sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.centerD_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.37382311091559 | 58.48578687561312 | 20 common val legacyC sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.legacyC_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.4814674753895 | 58.48578687561312 | 20 common val centerC sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.centerC_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.43656103421079 | 58.48578687561312 | 20 common val legacyB sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.legacyB_masa.HOTA |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 58.376266431341946 | 58.48578687561312 | 20 common val centerR1 sameMASA | `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · pooled.centerR1_masa.HOTA |

## Доказательства
- Фиксированные условия,критерии,контроли и бюджет до запуска: `runs/z13_val20_matched_20261024/common/protocol.json` · 
- Готовая исходная ступень20val без FlyVis/MASA: `runs/z13_val20_matched_20261024/raw/report.json` · 
- Явное разрешение,новый служебный контроль,86готовых результатов и исходный абсолютный срок.: `runs/z13_val20_matched_20261024/common_authorized/protocol.json` · 
- 118готовых заданий,прежняя цена/срок и ожидание ресурсов;старые логи/ошибка сохранены.: `runs/z13_val20_matched_20261024/common_authorized_recovery1/protocol.json` · 
- Диагностический повтор,181готовый результат и наблюдение стека без изменения научных настроек.: `runs/z13_val20_matched_20261024/common_authorized_recovery2/protocol.json` · 
- Все общие исходы и парные интервалы: `runs/z13_val20_matched_20261024/common_authorized_recovery2/report.json` · 

## Код
- `scripts/z13_val20_common_20261024.py`
- `scripts/z13_val20_common_console.ps1`
- `scripts/z13_val20_common_finish_20261024.py`

