---
aliases:
- 'Общие20val: десять систем с одинаковой MASA и контролями C/B/R1'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/benchmark
- метка/fly-problem-P8
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-24'
---

# 🟡 Общие20val: десять систем с одинаковой MASA и контролями C/B/R1

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_val20_common_v1`

Общее10-системное сравнение остановлено на точном контроле D;готовы4контроля,эффективность не измерена.

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
Не продолжать ветвь FlyVis с изменённым допуском без решения автора;вопрос задан. Независимо выполняется отдельная общая MASA четырёхтрекеров с прежними точнымиконтролями. Старые ошибка/источники/код/журналы сохранены;бесконечногоповтораDнет.

## Нельзя заявлять
All200val videos were used in older low-frame-rate tracker research; detector val monitored during weight selection. New20 are additional original-frame caches, not untouched independent confirmation or final test. No outcome-dependent exclusion; ambiguous alignment stops preparation, no replacement video. Additional20 were selected before outcomes, but are not untouched final test. Native algorithms not optimally tuned. MASA scorer trained on XYWH pairs may transfer unequally;show raw and MASA. One saved D is not a seed-independence result. Prior7 subset and these20 remain separate tables;do not compare their raw numbers as changes. C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (16f908cea316bfd29863c62e5dda42ec4007680f) — Строгий контроль старой D повторно не выполнен;допуск без автора не ослаблен.
  > Зарегистрировано до общего сравнения;числа CPUraw20 не подменяют общий итог с FlyVis/MASA.

## Критерии приёмки (записаны до прогона)
- ❌ First old fixed control video: all10 raw output arrays exactly reproduce stored previous signatures;XYWH feature arrays and MASA baseline embeddings exactly repeat. Fresh process per signature/MASA,one namedGPUmutex. Then first fixed new video all10+MASA retained;2x its full walltime*remaining19+900seconds must fit remaining stage86400s and author deadline. Scientific thresholds unchanged. Stop after600s without frame/stage progress,retain completed jobs,no automatic repeated failed child.  
  *runs/z13_val20_matched_20261024/common_replay_diagnostic/report.json*
- ⏳ Все20×10систем raw/MASA,231задание,первичные строки/хеши и независимая сверка;своё дерево закончено в бюджете.
- ⏳ Report all10 raw/MASA pooled HOTA and all per-video values. For each concrete benefit claim require paired-video95%CI strict lower>0 for the exact matched contrast. FlyVis-to-stand requires each D versus xywh; modernization versus stock additionally D versus corresponding C, and matched receptor/smallGRU contrast. No secondary comparison rescues a failed primary. Complete20 and all231jobs required; otherwise only technical remainder, no efficacy verdict.

## Доказательства
- Фиксированные условия,критерии,контроли и бюджет до запуска: `runs/z13_val20_matched_20261024/common/protocol.json` · 
- Готовая исходная ступень20val без FlyVis/MASA: `runs/z13_val20_matched_20261024/raw/report.json` · 

## Код
- `scripts/z13_val20_common_20261024.py`
- `scripts/z13_val20_common_console.ps1`
- `scripts/z13_val20_common_finish_20261024.py`

