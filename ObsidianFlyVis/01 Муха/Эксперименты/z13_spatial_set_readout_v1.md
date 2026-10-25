---
aliases:
- Считывание FlyVis по активности отдельных колонок и их положениям
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

# 🔴 Считывание FlyVis по активности отдельных колонок и их положениям

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_spatial_set_readout_v1`

На25.10.2026 пространственное считывание отдельных колонок не удалось успешно реализовать: критерий75моделей не пройден. Модуль отложен по решению автора.

## Вопрос
Даёт ли расположение активности внутри реальной области пользу, потерянную её средним?

## Гипотеза
Рабочая гипотеза: пары активность–положение различают похожие области при одинаковой средней активности. Структурная возможность проверена синтетически; причина реальных ошибок и польза не установлены.

## Ожидалось (до прогона)
Replace the internal regional embedding of FlyVis activity with a Deep-Sets-style mean of learned functions of every supported (68-channel activity,2relative coordinates) pair. Keep actual full-box/NULL actions and unchanged target descriptor. No quadrants, subsampling or occupancy gate. New outputs can change the chosen actual assignment action. D_spatial versus D_mean: latter repeats the68mean computed from the EXACT SAME new fullfloat32activity at each same supported coordinate before the SAME point network. Thus all positions/support/cardinality, train splits, initialization, parameters and loss identical. C_spatial stock fly; R1_spatial equal-dimensional rich retinal history; geometry has identical support/coordinates/context but zeros activity and target. Three seeds equally averaged, no best cell selection.

## Что сделано
2026-10-25

## Получено
runs/z13_spatial_set_readout_20261025/recovery2/training/internal_report.json → paired: {'D-baseline': {'mean_per_video': -0.00011188956124788997, 'ci95': [-0.0007302264487472782, 0.000525070479122962], 'better': 23, 'n': 104}, 'D-D_mean': {'mean_per_video': 0.0006859438791967456, 'ci95': [-0.00010731459688319186, 0.0015064673485447172], 'better': 40, 'n': 104}, 'D-C_spatial': {'mean_per_video': -0.000253899725459348, 'ci95': [-0.0008960737920447742, 0.00040865282072537375], 'better': 23, 'n': 104}, 'D-R1_spatial': {'mean_per_video': -0.00028985435992858686, 'ci95': [-0.001259120924958217, 0.00062104862279023], 'better': 33, 'n': 104}, 'D-geometry': {'mean_per_video': 6.308024565288488e-06, 'ci95': [-0.0008338198311471847, 0.000831314591849844], 'better': 32, 'n': 104}}. Все75моделей по1150шагов; audit.json в training/result_audit независимо проверил строки/метки/пороги/интервалы. Старый controller_result сохраняет exit1: после сохранения сравнений сработала оставшаяся проверка CPU-only; обучение не повторялось.

## Интерпретация
Эта конечная модель пространственного считывания не дала подтверждённой чистой пользы или преимущества над равными D_mean/C/R1/геометрией. Все5интервалов включают0; это не доказательство нулевого истинного эффекта или отсутствия информации в активности. Оценка на повторно использованных104видео; интервалы условны на разбиении. Образец по-прежнему сжат до68признаков, поэтому полное пространственное сопоставление образца и области не проверено.

## Решение
Модуль временно ОТЛОЖЕН25.10.2026 по решению автора; к нему можно вернуться позже. Разработка не запрещена, направление не закрыто, дополнительных ограничений на возврат этим решением не введено. Сохранить исходники, веса, кэши, отрицательный итог75моделей и служебные ошибки для продолжения. Завершённая версия не прошла предварительный критерий, поэтому её15заключительных моделей, новая26оценка иHOTAне запускались. Проблема и выполненные проверки: dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md. Общие правила экспериментов остаются прежними.

## Нельзя заявлять
Нельзя заявлять улучшениеHOTA, исправление отказа, преимущество D надC/R1/контролем безмухи, доказанное отсутствие пространственной информации или итог по всей линииFlyVis. Конечные16скрытых признаков,1150шагов и сжатый образец ограничивают проверенный механизм. ОтдельнаяGPUреализация была допущена по стоимости;5одинаковыхблоковCPU и70GPU, старые ошибки и исходники сохранены.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.
  > 
- **Получено**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.
  > 
- **Интерпретация**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.
  > 
- **Решение**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.
  > 
- **Нельзя заявлять**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.
  > Relative spatial evidence may be redundant/noisy; only16aggregate dimensions and1150steps may be insufficient. Target still the prior68descriptor, so failure does not test full spatial exemplar matching. Irregular support supplies geometry, hence geometry and repeated-mean controls mandatory. Point-set theory supplies an architecture, NOT a guarantee of universal representation with this finite embedding or tracking improvement.
- **Суть**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Прямое решение автора25.10.2026: описать проблему/сделанные проверки и отложить данный модуль; прежние итоговые поля сохранены в revisions.
  > Пространственное считывание отдельных колонок: внутренний критерий не пройден;75моделей на104видео, новых26/HOTAнет.
- **Решение**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Прямое решение автора25.10.2026: описать проблему/сделанные проверки и отложить данный модуль; прежние итоговые поля сохранены в revisions.
  > Не принимать эту реализацию и не запускать15итоговых моделей, новую26оценку илиHOTA. Не менять порог/сид/длину обучения по этому итогу. Следующий кандидат по заранее записанному ранжированию — пространственное сопоставление образца и области; сначала дешёвая проверка доступности исходных наблюдений и отличий от уже испытанного внимания. Готового исполнителя следующего обучения нет. Служебный сбой завершения не скрывать и не повторять75готовыхмоделей.
- **Решение**, 2026-10-25 (894e37ba1c7877f148fab08aa254743070607fcf) — Автор уточнил25.10.2026: сказано отложить, а не запретить продолжение. Требование отдельного нового разрешения на возврат было лишним добавлением агента.
  > ОТЛОЖЕНО25.10.2026 по прямому решению автора. Не продолжать разработку этого модуля, не запускать15заключительных моделей, новую26оценку илиHOTA этойверсии. Не подбирать другой порог/сид/длину по данному исходу. Сохранить все исходники, результаты и служебные ошибки. Возврат только по новому решению автора с отдельной постановкой, отличающейся от уже проверенной. Проблема, история попыток и ограничения: dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md. НаправлениеFlyVis и чужие подтверждённые узлы этим решением не закрываются.

## Критерии приёмки (записаны до прогона)
- ✅ Unchanged130query/episode/frame/choices/masks binding and original normalization/weights/history. Every old peak center<0.5px,score<.001, old tolerances unchanged. Mean-control features are computed with the ORIGINAL pool_activity on the SAME new fullfloat32arrays, save/reload means exact. Against previous separately extracted mean arrays record maximum difference descriptively; separateGPUpasses are not assumed bit-exact. New models exactly reproduce saved validation predictions from weights.  
  *runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/audit.json;features/report.json complete;75точных сохранённых повторов предсказаний, хеши/метки/разбиения проверены*
- ❌ Same five nested video folds onfit104 as previous trial; threshold selected only on disjoint calibration videos within each fold. Primary D_spatial mean-video(help-harm)/known>=.02 and ALL paired95%lower>0 versus zero,D_mean,C_spatial,R1_spatial,geometry. Seeds equally averaged. Conditional descriptive intervals on reused fit104; no information-absence claim on failure.  
  *runs/z13_spatial_set_readout_20261025/recovery2/training/internal_report.json: mean<.02;все5основныхДИвключают0*
- ⏳ Only if internal admission passes:15new models fit84/cal20; once evaluate26, same criteria including frozen previous D_context from full-box followup. No automatic HOTA. Failed internal stops before26.  
  *Заключительная ступень не запускалась после непройденного внутреннего критерия;0новых26оценок*

## Доказательства
- Отличия, равные контроли, критерии, источники и бюджет до новых признаков: `runs/z13_spatial_set_readout_20261025/protocol.json` · 
- Служебное восстановление CUDA после CPU-импорта; научный протокол и прежние сроки сохранены: `runs/z13_spatial_set_readout_20261025/recovery1/protocol.json` · 
- Упаковка16 только замороженного вывода, готовые46строк переиспользованы; прежние критерии и сроки: `runs/z13_spatial_set_readout_20261025/recovery2/protocol.json` · 
- CPU-служебное восстановление памяти; все1863GPUблока переиспользуются,0моделей повторено, прежний срок: `runs/z13_spatial_set_readout_20261025/recovery2/training/recovery3/protocol.json` · 
- Равное аппаратное расписание:CPUseed0fold0 для всех,остальныеGPU; перваяCPUмодель сохранена, критерии и срок прежние: `runs/z13_spatial_set_readout_20261025/recovery2/training/recovery4_gpu/protocol.json` · 
- Все5сравнений75моделей: `runs/z13_spatial_set_readout_20261025/recovery2/training/internal_report.json` · 
- Независимая сверка и сохранённая служебная ошибка: `runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/audit.json` · 
- Суть проблемы, что сделано, все5сравнений и решение отложить: `dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md` · 

