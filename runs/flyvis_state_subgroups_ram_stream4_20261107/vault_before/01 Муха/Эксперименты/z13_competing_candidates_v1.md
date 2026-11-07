---
aliases:
- 'З13: совместный выбор кандидата или отсутствия полезного ответа'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/engineering-future
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: совместный выбор кандидата или отсутствия полезного ответа

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_competing_candidates_v1`

Эта совместная функция ошибки не прошла заданные поисковые критерии;действующий считыватель сохраняется.

## Вопрос
Улучшит ли одна замена функции ошибки выбор полезных вмешательств при той же архитектуре и информации?

## Гипотеза
Конкуренция кандидатов с вариантом отсутствия полезного ответа может улучшить ранжирование уверенности; преимущества над BCE заранее нет.

## Ожидалось (до прогона)
Replace independent weighted candidate BCE by a joint categorical action likelihood over three candidate logits and a fixed zero null logit. Null means no useful candidate, not necessarily physical target absence. Acceptable set includes every candidate already labelled useful. Loss=-log(sum probability of acceptable actions); null is the sole acceptable action if the original useful labels are all zero. Same original positive weight multiplies useful-row loss; fixed optimizer/batches/seeds/1150steps/145950presentations. Objective and its normalization are the intervention; no new parameters. Same original mean-exemplar ranker and initialization,68-16 embedding/88-32-1 head, same cached104train and26evaluation, core and candidates fixed. Reuse18 original BCE models only after full ensemble replay and32-step original loss reproduction. Both BCE and new likelihood use the identical four-way normalized maximum candidate probability for confidence; additionally report original BCE native-logit selection. Geometry and R1 use exactly the same intervention. Unknown training labels excluded.

## Что сделано
2026-10-24

## Получено
runs/z13_competing_candidates_20261024/recovery2/report.json → paired_precision/paired_centers/summary/quota_details: {"precision": {"D_joint_activity-D_bce_common_activity": {"mean_per_video": 0.048117205721694895, "ci95": [0.007281395505079722, 0.09292618016604565], "better": 12, "n": 24}, "D_joint_activity-D_bce_native_activity": {"mean_per_video": 0.06713266577333915, "ci95": [-0.0014995414578183478, 0.13715164793156756], "better": 13, "n": 24}, "D_joint_activity-D_joint_geometry": {"mean_per_video": 0.09081360407308549, "ci95": [0.02397008407130796, 0.17174655399210348], "better": 17, "n": 24}, "C_joint_activity-C_bce_common_activity": {"mean_per_video": 0.022801629742806213, "ci95": [-0.007986805025040317, 0.056523287496816864], "better": 11, "n": 25}, "C_joint_activity-C_bce_native_activity": {"mean_per_video": 0.03308074313027873, "ci95": [-0.04032048682311841, 0.11018151249335457], "better": 11, "n": 25}, "C_joint_activity-C_joint_geometry": {"mean_per_video": 0.07402401210760344, "ci95": [-0.01651348528492492, 0.16284717594926568], "better": 13, "n": 25}, "R1_joint_activity-R1_bce_common_activity": {"mean_per_video": 0.021767264136829348, "ci95": [-0.01183224203332899, 0.061862525397307994], "better": 8, "n": 25}, "R1_joint_activity-R1_bce_native_activity": {"mean_per_video": 0.013776119888248036, "ci95": [-0.027989711019974172, 0.05828968096485255], "better": 9, "n": 25}, "R1_joint_activity-R1_joint_geometry": {"mean_per_video": -0.013295928014194274, "ci95": [-0.07437015488811155, 0.048546206485293154], "better": 9, "n": 25}, "D_joint_activity-R1_joint_activity": {"mean_per_video": 0.18256258406345904, "ci95": [0.09917125398923926, 0.27467907812898723], "better": 16, "n": 23}, "C_joint_activity-R1_joint_activity": {"mean_per_video": 0.14801917352226948, "ci95": [0.05116007941254652, 0.25560416962530924], "better": 16, "n": 24}}, "centers": {"D_joint_activity-D_bce_common_activity": {"mean_per_video": -0.005671492921596097, "ci95": [-0.01951929464297425, 0.006247749933839948], "better": 8, "n": 26}, "D_joint_activity-D_bce_native_activity": {"mean_per_video": -0.005671492921596097, "ci95": [-0.01951929464297425, 0.006247749933839948], "better": 8, "n": 26}, "D_joint_activity-D_joint_geometry": {"mean_per_video": -0.04690733201059071, "ci95": [-0.07095367339108775, -0.02367521698065952], "better": 3, "n": 26}, "C_joint_activity-C_bce_common_activity": {"mean_per_video": 0.0009898075214017621, "ci95": [-0.008314999523777889, 0.009987198670370817], "better": 9, "n": 26}, "C_joint_activity-C_bce_native_activity": {"mean_per_video": 0.0009898075214017621, "ci95": [-0.008314999523777889, 0.009987198670370817], "better": 9, "n": 26}, "C_joint_activity-C_joint_geometry": {"mean_per_video": -0.024064956108592304, "ci95": [-0.049560959899155786, -0.00020940889011048736], "better": 9, "n": 26}, "R1_joint_activity-R1_bce_common_activity": {"mean_per_video": 0.004670329105026095, "ci95": [0.0006035618638713041, 0.008856571949011146], "better": 9, "n": 26}, "R1_joint_activity-R1_bce_native_activity": {"mean_per_video": 0.004670329105026095, "ci95": [0.0006035618638713041, 0.008856571949011146], "better": 9, "n": 26}, "R1_joint_activity-R1_joint_geometry": {"mean_per_video": -0.009842720136834357, "ci95": [-0.017195881756849954, -0.0012568253660515565], "better": 3, "n": 26}, "D_joint_activity-R1_joint_activity": {"mean_per_video": 0.09346122308893351, "ci95": [0.06202688821078995, 0.1263754021353437], "better": 23, "n": 26}, "C_joint_activity-R1_joint_activity": {"mean_per_video": 0.12607272057845437, "ci95": [0.08563161501579818, 0.17095535820790517], "better": 24, "n": 26}}}.18новыхмоделей и18старыхконтролей независимо воспроизведены из весов;все первичные строки и квоты проверены.

## Интерпретация
Эта совместная функция ошибки не прошла заданные поисковые критерии;действующий считыватель сохраняется. Причина прежних неудач этим опытом не устанавливается; совместная нормировка доступна обоим сравниваемым вариантам.

## Решение
Сохранить все исходы и3сида;не выбирать порог/сид поtrain26. Не переносить эту версию в HOTA;выбрать следующий узел по измерениям и первоисточникам.

## Нельзя заявлять
Independent BCE can already learn negatives. Joint competition may add no target identity information, suppress several useful candidates or change gradient scaling. A gain in confidence could be score normalization alone; common-normalization BCE separates that. Unequal knownness of selected cases limits interpretation. It does not establish the cause of FlyVis failures or replace a closed-loop test. Повторно использованные26train;неполнаяKeepTrack,неHOTA.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (3f903b040856042f18d65e571c205cd3434191ba) — Полный независимо воспроизведённый исход функции ошибки.
  > Критерии и сравнения зафиксированы до результатов;дешёвая CPUступень до1800с.

## Критерии приёмки (записаны до прогона)
- ✅ Unit loss/padding/permutation/null gradients; full18controls replay; original32loss<=1e-6; conservative2x first32-step projected18models plus120s fits1800s; sources/caches/weights final hashes; independent controller and visible console.  
  *runs/z13_competing_candidates_20261024/recovery2/report.json; reproduction.json; timing.json; controller_result.json; finish_recorded.json*
- ❌ Основной Dприрост точности среди известных при одинаковом покадровом охвате5%по видео>=5п.п. и нижняя95%ДИ>0;26видео имеют определённую метрику.  
  *runs/z13_competing_candidates_20261024/recovery2/report.json; reproduction.json; timing.json; controller_result.json; finish_recorded.json*
- ❌ Dсовместный превосходит соответствующие геометрию и R1 по той же метрике с нижней95%ДИ>0; средняя доля полезных центровDне ниже BCE.  
  *runs/z13_competing_candidates_20261024/recovery2/report.json; reproduction.json; timing.json; controller_result.json; finish_recorded.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [known_useful_selected_fraction↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D1%85%20%D1%81%D1%80%D0%B5%D0%B4%D0%B8%20%D0%B2%D1%8B%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D1%85%20%D1%81%20%D0%B8%D0%B7%D0%B2%D0%B5%D1%81%D1%82%D0%BD%D1%8B%D0%BC%20%D0%B8%D1%81%D1%85%D0%BE%D0%B4%D0%BE%D0%BC) | 0.30224884442183203 | 0.25413163870013716 | Повторные26train,верхние5%по каждому видео;известные выбранные исходы | `runs/z13_competing_candidates_20261024/recovery2/report.json` · summary.D_joint_activity.known_precision_mean_video |

## Доказательства
- Метод,критерии,сравнения и хеши до прогона: `runs/z13_competing_candidates_20261024/protocol.json` · 
- Полный исход с независимо воспроизведёнными весами/строками/квотами: `runs/z13_competing_candidates_20261024/recovery2/report.json` · 

## Код
- `scripts/z13_competing_candidates_20261024.py`
- `tracking/z13_competing_candidates.py`
- `tests/test_z13_competing_candidates.py`

