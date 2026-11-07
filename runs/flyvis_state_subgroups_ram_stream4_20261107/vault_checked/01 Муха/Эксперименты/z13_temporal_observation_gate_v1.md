---
aliases:
- Готовность прежних наблюдений для межкадрового считывания FlyVis
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: положительный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Готовность прежних наблюдений для межкадрового считывания FlyVis

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_temporal_observation_gate_v1`

Проверка подготовки данных пройдена; качество модели не измерялось.

## Вопрос
Можно ли расширить временные пары за пределы прежних14252без будущих кадров/истинной личности?

## Гипотеза
В сохранённых эпизодах есть более ранние наблюдения с реальными областями, даже если прежнего запроса к мухе не было; это не гипотеза о качестве будущей модели.

## Ожидалось (до прогона)
Prepare a distinct temporal FlyVis readout input: real candidate regions in an earlier stored observation of the SAME episode plus unchanged current regions. This gate only chooses and binds inputs; no learner or feature extraction.

## Что сделано
2026-10-25

## Получено
runs/z13_temporal_observation_gate_20261025/report.json: supported_pairs=22090из37296вместо14252; expanded_oracle={'mean_per_video': 0.05142068680371603, 'ci95': [0.04160032885690229, 0.06108713813385377], 'better': 25, 'n': 26}; reserve_increment={'mean_per_video': 0.03073277577588523, 'ci95': [0.02393740612342265, 0.0375957906076213], 'better': 25, 'n': 26}. independent_audit.json проверил строки/прошедшие кадры/метки/интервалы.88,281сметаданных,0моделей.

## Интерпретация
Подготовка более ранних наблюдений расширила доступные пары до59,23%. Идеальный запас на26повторныхвидео вырос с2,069до5,142п.п.; это доступность действий при знании исхода, не качество FlyVis и не HOTA. Старый запас точно воспроизведён. Истинная личность не использовалась для выбора предыдущего наблюдения; оно не обязано содержать ту же истинную цель.

## Решение
Допущена только отдельная подготовка равных временных признаков D/C/R1 с проверкой текущих центров/баллов и стоимости. Обучение временного узла ещё не выполнено и автоматически не разрешается этим числом; использовать действующее авторское разрешение и отдельный протокол. Пространственный модуль временно отложен, не возобновляется этим опытом.

## Нельзя заявлять
Нельзя заявлять прирост модели на3,07п.п., доказанную причину прошлых ошибок, готовую память личности, равные уже извлечённые признаки D/C/R1, улучшениеHOTA или независимый тест. Есть только проверенные ссылки на прошлые наблюдения/области; следующее извлечение должно сохранить текущую историю и пройти прежние допуски.

## Прежние версии (до исправлений)
- **Получено**, 2026-10-25 (f5c5fd1a88bcd2de35479d49bd5f06a8ed2beee8) — Независимо проверен полный результат подготовки130видео без обучения/GPU.
  > 
- **Интерпретация**, 2026-10-25 (f5c5fd1a88bcd2de35479d49bd5f06a8ed2beee8) — Независимо проверен полный результат подготовки130видео без обучения/GPU.
  > 
- **Решение**, 2026-10-25 (f5c5fd1a88bcd2de35479d49bd5f06a8ed2beee8) — Независимо проверен полный результат подготовки130видео без обучения/GPU.
  > 
- **Нельзя заявлять**, 2026-10-25 (f5c5fd1a88bcd2de35479d49bd5f06a8ed2beee8) — Независимо проверен полный результат подготовки130видео без обучения/GPU.
  > A previous visible same-class region is not identity evidence; stale/wrong track IDs and distractors may defeat learning. This is an availability/oracle gate on repeated train26, not predicted model performance/HOTA or proof of temporal causality.

## Критерии приёмки (записаны до прогона)
- ✅ For each unchanged query, scan stored episode frames strictly earlier than current in reverse chronological order; choose latest frame with supported same-class real high-score regions under unchanged regions() API. Initial four observed exemplar frames are eligible. Never read ep_identity/ground-truth when selecting. Never use future frames or another episode. Preserve current state/history/choices exactly.  
  *runs/z13_temporal_observation_gate_20261025/independent_audit.json;report.json*
- ✅ All37296 current UID/episode/frame/track/choice bindings reproduce exactly; every selected reference strictly earlier/same episode; no true identity selection. Candidate resource admission: supported pair count>14252 and paired-video95%lower of oracle-reserve increment over old eligible pairs>0; expanded oracle mean>=.02 and lower>0. Existing model efficacy criteria do not change. If failed do not train this construction.  
  *runs/z13_temporal_observation_gate_20261025/independent_audit.json;report.json*

## Доказательства
- Правила и критерии до проверки130видео: `runs/z13_temporal_observation_gate_20261025/protocol.json` · 
- Проверенная готовность ссылок на прошлые наблюдения; не эффективность модели: `runs/z13_temporal_observation_gate_20261025/independent_audit.json` · 

