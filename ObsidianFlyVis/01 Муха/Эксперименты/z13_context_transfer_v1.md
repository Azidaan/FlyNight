---
aliases:
- 'З13: отдельный перенос пространственного считывания C и контроля R1'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/Z13
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-20'
---

# 🟢 З13: отдельный перенос пространственного считывания C и контроля R1

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_context_transfer_v1`

Пространственное считывание C не прошло составные критерии переноса; преимущество над входом глаза подтверждено только для доли полезных центров — runs/z13_context_transfer_20261020/report.json.

## Вопрос
Переносится ли вспомогательный результат пространственного считывания C на другие видео?

## Гипотеза
Фиксированное пространственное считывание C может улучшить центры и выбор поправок; перенос не установлен.

## Ожидалось (до прогона)
C directional minus C mean: paired video95%CI lower>0 and mean>=0.02; C directional minus R1 directional lower>0. At1% of all rows: C directional map utility minus C mean map utility, C directional geometry, and R1 directional map utility each paired video95%CI lower>0. Both localization and selection required.

## Что сделано
2026-10-20

## Получено
runs/z13_context_transfer_20261020/report.json: {"localization": {"C_directional-C_mean": {"mean_per_video": 0.03267535168228374, "ci95": [-0.010413418156841233, 0.06969899803558438], "better": 19, "n": 35}, "C_directional-R1_directional": {"mean_per_video": 0.129322027258749, "ci95": [0.09235018641430977, 0.17013771438341696], "better": 30, "n": 35}}, "selection_at1percent": {"C_context_map-C_mean_map": {"mean_per_video": 0.0015218141077948408, "ci95": [-0.004014787783516008, 0.006904105522475064], "better": 6, "n": 35}, "C_context_map-C_context_geometry": {"mean_per_video": 0.004228555795885166, "ci95": [-0.0004511555904413045, 0.009657332282506495], "better": 6, "n": 35}, "C_context_map-R1_context_map": {"mean_per_video": -0.0014848069778098628, "ci95": [-0.010724112582772444, 0.006503061243169224], "better": 6, "n": 35}}, "localization_pass": false, "selection_pass": false, "proceed": false}; coverage.C.map_utility.0.01: {"selected": 94, "known": 58, "unknown": 36, "useful": 18, "harmful": 40, "useful_share_known": 0.3103448275862069}

## Интерпретация
На этих видео интервал добавки к прежней C включает ноль; это неопределённость, а не доказательство равенства, бесполезности FlyVis или определённой причины неуспеха.

## Решение
context_transfer_not_confirmed; no_dev_retuning_or_tracker_application

## Нельзя заявлять
Не установлены улучшение HOTA, польза стенду, преимущество обучения ядра D, независимое обобщение или безопасность неизвестных исходов; повторно использованный dev35 и глобальный отбор 1% не являются слепой проверкой и рабочим правилом применения.

## Прежние версии (до исправлений)
- **Получено**, 2026-10-20 (d775314251667690af4ac83be58e59b59bfda3d7) — Служебная ошибка сборки исправлена отдельно; исходный код завершения и прежний частичный итог сохранены
  > runs/z13_context_transfer_20261020/controller_result.json: {"exit_code": 1, "worker_ended": true, "hard_timeout": false, "ended": "2026-10-20T19:41:56.015605+00:00", "chat_calls": 0}
- **Интерпретация**, 2026-10-20 (d775314251667690af4ac83be58e59b59bfda3d7) — Служебная ошибка сборки исправлена отдельно; исходный код завершения и прежний частичный итог сохранены
  > Сохранён частичный расчёт без вывода об эффективности.
- **Решение**, 2026-10-20 (d775314251667690af4ac83be58e59b59bfda3d7) — Служебная ошибка сборки исправлена отдельно; исходный код завершения и прежний частичный итог сохранены
  > save_partial_without_efficiency_verdict
- **Нельзя заявлять**, 2026-10-20 (d775314251667690af4ac83be58e59b59bfda3d7) — Служебная ошибка сборки исправлена отдельно; исходный код завершения и прежний частичный итог сохранены
  > Нет HOTA,75отложенных, слепогоdev или гарантии безопасности неизвестных исходов. Это новая версия считывания C, не доказательство пользы дообучения D.

## Критерии приёмки (записаны до прогона)
- ✅ Unchanged mean centers<0.5px, scores<0.001, same labels/kind/useful on every video. Technical C directional center<0.5px, score<0.001, map features<1e-10 against the saved same training video. Normalization exactly equals saved dev normalization.  
  *runs/z13_context_transfer_20261020/publication_state_audit.json*
- ❌ C directional minus C mean: paired video95%CI lower>0 and mean>=0.02; C directional minus R1 directional lower>0.  
  *runs/z13_context_transfer_20261020/report.json*
- ❌ At1% of all rows: C directional map utility minus C mean map utility, C directional geometry, and R1 directional map utility each paired video95%CI lower>0. Both localization and selection required.  
  *runs/z13_context_transfer_20261020/report.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [selected_success_fraction_positive↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%BE%D0%B6%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D1%85%20%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B5%D0%B2%20%D1%81%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D0%BC%20%D0%B2%D1%8B%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC%20%D0%BE%D1%82%D0%B2%D0%B5%D1%82%D0%BE%D0%BC) | 0.009037738869876603 | 0.007515924762081762 | dev35; глобальный отбор 1% всех строк | `runs/z13_context_transfer_20261020/recovery_summary_audit.json` · levels_mean_per_video.C_context_map |

## Доказательства
- Условия зафиксированы до переноса; основание запуска и конечный CPU-бюджет сохранены в full_authorization.json: `runs/z13_context_transfer_20261020/protocol.json` · 
- Точное воспроизведение направления и прежнего контроля; это техническая готовность: `runs/z13_context_transfer_20261020/technical.json` · 
- Полный либо частичный первичный итог: `runs/z13_context_transfer_20261020/controller_result.json` · 
- Точное воспроизведение направления и прежнего контроля; это техническая готовность: `runs/z13_context_transfer_20261020/technical.json` · 
- Полные сравнения по заранее заданным критериям: `runs/z13_context_transfer_20261020/report.json` · 
- Предсказания и исходники неизменны; неизвестные исходы не подменены: `runs/z13_context_transfer_20261020/recovery_summary_audit.json` · 
- Отдельное успешное завершение сборки, прежняя ошибка сохранена: `runs/z13_context_transfer_20261020/recovery_summary_controller_result.json` · 

## Код
- `scripts/z13_context_transfer_20261020.py`
- `scripts/z13_context_transfer_progress.ps1`
- `scripts/z13_record_context_transfer_20261020.py`
- `scripts/z13_context_transfer_service_20261020.py`
- `scripts/z13_context_summary_recovery_20261020.py`
- `scripts/z13_context_summary_recovery_progress.ps1`
- `scripts/z13_record_context_summary_recovery_20261020.py`
- `tests/test_z13_context_summary_recovery.py`

## Данные
- `runs/z13_context_transfer_20261020`

## Коммиты
- `d775314251667690af4ac83be58e59b59bfda3d7`

