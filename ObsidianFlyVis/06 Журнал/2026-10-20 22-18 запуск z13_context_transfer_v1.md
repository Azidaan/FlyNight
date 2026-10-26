---
tags:
- журнал
дата: '2026-10-20'
время: '22:18'
---

# 2026-10-20 22:18 — запуск z13_context_transfer_v1

↑ [[06 Журнал/Дни/2026-10-20|2026-10-20]]

*Проба.* runs/z13_context_transfer_20261020/technical.json: {"passed": true, "video": "0001542f-ec815219", "timing": {"seconds": 27.139999999999873, "frames": 1722, "episodes": 51}, "directional_center_max_error": 0.0, "directional_peak_max_error": 0.0, "directional_features_max_error": 0.0, "mean_reproduction": {"max_center_difference": 0.0, "max_peak_difference": 0.0, "known_outcomes_match": true}, "cuda_initialized": false, "scientific_efficacy_measured": false}

*Продолжение.* Автор20.10попросил продолжить работу; выполняется обычный CPU-пилот по существующему плану, без GPU/обучения ядра. Самостоятельно установлен конечный лимит 2400с плюс30с завершения; источник полномочий и расчёт цены — full_authorization.json. Фактический запуск и сроки — controller.json: {"pid": 17852, "create_time": 1792523855.4510024, "controller_pid": 21272, "started_at": "2026-10-20T22:17:35.422722+03:00", "soft_timestamp": 1792526255.4227228, "save_stop_at": "2026-10-20T22:57:35.422722+03:00", "hard_stop_at": "2026-10-20T22:58:05.422722+03:00", "command": ["E:\\Dissertation\\.venv\\Scripts\\python.exe", "-u", "E:\\Dissertation\\scripts\\z13_context_transfer_20261020.py", "worker"]}. Кандидат C выбран по вспомогательным данным; исходный отрицательный критерий D не пересматривается. Train_rationale.json: парные интервалы C на train26 включают ноль, поэтому перенос остаётся проверкой неопределённой гипотезы. Проверка не является независимой итоговой оценкой, не запускает HOTA,75или чат-автоматизации.
