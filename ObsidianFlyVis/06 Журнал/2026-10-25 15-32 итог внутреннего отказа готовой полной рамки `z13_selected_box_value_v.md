---
tags:
- журнал
дата: '2026-10-25'
время: '15:32'
---

# 2026-10-25 15:32 — итог внутреннего отказа готовой полной рамки `z13_selected_box_value_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_selected_box_value_20261025/report.json → paired/counts: {'paired': {'Dvalue-baseline': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}, 'Dvalue-Dprior': {'mean_per_video': 0.0021500252590945096, 'ci95': [-0.0016259243890012205, 0.0064890309434590026], 'better': 9, 'n': 26}, 'Dvalue-C': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}, 'Dvalue-R1': {'mean_per_video': 0.0005940990118390587, 'ci95': [-0.00020024782740678485, 0.0018104264734094926], 'better': 3, 'n': 26}, 'Dvalue-geometry': {'mean_per_video': 0.0006220862470862471, 'ci95': [0.0, 0.0017569930069930072], 'better': 2, 'n': 26}}, 'counts': {'D': {'rows': 8326, 'known': 5822, 'supported': 3820, 'applied': 49, 'beneficial': 2, 'harmful': 0, 'unknown': 2504}, 'C': {'rows': 8326, 'known': 5822, 'supported': 3890, 'applied': 49, 'beneficial': 0, 'harmful': 0, 'unknown': 2504}, 'R1': {'rows': 8326, 'known': 5822, 'supported': 3448, 'applied': 29, 'beneficial': 1, 'harmful': 1, 'unknown': 2504}, 'geometry': {'rows': 8326, 'known': 5822, 'supported': 3865, 'applied': 25, 'beneficial': 0, 'harmful': 0, 'unknown': 2504}}}

*Вывод.* Проверен более узкий внутренний отказ для одного готового ответа. Все четыре модели и строки независимо пересчитаны из старых меток фактического действия. Это не завершение выбора среди всех рамок и не HOTA;26видео повторно используются.

*Решение.* Pass -> separate frozen matched closed-loop parity/price/HOTA with all controls. Fail -> preserve this exact single-proposal value decoder; no lambda/threshold/feature/normalizer grid on26. Full all-region supervision remains untested, with4videos preserved. Основной критерий: False

HOTA и процент к стенду не измерены,доляпотолканеустановлена.
