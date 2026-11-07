---
tags:
- журнал
дата: '2026-10-25'
время: 03:33
---

# 2026-10-25 03:33 — итог обучаемого регионального считывателя `z13_region_assignment_v1`

↑ [[06 Журнал/Дни/2026-10-25|2026-10-25]]

*Получено.* runs/z13_region_assignment_20261025/recovery2/training/report.json → paired: {'D-baseline': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}, 'D-C': {'mean_per_video': -3.141057580732824e-05, 'ci95': [-0.0001888788712676626, 0.00011488295758832168], 'better': 2, 'n': 26}, 'D-R1': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}, 'D-geometry': {'mean_per_video': 2.0918047949866945e-05, 'ci95': [-8.422162375327225e-05, 0.00013772790303388633], 'better': 2, 'n': 26}};12моделей,трисидa,порог fit20; первичные строки и все контроли сохранены.

*Вывод.* Полный внутренний узел оценён по точным однокадровым назначениям. Это не HOTA и не изоляция одного свойства ядра. Контроль геометрии отделяет пользу внешнего применения; C/R1 отделяют вклад D.

*Решение.* Complete criteria pass -> separate matched closed-loop test with frozen thresholds and equal MASA/raw inputs. Fail -> preserve all controls and negative seed results, do not run expensive HOTA for this version. Technical failure -> preserve sources/logs, repair only verified service issue within original phase/global budgets. Основной итог: False

HOTA и процент к стенду не измерены; доля оракульного потолка не переносится в HOTA. Подтверждённые сравнения и архитектурную веху проверить отдельно по первичным интервалам, не красить весь узел зелёным по частному числу.
