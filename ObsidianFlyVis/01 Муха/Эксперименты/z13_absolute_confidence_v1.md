---
aliases:
- 'З13: абсолютная высота отклика вместо нормировки текущей картой'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-22'
---

# 🟢 З13: абсолютная высота отклика вместо нормировки текущей картой

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_absolute_confidence_v1`

Основной сравнительный критерий не выполнен. Этот механизм отказа/балла не принят; частные сравнения и исправление одного R1примера не заменяют исход D.

## Вопрос
Улучшит ли исходная высота отклика выбор полезных центров D сверх max-z и прежних селекторов?

## Гипотеза
Деление на разброс текущей карты может усиливать пики при отсутствии цели; исходный масштаб отклика может быть полезен, но это не установлено.

## Ожидалось (до прогона)
Фиксированная замена одного балла, прежние центры/ядро и26train-видео; одинаковый отбор84,CPUдо600с.

## Что сделано
2026-10-22

## Получено
runs/z13_absolute_confidence_20261022/report.json → paired: {'C_absolute-C_raw': {'mean_per_video': 0.002786647534555329, 'ci95': [-0.0006239347877480873, 0.007528231223507439], 'better': 4, 'n': 26}, 'C_absolute-C_geometry6': {'mean_per_video': -0.0024340285605962127, 'ci95': [-0.00969987345349989, 0.004658402250915164], 'better': 4, 'n': 26}, 'C_absolute-C_map14': {'mean_per_video': -0.008969320353905684, 'ci95': [-0.02208764782145774, 0.0017876275834129154], 'better': 4, 'n': 26}, 'D_absolute-D_raw': {'mean_per_video': 0.0028528653305407202, 'ci95': [-0.001682863337872062, 0.008025665350098268], 'better': 5, 'n': 26}, 'D_absolute-D_geometry6': {'mean_per_video': 0.0014531317251150185, 'ci95': [-0.0035227868304839676, 0.00699272948556477], 'better': 4, 'n': 26}, 'D_absolute-D_map14': {'mean_per_video': -0.0005791711442874032, 'ci95': [-0.00728483452860217, 0.006059842551972021], 'better': 4, 'n': 26}, 'R1_absolute-R1_raw': {'mean_per_video': 0.0035832396800852787, 'ci95': [9.282337773957886e-05, 0.008795900274236672], 'better': 4, 'n': 26}, 'R1_absolute-R1_geometry6': {'mean_per_video': -0.007558943826471792, 'ci95': [-0.024833196466199053, 0.004677220580931166], 'better': 3, 'n': 26}, 'R1_absolute-R1_map14': {'mean_per_video': -0.001286255867737323, 'ci95': [-0.009093371014718098, 0.0060092904990449985], 'better': 3, 'n': 26}, 'C_absolute-R1_absolute': {'mean_per_video': -5.6665520406618786e-05, 'ci95': [-0.006379116568631947, 0.0059234031405876455], 'better': 4, 'n': 26}, 'D_absolute-R1_absolute': {'mean_per_video': 0.0005695584978701319, 'ci95': [-0.0034042827998738237, 0.005006469553030721], 'better': 4, 'n': 26}, 'D_gain-R1_gain': {'mean_per_video': -0.0007303743495445579, 'ci95': [-0.005782994769837503, 0.004367086658408314], 'better': 4, 'n': 26}}; summary: {'C_raw': {'selected': 84, 'known': 67, 'unknown': 17, 'useful': 2, 'known_nonuseful': 65, 'strictly_worse_center': 1, 'absent': 64}, 'C_absolute': {'selected': 84, 'known': 41, 'unknown': 43, 'useful': 7, 'known_nonuseful': 34, 'strictly_worse_center': 1, 'absent': 30}, 'C_geometry6': {'selected': 84, 'known': 40, 'unknown': 44, 'useful': 7, 'known_nonuseful': 33, 'strictly_worse_center': 20, 'absent': 8}, 'C_map14': {'selected': 84, 'known': 47, 'unknown': 37, 'useful': 14, 'known_nonuseful': 33, 'strictly_worse_center': 20, 'absent': 9}, 'D_raw': {'selected': 84, 'known': 64, 'unknown': 20, 'useful': 2, 'known_nonuseful': 62, 'strictly_worse_center': 0, 'absent': 62}, 'D_absolute': {'selected': 84, 'known': 50, 'unknown': 34, 'useful': 7, 'known_nonuseful': 43, 'strictly_worse_center': 14, 'absent': 27}, 'D_geometry6': {'selected': 84, 'known': 38, 'unknown': 46, 'useful': 5, 'known_nonuseful': 33, 'strictly_worse_center': 15, 'absent': 13}, 'D_map14': {'selected': 84, 'known': 46, 'unknown': 38, 'useful': 11, 'known_nonuseful': 35, 'strictly_worse_center': 14, 'absent': 16}, 'R1_raw': {'selected': 84, 'known': 48, 'unknown': 36, 'useful': 1, 'known_nonuseful': 47, 'strictly_worse_center': 13, 'absent': 30}, 'R1_absolute': {'selected': 84, 'known': 65, 'unknown': 19, 'useful': 7, 'known_nonuseful': 58, 'strictly_worse_center': 29, 'absent': 20}, 'R1_geometry6': {'selected': 84, 'known': 59, 'unknown': 25, 'useful': 8, 'known_nonuseful': 51, 'strictly_worse_center': 13, 'absent': 26}, 'R1_map14': {'selected': 84, 'known': 60, 'unknown': 24, 'useful': 6, 'known_nonuseful': 54, 'strictly_worse_center': 21, 'absent': 23}}. Полный train26,3варианта, по84выбранных ответа, исходные центры/баллы воспроизведены; CUDA/dev/HOTA не использовались.

## Интерпретация
Основной сравнительный критерий не выполнен. Этот механизм отказа/балла не принят; частные сравнения и исправление одного R1примера не заменяют исход D.

## Решение
Advance only if all equal-quota D criteria pass. If not, stop scalar confidence replacement; do not keep tuning this selector on train26. Retain original centered readout and move to a separately specified model-fitting change. No dev tuning or HOTA rescue.

## Нельзя заявлять
Не калиброванная вероятность, не HOTA и не преимущество анатомии/обучения ядра. Повторно используемые train26; один R1dev-пример мотивирует вопрос, но не задаёт параметры. Критерий90%/50 сохраняется.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-22 (427f98398180dda091a19b1ceb9368f55b266573) — Проверенный полный итог CPU-сравнения с прежними контролями.
  > До исходов зарегистрирована одна замена балла.

## Критерии приёмки (записаны до прогона)
- ✅ All C/D/R1 raw centers<1e-6px and original z-scores<1e-9 to saved last rows; all maps/hash unchanged.  
  *runs/z13_absolute_confidence_20261022/report.json*
- ❌ D absolute-D raw, D absolute-D geometry6 and D absolute-D map14: all paired-video95%lower>0 for selected_success_fraction_positive at1%equal quota.  
  *runs/z13_absolute_confidence_20261022/report.json*
- ❌ D absolute-R1 absolute paired-video95%lower>0. Controls keep the same history and maps; no anatomy claim.  
  *runs/z13_absolute_confidence_20261022/report.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [selected_success_fraction_positive↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%BE%D0%B6%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D1%85%20%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B5%D0%B2%20%D1%81%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D0%BC%20%D0%B2%D1%8B%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC%20%D0%BE%D1%82%D0%B2%D0%B5%D1%82%D0%BE%D0%BC) | 0.004367667107908139 | 0.0015148017773674188 | train26;D absolute | `runs/z13_absolute_confidence_20261022/review.json` · means.D_absolute |

## Доказательства
- Одно изменение и прежние контроли до расчёта: `runs/z13_absolute_confidence_20261022/protocol.json` · 
- Предыдущий запрет внешнего кольца не улучшил D: `runs/z13_boundary_readout_20261022/report.json` · 
- Полный исход, контроль неизменности и фактическое исполнение: `runs/z13_absolute_confidence_20261022/report.json` · 
- Полный исход, контроль неизменности и фактическое исполнение: `runs/z13_absolute_confidence_20261022/reproduction.json` · 
- Полный исход, контроль неизменности и фактическое исполнение: `runs/z13_absolute_confidence_20261022/controller_result.json` · 
- Полный исход, контроль неизменности и фактическое исполнение: `runs/z13_absolute_confidence_20261022/launch.json` · 

## Код
- `tracking/z13_absolute_confidence.py`
- `tests/test_z13_absolute_confidence.py`
- `scripts/z13_absolute_confidence_20261022.py`

## Коммиты
- `427f98398180dda091a19b1ceb9368f55b266573`

