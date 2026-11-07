---
aliases:
- 'Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/flyvis
- метка/tiled
- метка/stress
- метка/occlusion
- метка/30hz
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-09-13'
завершён: '2026-09-13'
---

# 🔴 Стресс-проверка FlyVis: полный кадр и перекрывающиеся фрагменты на семи видео

↑ [[01 Муха/Темы/Движение мухи|Движение мухи]] · **Исход:** отрицательный · `flyvis_stress`

Способен ли текущий FlyVis быть основным источником движения и идентичности трекера на видео ~30 Гц? Нет: провал в ассоциации и идентичности, не в числе рамок.

## Вопрос
Конкурирует ли самостоятельный FlyVisTracker с ByteTrack по идентичности при одинаковых детекциях?

## Ожидалось (до прогона)
Сохранённые ошибка движения, HOTA, AssA, IDF1, IDSw, FP, FN, время и калибровочная зависимость motion_confidence (гейт этапа 1).

## Что сделано
ByteTrack, FlyVis по полному кадру и по фрагментам на семи видео с одинаковыми детекциями; три стресс-видео; окклюзионный оценщик переходов.

## Получено
ByteTrack HOTA 52.44 / AssA 52.97 / IDF1 59.82 / 573 IDSw; FlyVis полный кадр 28.12 / 14.98 / 25.48 / 3844; фрагменты 29.11 / 16.00 / 26.76 / 3384. DetA у FlyVis выше (54.41 против 52.81). Сохранение ID на окклюзионных переходах: 96.42% ByteTrack, 66.05% полный, 70.88% фрагменты. Скорость 31.88 / 20.71 / 17.37 кадр/с.

## Интерпретация
Провал относится к сохранению идентичности и правильной ассоциации во времени; фрагменты помогают на тёмном видео и сложном фоне, но не сокращают разрыв.

## Решение
Текущий декодированный поток FlyVis не подходит для безусловной замены модели движения ByteTrack; этап 1 дорожной карты выполнен.

## Ограничения
Семь видео; линия A не закрывается только из-за низкой первой HOTA (правила остановки дорожной карты).

## Нельзя заявлять
Что FlyVis бесполезен как признак: условная полезность проверяется отдельно гейтом q_match.

## Условия
- **baseline:** ByteTrack на тех же детекциях
- **variant:** flyvis_full_calibrated, flyvis_tiled_calibrated
- **dataset:** BDD100K, исходные видео
- **split:** 7 видео val ~30 Гц
- **detector:** YOLO26s seed42, 960
- **tracker:** FlyVisTracker (full / tiled) против ByteTrack

## Критерии приёмки (записаны до прогона)
- ✅ сохранены HOTA, AssA, IDF1, IDSw, FP, FN, время (гейт этапа 1)  
  *summary.json*
- ❌ FlyVis не хуже ByteTrack по идентичности  
  *AssA 16.0 против 53.0*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 29.112995077709662 | 52.43696923664447 | фрагменты / ByteTrack, 7 видео | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_tiled_calibrated.HOTA |
| [AssA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAssA) | 15.99516434125162 | 52.97481252366944 | фрагменты / ByteTrack | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_tiled_calibrated.AssA |
| [DetA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FDetA) | 54.41457166042599 | 52.81188137743227 | фрагменты / ByteTrack | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_tiled_calibrated.DetA |
| [IDF1↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDF1) | 26.763852076233754 | 59.824299685065476 | фрагменты / ByteTrack | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_tiled_calibrated.IDF1 |
| [IDSw↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FIDSw) | 3384 | 573 | фрагменты / ByteTrack | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_tiled_calibrated.IDSw |
| [HOTA↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FHOTA) | 28.11505488731735 | 52.43696923664447 | полный кадр / ByteTrack | `runs/flyvis_stress_gate/combined/summary.json` · metrics.flyvis_full_calibrated.HOTA |
| [fps↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%BA%D0%B0%D0%B4%D1%80%20%D1%81) | 17.36673272604878 | 31.88206366198927 | фрагменты / ByteTrack, оценка по сумме компонентов | `runs/flyvis_stress_gate/combined/summary.json` · timing.pipeline_estimate.flyvis_tiled_calibrated.fps_from_component_sum |
| [id_kept_pct↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D1%81%D0%BE%D1%85%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5%20ID%20%D0%BF%D1%80%D0%B8%20%D0%BE%D0%BA%D0%BA%D0%BB%D1%8E%D0%B7%D0%B8%D0%B8) | 70.88 | 96.42 | окклюзионные переходы, фрагменты / ByteTrack | `runs/flyvis_stress_gate/combined/occlusion_transitions.json` ·  |

## Доказательства
- метрики трёх вариантов: `runs/flyvis_stress_gate/combined/summary.json` · metrics
- окклюзионные переходы: `runs/flyvis_stress_gate/combined/occlusion_transitions.json` · 
- итог на семи видео и стресс-видео: `FLYVIS_STRESS_AND_SECOND_ASSOCIATION_REPORT.md` · §3–5
- текущее состояние 13.09: [§0↗](obsidian://open?vault=ObsidianFlyVis&file=05%20%D0%9F%D0%BB%D0%B0%D0%BD%2F0.%20%D0%A2%D0%B5%D0%BA%D1%83%D1%89%D0%B5%D0%B5%20%D1%81%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%2013.09.2026)

## Код
- `scripts/run_bdd_video_rate_experiment.py`
- `scripts/summarize_video_rate_experiments.py`
- `scripts/evaluate_occlusion_transitions.py`

## Данные
- `runs/flyvis_stress_gate/combined`

## Коммиты
- `889913f`
- `c5c0e20`

