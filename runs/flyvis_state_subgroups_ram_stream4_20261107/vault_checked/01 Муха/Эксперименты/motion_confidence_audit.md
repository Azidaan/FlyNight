---
aliases:
- 'Аудит motion_confidence FlyVis: антиинформативна в рабочем диапазоне'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/calibration
- метка/motion-confidence
- метка/reliability
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-09-13'
завершён: '2026-09-13'
---

# 🔴 Аудит motion_confidence FlyVis: антиинформативна в рабочем диапазоне

↑ [[01 Муха/Темы/Движение мухи|Движение мухи]] · **Исход:** отрицательный · `motion_confidence_audit`

Текущая motion_confidence (покрытие фасеток / (1 + разброс векторов)) — эвристика внутреннего согласия, а не вероятность правильного движения.

## Вопрос
Ранжирует ли motion_confidence сегменты движения по фактической ошибке?

## Ожидалось (до прогона)
У верхней четверти по confidence ошибка движения ниже, чем на полном наборе.

## Что сделано
В run_bdd_video_rate_experiment.py добавлена зависимость motion_confidence от фактической ошибки по корзинам и верхним долям; 16 547 размеченных сегментов; пять тестов.

## Получено
Фрагменты: средняя нормированная ошибка 0.7682 на всех сегментах и 1.2813 у верхней четверти; доля приемлемых ошибок (≤ 0.25 диагонали) падает с 31.49% до 8.67%. Полный кадр: 1.0894 → 1.3732, 21.50% → 11.33%. В 13 из 14 сочетаний «видео × режим» верхняя четверть хуже полного набора.

## Интерпретация
Мера не просто не откалибрована: она ранжирует движение в неправильную сторону.

## Решение
Исключена из роли готового q_match; остаётся диагностическим внутренним весом. Этап 6 (голова достоверности) — не выполнен.

## Ограничения
Адрес значений внутри motion_endpoint в summary.json не зафиксирован в каталоге — числа взяты из отчёта.

## Условия
- **dataset:** BDD100K, исходные видео
- **split:** 7 видео, 16 547 сегментов
- **tracker:** FlyVisTracker

## Критерии приёмки (записаны до прогона)
- ❌ верхняя четверть по confidence точнее полного набора  
  *13 из 14 сочетаний — наоборот*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [motion_err_norm↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%BD%D0%BE%D1%80%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%B0%D1%8F%20%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B0%20%D0%B4%D0%B2%D0%B8%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F) | 1.2813 | 0.7682 | фрагменты: верхняя четверть по confidence / все сегменты | `runs/flyvis_stress_gate/combined/summary.json` ·  |
| [motion_err_norm↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%BD%D0%BE%D1%80%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%B0%D1%8F%20%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B0%20%D0%B4%D0%B2%D0%B8%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F) | 1.3732 | 1.0894 | полный кадр: верхняя четверть / все | `runs/flyvis_stress_gate/combined/summary.json` ·  |

## Доказательства
- надёжность motion confidence: `FLYVIS_STRESS_AND_SECOND_ASSOCIATION_REPORT.md` · §6
- ошибка движения по сегментам: `runs/flyvis_stress_gate/combined/summary.json` · motion_endpoint

## Код
- `scripts/run_bdd_video_rate_experiment.py`
- `tests/test_motion_confidence_reliability.py`

## Данные
- `runs/flyvis_stress_gate/combined`

## Коммиты
- `c5c0e20`

