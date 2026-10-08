---
aliases:
- Проверка влияния ошибки дублирования строк в признаках цепочки на выводы chain_filter_rank_v1
tags:
- эксперимент
- исход/running
- направление/flyvis
- метка/fly-problem-P1
- метка/tracker
- метка/detection
исход: идёт
направление: FlyVis
начат: '2026-10-09'
---

# 🔵 Проверка влияния ошибки дублирования строк в признаках цепочки на выводы chain_filter_rank_v1

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** идёт · `chain_feature_dedup_check_v1`

Ошибка 09.10: в признаках цепочки (chain_task) строки трекера собирались по номеру для каждой жизни цепочки — номер с несколькими жизнями учтён несколько раз. Разбор автора: прежние результаты ограничены до проверки влияния. Здесь этап ранжирования chain_filter_rank повторяется с исправлением (строки внутри интервала каждой записи, --fixed, выход runs/chain_filter_rank_fixed) при том же всём остальном; сравнение AP и удаления при бюджете с прежними числами. Работа без выбора автора. scripts/chain_filter_rank.py --fixed.

## Вопрос
Меняет ли исправление дублирования строк выводы [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) (муха ухудшает ранжирование цепочек на test/val7)?

## Гипотеза
Средние и квантили почти не меняются — выводы сохраняются.

## Ожидалось (до прогона)
Диагностика: AP без мухи / с мухой и разность с ДИ на dev (вне обучения), test, val7 — исправленные против прежних (runs/chain_filter_rank/report_rank.json); вывод сохраняется, если знак и выход ДИ за 0 у разности «с мухой − без мухи» на test и val7 те же.

## Условия
- **baseline:** [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) (с ошибкой)
- **variant:** исправленные признаки цепочки
- **dataset:** BDD100K MOT train_cache dev 35 / test 35 + val7

## Критерии приёмки (записаны до прогона)
- ⏳ вывод [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) сохраняется: AP с мухой − без мухи на test и val7 — тот же знак, ДИ ниже 0

## Доказательства
- прежние числа: `runs/chain_filter_rank/report_rank.json` · AP_diff

## Код
- `scripts/chain_filter_head.py`
- `scripts/chain_filter_rank.py`

## Данные
- `runs/chain_filter_rank_fixed`

