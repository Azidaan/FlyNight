---
aliases:
- Проверка влияния ошибки дублирования строк в признаках цепочки на выводы chain_filter_rank_v1
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P1
- метка/tracker
- метка/detection
- муха
- проверка-гипотезы/завершена
исход: положительный
направление: FlyVis
начат: '2026-10-09'
---

# 🟢 Проверка влияния ошибки дублирования строк в признаках цепочки на выводы chain_filter_rank_v1

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `chain_feature_dedup_check_v1`

Ошибка 09.10: в признаках цепочки (chain_task) строки трекера собирались по номеру для каждой жизни цепочки — номер с несколькими жизнями учтён несколько раз. Разбор автора: прежние результаты ограничены до проверки влияния. Здесь этап ранжирования chain_filter_rank повторяется с исправлением (строки внутри интервала каждой записи, --fixed, выход runs/chain_filter_rank_fixed) при том же всём остальном; сравнение AP и удаления при бюджете с прежними числами. Работа без выбора автора. scripts/chain_filter_rank.py --fixed.

## Вопрос
Меняет ли исправление дублирования строк выводы [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) (муха ухудшает ранжирование цепочек на test/val7)?

## Гипотеза
Средние и квантили почти не меняются — выводы сохраняются.

## Ожидалось (до прогона)
Диагностика: AP без мухи / с мухой и разность с ДИ на dev (вне обучения), test, val7 — исправленные против прежних (runs/chain_filter_rank/report_rank.json); вывод сохраняется, если знак и выход ДИ за 0 у разности «с мухой − без мухи» на test и val7 те же.

## Что сделано
2026-10-09

## Получено
runs/chain_filter_rank_fixed/report_rank.json против runs/chain_filter_rank/report_rank.json. AP (dev — вне обучения): dev: исправлено — без мухи 0.5388, с мухой 0.5363, разность -0.0025 [-0.0282; +0.0202]; прежнее — 0.5340 / 0.5337, -0.0003 [-0.0257; +0.0216]; test: исправлено — без мухи 0.5696, с мухой 0.5173, разность -0.0523 [-0.0708; -0.0293]; прежнее — 0.5643 / 0.5139, -0.0505 [-0.0691; -0.0267]; val7: исправлено — без мухи 0.4677, с мухой 0.3825, разность -0.0852 [-0.1143; -0.0399]; прежнее — 0.4681 / 0.3785, -0.0896 [-0.1203; -0.0528]. Бюджет 1 %: без мухи — исправлено порог 0.85, test — ложных 372, верных 148, потеряно 18, прежнее порог 0.85, test — ложных 372, верных 148, потеряно 19; с мухой — исправлено порог 0.95, test — ложных 344, верных 279, потеряно 28, прежнее порог 0.95, test — ложных 371, верных 309, потеряно 28.

## Интерпретация
Исправление дублирования строк почти не меняет числа: вывод [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) сохраняется — муха в голове цепочки (сырые сигнатуры) ухудшает ранжирование на test и val7 (ДИ ниже 0), на dev без разницы.

## Решение
Оговорка о дублировании строк в карточках цепочечных фильтров снимается для [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) (вывод проверен); для [chain_filter_fly_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_fly_v1), [chain_filter_compact_fly_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_compact_fly_v1), [chain_filter_train130_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_train130_v1) влияние отдельно не проверялось — по аналогии ожидается малым.

## Нельзя заявлять
Проверен этап ранжирования [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1); HOTA с исправленными признаками не пересчитывалась.

## Условия
- **baseline:** [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) (с ошибкой)
- **variant:** исправленные признаки цепочки
- **dataset:** BDD100K MOT train_cache dev 35 / test 35 + val7

## Критерии приёмки (записаны до прогона)
- ✅ вывод [chain_filter_rank_v1↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Fchain_filter_rank_v1) сохраняется: AP с мухой − без мухи на test и val7 — тот же знак, ДИ ниже 0  
  *test -0.0523 [-0.0708; -0.0293], val7 -0.0852 [-0.1143; -0.0399] — runs/chain_filter_rank_fixed/report_rank.json, AP_diff*

## Доказательства
- прежние числа: `runs/chain_filter_rank/report_rank.json` · AP_diff

## Код
- `scripts/chain_filter_head.py`
- `scripts/chain_filter_rank.py`

## Данные
- `runs/chain_filter_rank_fixed`

