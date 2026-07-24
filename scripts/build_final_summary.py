"""Сборка финальной сводной таблицы работы (Этап 10, подэтап 10.4).

Сводит все накопленные CSV из experiments/results/ в единую структуру
experiments/results/final/: центральная main_results.csv (конфигурация x
поднабор x метрика) + копии исходных таблиц. На неё ссылаются все главы
диссертации.

Использование: python scripts/build_final_summary.py
"""
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / 'experiments' / 'results'
FINAL = RESULTS / 'final'

SUMMARY_RUNS = ('baseline', 'drb_full', 'drb_aug_full',
                'ablation_lqa_only', 'ablation_fga_only', 'ablation_drb_input')
COPY_TABLES = ('tracking_param_search.csv', 'tracking_three_modes.csv',
               'q_aggregation_ablation.csv', 'confidence_qi_correlation.csv',
               'drb_learned_weights.csv', 'performance_table.csv',
               'ablation_synergy.csv', 'ablation_paradigm.csv',
               'ablation_by_size.csv', 'brightness_per_video.csv')


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / 'all_csvs').mkdir(exist_ok=True)

    # Центральная таблица: все конфигурации в длинном формате
    rows = []
    for run in SUMMARY_RUNS:
        path = RESULTS / f'{run}_summary.csv'
        if not path.exists():
            print(f'[WARN] нет {path.name}')
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append({'config': run, **r})
    if rows:
        with open(FINAL / 'main_results.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'[OK ] {FINAL / "main_results.csv"} ({len(rows)} строк, '
              f'{len({r["config"] for r in rows})} конфигураций)')

    # Копии остальных таблиц
    n = 0
    for name in COPY_TABLES:
        src = RESULTS / name
        if src.exists():
            shutil.copy2(src, FINAL / 'all_csvs' / name)
            n += 1
    print(f'[OK ] скопировано таблиц: {n}/{len(COPY_TABLES)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
