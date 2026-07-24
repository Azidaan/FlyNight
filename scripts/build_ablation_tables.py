"""Сводные таблицы ablation (Этап 7): синергия ветвей и парадигма обработки.

Читает summary-файлы всех конфигураций (evaluate_run.py) и строит:

1. ablation_all.csv      — все конфигурации x поднаборы x метрики (mean/std);
2. ablation_by_size.csv  — только AP_small/medium/large (проверка
                           специализации ветвей по размерам объектов);
3. ablation_synergy.csv  — критерий синергии по плану:
       synergy = drb_full − max(lqa_only, fga_only)   [ОСНОВНОЙ]
       synergy_vs_mean = drb_full − (lqa+fga)/2       [справочный]
   Сравнение со средним — слабый критерий: DRB может превосходить среднее,
   оставаясь хуже лучшей ветви (план прямо предупреждает об этом);
4. ablation_paradigm.csv — baseline vs DRB-at-input vs DRB-in-neck
                           (3-й пункт новизны).

Использование: python scripts/build_ablation_tables.py
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_runs import load_summary  # noqa: E402

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
CONFIGS = ('baseline', 'ablation_lqa_only', 'ablation_fga_only',
           'drb_full', 'drb_aug_full', 'ablation_drb_input')
SIZE_METRICS = ('AP_small', 'AP_medium', 'AP_large')


def main() -> int:
    summaries = {}
    for run in CONFIGS:
        try:
            summaries[run] = load_summary(run)
        except FileNotFoundError:
            print(f'[WARN] нет summary для {run} — колонка будет пустой')

    if not summaries:
        print('[FAIL] нет ни одного summary — сначала evaluate_run.py')
        return 1

    keys = sorted({k for s in summaries.values() for k in s})

    # 1. Всё вместе
    with open(RESULTS / 'ablation_all.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'metric']
                   + [f'{r}_{c}' for r in summaries for c in ('mean', 'std')])
        for subset, metric in keys:
            row = [subset, metric]
            for run in summaries:
                if (subset, metric) in summaries[run]:
                    m, s = summaries[run][(subset, metric)]
                    row += [f'{m:.5f}', f'{s:.5f}']
                else:
                    row += ['', '']
            w.writerow(row)

    # 2. По размерам объектов
    with open(RESULTS / 'ablation_by_size.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'metric'] + list(summaries))
        for subset, metric in keys:
            if not any(metric.startswith(sm) for sm in SIZE_METRICS):
                continue
            row = [subset, metric]
            for run in summaries:
                v = summaries[run].get((subset, metric))
                row.append(f'{v[0]:.5f}' if v else '')
            w.writerow(row)

    # 3. Синергия (нужны все три конфигурации Серии I)
    need = ('ablation_lqa_only', 'ablation_fga_only', 'drb_full')
    if all(r in summaries for r in need):
        with open(RESULTS / 'ablation_synergy.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['subset', 'metric', 'lqa_only', 'fga_only', 'drb_full',
                        'best_single', 'synergy', 'synergy_vs_mean',
                        'synergy_positive'])
            for subset, metric in keys:
                vals = [summaries[r].get((subset, metric)) for r in need]
                if any(v is None for v in vals):
                    continue
                lqa, fga, full = (v[0] for v in vals)
                best = max(lqa, fga)
                syn = full - best
                syn_mean = full - (lqa + fga) / 2
                w.writerow([subset, metric, f'{lqa:.5f}', f'{fga:.5f}',
                            f'{full:.5f}', f'{best:.5f}', f'{syn:+.5f}',
                            f'{syn_mean:+.5f}', 'yes' if syn > 0 else 'no'])
        print(f'[OK ] {RESULTS / "ablation_synergy.csv"}')
    else:
        print(f'[WARN] синергия пропущена — нужны {need}')

    # 4. Парадигма обработки
    need_p = ('baseline', 'ablation_drb_input', 'drb_full')
    if all(r in summaries for r in need_p):
        with open(RESULTS / 'ablation_paradigm.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['subset', 'metric', 'baseline', 'drb_at_input',
                        'drb_in_neck', 'neck_vs_input'])
            for subset, metric in keys:
                vals = [summaries[r].get((subset, metric)) for r in need_p]
                if any(v is None for v in vals):
                    continue
                base, inp, neck = (v[0] for v in vals)
                w.writerow([subset, metric, f'{base:.5f}', f'{inp:.5f}',
                            f'{neck:.5f}', f'{neck - inp:+.5f}'])
        print(f'[OK ] {RESULTS / "ablation_paradigm.csv"}')
    else:
        print(f'[WARN] парадигма пропущена — нужны {need_p}')

    print(f'[OK ] {RESULTS / "ablation_all.csv"}')
    print(f'[OK ] {RESULTS / "ablation_by_size.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
