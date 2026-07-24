"""Сравнительная таблица конфигураций (Этапы 5-7).

Сопоставляет summary-файлы двух и более конфигураций (baseline, drb_full,
drb_aug_full, ablation_*). Для каждой пары (поднабор, метрика) считает
прирост относительно ПЕРВОЙ указанной конфигурации и значимость.

Критерий значимости плана (Этап 5): прирост значим, если |Δ| больше суммы
стандартных отклонений двух конфигураций.

Выход: experiments/results/<name>.csv со столбцами
    subset, metric, <run>_mean, <run>_std, ..., delta, significant

Использование (покрывает плановые compare_drb_vs_baseline.py, compare_drb_aug.py):
    python scripts/compare_runs.py --runs baseline drb_full --out drb_vs_baseline
    python scripts/compare_runs.py --runs baseline drb_full drb_aug_full --out drb_aug_vs_drb_vs_baseline
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / 'experiments' / 'results'


def load_summary(run: str) -> dict[tuple[str, str], tuple[float, float]]:
    """(subset, metric) -> (mean, std) из <run>_summary.csv."""
    path = RESULTS / f'{run}_summary.csv'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} не найден — сначала python scripts/evaluate_run.py --run {run}')
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[(r['subset'], r['metric'])] = (float(r['mean']), float(r['std']))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', required=True,
                    help='конфигурации; первая — точка отсчёта для delta')
    ap.add_argument('--out', required=True, help='имя выходного csv без расширения')
    args = ap.parse_args()

    summaries = {run: load_summary(run) for run in args.runs}
    base = args.runs[0]
    last = args.runs[-1]

    keys = sorted(set.intersection(*(set(s.keys()) for s in summaries.values())))
    if not keys:
        print('[FAIL] нет общих (subset, metric) между конфигурациями')
        return 1

    out_path = RESULTS / f'{args.out}.csv'
    fieldnames = (['subset', 'metric']
                  + [f'{r}_{s}' for r in args.runs for s in ('mean', 'std')]
                  + ['delta', 'significant'])
    n_sig = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for subset, metric in keys:
            row = {'subset': subset, 'metric': metric}
            for run in args.runs:
                m, s = summaries[run][(subset, metric)]
                row[f'{run}_mean'] = f'{m:.5f}'
                row[f'{run}_std'] = f'{s:.5f}'
            m_base, s_base = summaries[base][(subset, metric)]
            m_last, s_last = summaries[last][(subset, metric)]
            delta = m_last - m_base
            significant = abs(delta) > (s_base + s_last)
            n_sig += significant
            row['delta'] = f'{delta:+.5f}'
            row['significant'] = 'yes' if significant else 'no'
            w.writerow(row)

    print(f'[OK ] {out_path}: {len(keys)} строк, {n_sig} значимых отличий '
          f'({last} vs {base})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
