"""Оценка обученной конфигурации на поднаборах (универсальный, Этапы 5-7).

Обобщение evaluate_baseline.py на любую конфигурацию: baseline, drb_full,
drb_aug_full, ablation_lqa_only, ablation_fga_only, ablation_drb_input.
Покрывает плановые скрипты evaluate_drb.py / evaluate_all_ablation.py.

Для каждого сида: детекция на Normal/Night/Rain/Fog (mAP, P, R, per-class,
AP_small/medium/large) + Δ-метрики. Опционально --tracking.

Выходы:
    experiments/results/<run>_metrics.csv   — seed, subset, metric, value
    experiments/results/<run>_summary.csv   — mean/std по сидам

Использование:
    python scripts/evaluate_run.py --run drb_full
    python scripts/evaluate_run.py --run ablation_lqa_only --seeds 42 123
    python scripts/evaluate_run.py --run ablation_drb_input --seeds 42
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection_metrics import evaluate_on_subsets  # noqa: E402
from evaluation.stress_delta import compute_stress_delta  # noqa: E402

SEEDS = (42, 123, 2024)
SUBSETS = ('normal', 'night', 'rain', 'fog')
RESULTS = PROJECT_ROOT / 'experiments' / 'results'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True,
                    help='имя конфигурации: drb_full | ablation_lqa_only | ...')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--tracking', action='store_true')
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in args.seeds:
        best = PROJECT_ROOT / 'weights' / f'{args.run}_seed{seed}' / 'best.pt'
        if not best.exists():
            print(f'[WARN] нет весов {best} — сид {seed} пропущен')
            continue
        print(f'\n=== {args.run} сид {seed}: детекция по поднаборам ===', flush=True)
        det = evaluate_on_subsets(str(best), list(SUBSETS))

        for subset, metrics in det.items():
            for metric, value in metrics.items():
                rows.append({'seed': seed, 'subset': subset,
                             'metric': metric, 'value': value})
        for stress in ('night', 'rain', 'fog'):
            for metric, value in compute_stress_delta(det['normal'], det[stress]).items():
                rows.append({'seed': seed, 'subset': f'delta_{stress}',
                             'metric': metric, 'value': value})

        if args.tracking:
            from scripts.evaluate_baseline import eval_tracking
            for subset, metrics in eval_tracking(str(best), seed).items():
                for metric, value in metrics.items():
                    rows.append({'seed': seed, 'subset': f'track_{subset}',
                                 'metric': metric, 'value': value})

    if not rows:
        print('[FAIL] Нет ни одного обученного сида')
        return 1

    metrics_csv = RESULTS / f'{args.run}_metrics.csv'
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'subset', 'metric', 'value'])
        writer.writeheader(); writer.writerows(rows)

    grouped: dict[tuple, list[float]] = {}
    for r in rows:
        if isinstance(r['value'], (int, float)):
            grouped.setdefault((r['subset'], r['metric']), []).append(float(r['value']))
    summary_csv = RESULTS / f'{args.run}_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'metric', 'mean', 'std', 'n_seeds'])
        for (subset, metric), vals in sorted(grouped.items()):
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            writer.writerow([subset, metric,
                             f'{statistics.mean(vals):.5f}', f'{std:.5f}', len(vals)])

    print(f'\n[OK ] {metrics_csv}')
    print(f'[OK ] {summary_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
