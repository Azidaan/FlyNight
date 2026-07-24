"""Оценка baseline-моделей на поднаборах (Этап 3).

Для каждого обученного сида (42, 123, 2024):
- детекция на Normal/Night/Rain/Fog (полный val): mAP, P, R, per-class,
  AP_small/medium/large;
- Δ-метрики устойчивости относительно Normal;
- опционально (--tracking) — трекинг ByteTrack + HOTA/IDF1/MOTA через
  официальные bdd100k eval tools (долго: ~30–60 мин на модель).

Выходы:
    experiments/results/baseline_metrics.csv   — seed, subset, metric, value
    experiments/results/baseline_summary.csv   — mean/std по сидам
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


def eval_tracking(model_path: str, seed: int) -> dict:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from evaluation.tracking_metrics import evaluate_tracking_bdd, run_tracker_on_videos

    det_json = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'bdd100k_labels_images_val.json'
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(p.name for p in
                     (PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val').iterdir()
                     if p.is_dir())
    out: dict[str, dict] = {}
    for subset in SUBSETS:
        vids = filter_videos_by_subset(attrs, all_ids, subset)
        track_dir = PROJECT_ROOT / 'runs' / 'track' / f'baseline_seed{seed}'
        run_tracker_on_videos(model_path, vids, track_dir)
        res = evaluate_tracking_bdd(
            track_dir, vids,
            RESULTS / f'tracking_baseline_seed{seed}_{subset}.json')
        out[subset] = res
        print(f'[OK ] трекинг {subset}: {res}')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracking', action='store_true',
                    help='дополнительно посчитать метрики трекинга (долго)')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for seed in args.seeds:
        best = PROJECT_ROOT / 'weights' / f'baseline_seed{seed}' / 'best.pt'
        if not best.exists():
            print(f'[WARN] нет весов {best} — сид {seed} пропущен')
            continue
        print(f'\n=== Сид {seed}: детекция по поднаборам ===', flush=True)
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
            for subset, metrics in eval_tracking(str(best), seed).items():
                for metric, value in metrics.items():
                    rows.append({'seed': seed, 'subset': f'track_{subset}',
                                 'metric': metric, 'value': value})

    if not rows:
        print('[FAIL] Нет ни одного обученного сида')
        return 1

    metrics_csv = RESULTS / 'baseline_metrics.csv'
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'subset', 'metric', 'value'])
        writer.writeheader(); writer.writerows(rows)

    # сводка mean/std по сидам
    grouped: dict[tuple, list[float]] = {}
    for r in rows:
        if isinstance(r['value'], (int, float)):
            grouped.setdefault((r['subset'], r['metric']), []).append(float(r['value']))
    summary_csv = RESULTS / 'baseline_summary.csv'
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
