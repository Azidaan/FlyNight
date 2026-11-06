"""Ablation агрегации q_i: mean vs min vs median (Этап 9, п. 9.4).

Режим LQA-guided гоняется трижды с разными агрегациями q по рамке
(кэш inference_with_q уже содержит все три массива q_mean/q_min/q_median —
дополнительный инференс не нужен, варьируется только ассоциация).

Выход: experiments/results/q_aggregation_ablation.csv
    (aggregation, subset, metric, value)

Использование (после run_tracking_three_modes.py — кэш уже построен):
    python scripts/q_aggregation_ablation.py --model weights/drb_aug_full_seed42/best.pt
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.run_tracker_grid_search import subset_videos  # noqa: E402
from scripts.tuned_params import resolve_k, resolve_tau  # noqa: E402
from tracking.inference_with_q import build_detection_cache_with_q  # noqa: E402
from tracking.cache_readonly import require_cached_videos  # noqa: E402
from tracking.lqa_guided_tracker import LQAGuidedByteTrack  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402
from tracking.q_aggregation import AGGREGATIONS  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'q_aggregation_ablation.csv'
RUNS = PROJECT_ROOT / 'runs' / 'track_qagg'
Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q'
SUBSETS = ('normal', 'night', 'rain', 'fog')
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', help='веса для построения кэша (не нужны с --cached-only)')
    ap.add_argument('--cache-dir', type=Path, default=Q_CACHE)
    ap.add_argument('--cached-only', action='store_true',
                    help='только читать полный существующий кэш, не подтверждая происхождение весов')
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--tau-base', type=float, default=None,
                    help='перекрыть автоподбор τ (по умолчанию — оптимум сетки)')
    ap.add_argument('--k', type=float, default=None,
                    help='перекрыть автоподбор k (по умолчанию — оптимум свипа)')
    args = ap.parse_args()
    if not args.cached_only and not args.model:
        ap.error('--model обязателен без --cached-only')

    by_subset = subset_videos(args.split, args.subsets)
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    if args.cached_only:
        require_cached_videos(args.cache_dir, args.split, all_ids,
                              required=tuple(f'q_{agg}' for agg in AGGREGATIONS))
    else:
        build_detection_cache_with_q(args.model, all_ids, args.split, args.cache_dir)

    # те же параметры, что и в сравнении режимов — иначе строки несопоставимы
    k_value, k_src = resolve_k(args.k)
    print(f'[INFO] k = {k_value:g} ({k_src})')

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['aggregation', 'subset', 'metric', 'value'])
        for agg in AGGREGATIONS:
            for subset in args.subsets:
                vids = by_subset[subset]
                out_dir = RUNS / f'{agg}_{subset}'
                tau, buf, _ = resolve_tau(subset, args.tau_base)
                t_args = make_tracker_args(track_high_thresh=tau,
                                           track_buffer=buf)
                t_args.lqa_k = k_value
                run_association(args.cache_dir, vids, out_dir, args.split,
                                args=t_args, tracker_cls=LQAGuidedByteTrack,
                                q_key=f'q_{agg}')
                metrics = evaluate_tracking_bdd(
                    out_dir, vids, out_dir / '_eval.json', args.split)
                for mk in METRIC_KEYS:
                    if mk in metrics:
                        writer.writerow([agg, subset, mk, metrics[mk]])
                f.flush()
                print(f'[OK ] {agg}/{subset}: HOTA={metrics.get("HOTA", "?")}',
                      flush=True)
    print(f'[OK ] {RESULTS_CSV}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
