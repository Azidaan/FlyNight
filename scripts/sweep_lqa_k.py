"""Подбор коэффициента адаптации k для LQA-режима (Этап 9, к п. 9.4).

Зачем. Порог τ базового ByteTrack подбирается сеточным поиском на val
(Этап 8). Если при этом коэффициент k предложенного режима взять «по
умолчанию», сравнение окажется перекошенным — уже в пользу базового
метода. Симметрия требует подбирать k тем же способом и на тех же данных.

τ_i = τ_base − k·(1 − q_i): k = 0 превращает режим в обычный ByteTrack,
большие k слишком опускают порог на «плохих» детекциях. Прогон идёт по
готовому q-кэшу детекций — это CPU-минуты на конфигурацию, не обучение.

Выход: experiments/results/lqa_k_sweep.csv (k, subset, HOTA, IDF1, MOTA, IDSw)
Лучший k отсюда автоматически подхватывает run_tracking_three_modes.py.

Использование:
    python scripts/sweep_lqa_k.py --model weights/drb_aug_full_seed42/best.pt
    python scripts/sweep_lqa_k.py --model ... --k-values 0 0.1 0.2 0.3 0.4
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.run_tracker_grid_search import subset_videos  # noqa: E402
from scripts.tuned_params import resolve_tau  # noqa: E402
from tracking.inference_with_q import build_detection_cache_with_q  # noqa: E402
from tracking.cache_readonly import require_cached_videos  # noqa: E402
from tracking.lqa_guided_tracker import LQAGuidedByteTrack  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'lqa_k_sweep.csv'
RUNS = PROJECT_ROOT / 'runs' / 'lqa_k_sweep'
Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q'
SUBSETS = ('normal', 'night', 'rain', 'fog')
K_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4)
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', help='веса для построения кэша (не нужны с --cached-only)')
    ap.add_argument('--cache-dir', type=Path, default=Q_CACHE)
    ap.add_argument('--cached-only', action='store_true',
                    help='только читать полный существующий кэш, не подтверждая происхождение весов')
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--k-values', nargs='+', type=float, default=list(K_VALUES))
    ap.add_argument('--agg', default='mean')
    args = ap.parse_args()
    if not args.cached_only and not args.model:
        ap.error('--model обязателен без --cached-only')

    by_subset = subset_videos(args.split, args.subsets)
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    print(f'[INFO] уникальных видео: {len(all_ids)}, '
          f'значений k: {len(args.k_values)}')
    if args.cached_only:
        require_cached_videos(args.cache_dir, args.split, all_ids, required=(f'q_{args.agg}',))
    else:
        build_detection_cache_with_q(args.model, all_ids, args.split, args.cache_dir)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[float, str]] = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline='', encoding='utf-8') as f:
            done = {(float(r['k']), r['subset']) for r in csv.DictReader(f)}
        print(f'[INFO] в CSV уже {len(done)} пар (k, поднабор) — пропускаются')

    write_header = not RESULTS_CSV.exists()
    q_key = f'q_{args.agg}'
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['k', 'subset', 'tau_base', 'track_buffer',
                             *METRIC_KEYS])
        for k in args.k_values:
            for subset in args.subsets:
                if (k, subset) in done:
                    continue
                tau, buf, src = resolve_tau(subset)
                vids = by_subset[subset]
                t_args = make_tracker_args(track_high_thresh=tau,
                                           track_buffer=buf)
                t_args.lqa_k = k
                out_dir = RUNS / f'k{k:g}_{subset}'
                run_association(args.cache_dir, vids, out_dir, args.split,
                                args=t_args, tracker_cls=LQAGuidedByteTrack,
                                q_key=q_key)
                metrics = evaluate_tracking_bdd(
                    out_dir, vids, out_dir / '_eval.json', args.split)
                writer.writerow([f'{k:g}', subset, f'{tau:g}', buf,
                                 *(metrics.get(m, '') for m in METRIC_KEYS)])
                f.flush()
                print(f'[OK ] k={k:g} / {subset} (τ={tau:g}, {src}): '
                      f'HOTA={metrics.get("HOTA", "?")}', flush=True)

    # итог: средний HOTA по поднаборам для каждого k
    from scripts.tuned_params import best_lqa_k
    best = best_lqa_k()
    print(f'\n[OK ] {RESULTS_CSV}')
    print(f'[OK ] лучший k по среднему HOTA: {best}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
