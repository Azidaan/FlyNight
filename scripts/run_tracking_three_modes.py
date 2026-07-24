"""Сравнение режимов трекера (Этап 9, п. 9.4) — главный эксперимент этапа.

Режимы (все на одном q-кэше детекций, варьируется только ассоциация):
  standard        — фиксированный τ_base (план: 0.5);
  brightness      — per-video τ = a·brightness + b (коэффициенты из Этапа 8);
  lqa             — per-detection τ_i = τ_base − k(1−q_i) (LQAGuidedByteTrack);
  control_global  — per-video τ = τ_base − k(1−q̄_v), q̄_v — среднее q по видео.
                    Контроль конфаундера: подтверждением 2-го пункта новизны
                    считается ТОЛЬКО двойное условие lqa > brightness И
                    lqa > control_global (иначе работает среднее снижение
                    порога, а не индивидуальность).

Выход: experiments/results/tracking_three_modes.csv
    (mode, subset, metric, value)

Использование:
    python scripts/run_tracking_three_modes.py --model weights/drb_aug_full_seed42/best.pt
    ... --brightness-coef 0.55 0.25     # a, b из notes/brightness_to_threshold.md
    ... --skip-modes brightness         # пока коэффициентов нет
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.run_tracker_grid_search import subset_videos  # noqa: E402
from tracking.brightness_estimator import get_brightness_map  # noqa: E402
from tracking.brightness_regression import apply_brightness_adaptive_threshold  # noqa: E402
from tracking.detection_cache import load_video_cache  # noqa: E402
from tracking.inference_with_q import build_detection_cache_with_q  # noqa: E402
from tracking.lqa_guided_tracker import LQAGuidedByteTrack  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'tracking_three_modes.csv'
RUNS = PROJECT_ROOT / 'runs' / 'track_modes'
Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q'
SUBSETS = ('normal', 'night', 'rain', 'fog')
MODES = ('standard', 'brightness', 'lqa', 'control_global')
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw', 'MOTP', 'DetA', 'AssA')


def mean_q_per_video(cache_dir: Path, split: str, video_ids: list[str],
                     q_key: str) -> dict[str, float]:
    """q̄_v — среднее q по всем детекциям видео (для Control-Global)."""
    out = {}
    for vid in video_ids:
        cache = load_video_cache(cache_dir, split, vid)
        q = cache.get(q_key)
        out[vid] = float(q.mean()) if q is not None and len(q) else 1.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--tau-base', type=float, default=0.5)
    ap.add_argument('--k', type=float, default=0.2, help='коэффициент адаптации')
    ap.add_argument('--agg', default='mean', help='агрегация q_i (кэш-ключ q_<agg>)')
    ap.add_argument('--brightness-coef', type=float, nargs=2, default=None,
                    metavar=('A', 'B'), help='коэффициенты τ = a·brightness + b')
    ap.add_argument('--skip-modes', nargs='+', default=[], choices=list(MODES))
    args = ap.parse_args()

    by_subset = subset_videos(args.split, args.subsets)
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    print(f'[INFO] уникальных видео: {len(all_ids)}')
    build_detection_cache_with_q(args.model, all_ids, args.split, Q_CACHE)

    q_key = f'q_{args.agg}'
    modes = [m for m in MODES if m not in args.skip_modes]
    if 'brightness' in modes and args.brightness_coef is None:
        print('[WARN] нет --brightness-coef — режим brightness пропущен '
              '(коэффициенты даёт Этап 8, notes/brightness_to_threshold.md)')
        modes.remove('brightness')

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline='', encoding='utf-8') as f:
            done = {(r['mode'], r['subset']) for r in csv.DictReader(f)}

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['mode', 'subset', 'metric', 'value'])
        for mode in modes:
            for subset in args.subsets:
                if (mode, subset) in done:
                    continue
                vids = by_subset[subset]
                out_dir = RUNS / f'{mode}_{subset}'
                kw: dict = {}
                if mode == 'standard':
                    kw['args'] = make_tracker_args(track_high_thresh=args.tau_base)
                elif mode == 'brightness':
                    a, b = args.brightness_coef
                    bright = get_brightness_map(vids, args.split)
                    kw['per_video_args'] = {
                        v: make_tracker_args(track_high_thresh=(
                            apply_brightness_adaptive_threshold(bright[v], a, b)))
                        for v in vids}
                elif mode == 'lqa':
                    t_args = make_tracker_args(track_high_thresh=args.tau_base)
                    t_args.lqa_k = args.k
                    kw['args'] = t_args
                    kw['tracker_cls'] = LQAGuidedByteTrack
                    kw['q_key'] = q_key
                elif mode == 'control_global':
                    qbar = mean_q_per_video(Q_CACHE, args.split, vids, q_key)
                    kw['per_video_args'] = {
                        v: make_tracker_args(track_high_thresh=(
                            args.tau_base - args.k * (1.0 - qbar[v])))
                        for v in vids}

                run_association(Q_CACHE, vids, out_dir, args.split, **kw)
                metrics = evaluate_tracking_bdd(
                    out_dir, vids, out_dir / '_eval.json', args.split)
                for mk in METRIC_KEYS:
                    if mk in metrics:
                        writer.writerow([mode, subset, mk, metrics[mk]])
                f.flush()
                print(f'[OK ] {mode}/{subset}: '
                      f'HOTA={metrics.get("HOTA", "?")} '
                      f'IDF1={metrics.get("IDF1", "?")}', flush=True)
    print(f'[OK ] {RESULTS_CSV}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
