"""Сеточный поиск параметров ByteTrack на кэше детекций (Этап 8, подэтап 8.2).

Конвейер:
1. Детектор прогоняется по всем видео поднаборов ровно один раз
   (tracking/detection_cache.py) — самый долгий шаг, только GPU-часть.
2. Сетка track_high_thresh x track_buffer гоняется оффлайн-ассоциацией
   на кэше (CPU, секунды на конфигурацию).
3. Каждая конфигурация оценивается официальным bdd100k.eval
   (evaluation/tracking_metrics.evaluate_tracking_bdd).

Выход: experiments/results/tracking_param_search.csv со столбцами
    subset, track_high_thresh, track_buffer, HOTA, IDF1, MOTA, IDSw, ...

Использование:
    python scripts/run_tracker_grid_search.py --model weights/baseline_seed42/best.pt
    python scripts/run_tracker_grid_search.py --model ... --subsets normal night
    python scripts/run_tracker_grid_search.py --model ... --thresholds 0.3 0.5 --buffers 30
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from tracking.detection_cache import CACHE_ROOT, build_detection_cache  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'tracking_param_search.csv'
GRID_RUNS = PROJECT_ROOT / 'runs' / 'track_grid'
SUBSETS = ('normal', 'night', 'rain', 'fog')
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw', 'MOTP', 'DetA', 'AssA')


def subset_videos(split: str, subsets: list[str]) -> dict[str, list[str]]:
    """video_id поднаборов из MOT split (джойн с Det-атрибутами)."""
    det_json = (DATA_ROOT / 'labels' / f'bdd100k_labels_images_{split}.json')
    attrs = load_video_attributes(str(det_json))
    mot_ids = sorted(p.name for p in
                     (DATA_ROOT / 'images' / 'track' / split).iterdir()
                     if p.is_dir())
    return {s: filter_videos_by_subset(attrs, mot_ids, s) for s in subsets}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='чекпойнт детектора (best.pt)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.3, 0.4, 0.5, 0.6])
    ap.add_argument('--buffers', nargs='+', type=int, default=[15, 30, 60])
    ap.add_argument('--cache-dir', default=str(CACHE_ROOT))
    args = ap.parse_args()

    by_subset = subset_videos(args.split, args.subsets)
    for s, vids in by_subset.items():
        print(f'[INFO] поднабор {s}: {len(vids)} видео')

    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    print(f'[INFO] всего уникальных видео: {len(all_ids)}')
    build_detection_cache(args.model, all_ids, args.split, args.cache_dir)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, float, int]] = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline='', encoding='utf-8') as f:
            done = {(r['subset'], float(r['track_high_thresh']),
                     int(r['track_buffer'])) for r in csv.DictReader(f)}
        print(f'[INFO] в CSV уже {len(done)} конфигураций — пропускаются')

    write_header = not RESULTS_CSV.exists()
    n_total = len(args.subsets) * len(args.thresholds) * len(args.buffers)
    n_done = 0
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['subset', 'track_high_thresh', 'track_buffer',
                             *METRIC_KEYS])
        for subset in args.subsets:
            vids = by_subset[subset]
            for thresh in args.thresholds:
                for buf in args.buffers:
                    n_done += 1
                    if (subset, thresh, buf) in done:
                        continue
                    tag = f'{subset}_t{thresh:g}_b{buf}'
                    out_dir = GRID_RUNS / tag
                    run_association(
                        args.cache_dir, vids, out_dir, args.split,
                        args=make_tracker_args(track_high_thresh=thresh,
                                               track_buffer=buf))
                    metrics = evaluate_tracking_bdd(
                        out_dir, vids, out_dir / '_eval.json', args.split)
                    writer.writerow([subset, thresh, buf,
                                     *(metrics.get(k, '') for k in METRIC_KEYS)])
                    f.flush()
                    print(f'[INFO] {n_done}/{n_total} {tag}: '
                          f'HOTA={metrics.get("HOTA", "?")} '
                          f'IDF1={metrics.get("IDF1", "?")}', flush=True)
    print(f'[INFO] готово: {RESULTS_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
