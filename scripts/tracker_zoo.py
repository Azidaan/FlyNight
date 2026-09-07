"""Гейт 0в: сколько запаса забирают уже готовые трекеры на наших детекциях.

Зачем. Прежде чем разрабатывать свой метод ассоциации, надо знать, что даёт
готовое. Если современный трекер снимает существенную часть измеренного запаса,
то мерить свою разработку надо от него, а не от ByteTrack, — иначе прирост
окажется приростом над устаревшей базой, и это первое, что спросят на защите.

Условия сравнения. Детектор один и тот же для всех: замороженный кэш детекций
YOLO26 (runs/det_cache_ctl), те же 50 видео val. Меняется ТОЛЬКО ассоциация.
Ни один трекер не использует ReID (with_reid=False во всех конфигурациях по
умолчанию) — значит внешность не участвует нигде, и сравнение честное по входу.

Компенсация движения камеры (GMC) у BoT-SORT и TrackTrack требует самого кадра,
поэтому для них кадры читаются с диска. Для дорожной съёмки с движущейся камеры
это существенная часть метода, и отключать её было бы нечестно по отношению к
BoT-SORT. Остальным трекерам изображение не нужно.

Метрики считаются двумя конвейерами сразу: штатным (IDF1/MOTA/IDSw через
bdd100k) и TrackEval (HOTA/AssA/DetA). Расхождение между ними по общим метрикам
— проверка того, что оба настроены на одни данные.

Запуск: .venv/Scripts/python.exe scripts/tracker_zoo.py
        ... --trackers bytetrack ocsort botsort --videos 50
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from evaluation.hota_eval import evaluate_hota  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.offline_association import _DetSlice  # noqa: E402

import yaml  # noqa: E402
from ultralytics.trackers import track as ul_track  # noqa: E402

IMAGES = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
CFG_DIR = Path(ul_track.__file__).resolve().parents[1] / 'cfg' / 'trackers'
RUNS = PROJECT_ROOT / 'runs' / 'tracker_zoo'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'tracker_zoo.csv'
ALL = ('bytetrack', 'ocsort', 'botsort', 'tracktrack', 'fasttrack', 'deepocsort')
KEYS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSw', 'MOTP')


def load_args(name: str, overrides: dict) -> SimpleNamespace:
    """Конфигурация трекера из штатного YAML ultralytics плюс наши правки."""
    cfg = yaml.safe_load((CFG_DIR / f'{name}.yaml').read_text(encoding='utf-8'))
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def needs_image(args_ns: SimpleNamespace) -> bool:
    return str(getattr(args_ns, 'gmc_method', 'none')).lower() not in ('none', '')


def run_tracker(name: str, vids: list[str], overrides: dict,
                out_dir: Path) -> float:
    """Прогон одного трекера по всем видео. Возвращает время в секундах."""
    import cv2
    from ultralytics.trackers.basetrack import BaseTrack

    tracker_cls = ul_track.TRACKER_MAP[name]
    args_ns = load_args(name, overrides)
    with_img = needs_image(args_ns)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for vid in vids:
        cache = load_video_cache(CACHE, 'val', vid)
        BaseTrack.reset_id()
        tracker = tracker_cls(args_ns)
        frames = []
        for fidx, fname, xyxy, conf, cls in iter_frames(cache):
            dets = _DetSlice(xyxy, conf, cls)
            img = None
            if with_img:
                img = cv2.imread(str(IMAGES / vid / fname))
            out = tracker.update(dets, img) if with_img else tracker.update(dets)
            labels = []
            for row in np.atleast_2d(out) if len(out) else []:
                x1, y1, x2, y2, tid, score, c = row[:7]
                labels.append({'id': str(int(tid)),
                               'category': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                               'score': float(score),
                               'box2d': {'x1': float(x1), 'y1': float(y1),
                                         'x2': float(x2), 'y2': float(y2)}})
            frames.append({'videoName': vid, 'name': fname,
                           'frameIndex': fidx, 'labels': labels})
        (out_dir / f'{vid}.json').write_text(json.dumps(frames), encoding='utf-8')
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--trackers', nargs='+', default=list(ALL))
    ap.add_argument('--tau', type=float, default=None,
                    help='общий track_high_thresh; по умолчанию — из YAML трекера')
    ap.add_argument('--buffer', type=int, default=None)
    args = ap.parse_args()

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    over = {}
    if args.tau is not None:
        over['track_high_thresh'] = args.tau
        over['new_track_thresh'] = args.tau
    if args.buffer is not None:
        over['track_buffer'] = args.buffer
    print(f'[INFO] видео {len(vids)}, трекеры {args.trackers}, '
          f'правки конфигурации: {over or "нет, штатные YAML"}')

    rows, res = [], {}
    for name in args.trackers:
        if name not in ul_track.TRACKER_MAP:
            print(f'[ПРОПУСК] {name}: нет в TRACKER_MAP')
            continue
        out_dir = RUNS / name
        try:
            sec = run_tracker(name, vids, over, out_dir)
        except Exception as e:                       # noqa: BLE001
            print(f'[ОШИБКА] {name}: {type(e).__name__}: {e}')
            continue
        met = evaluate_tracking_bdd(out_dir, vids, RUNS / f'{name}.json')
        hota = evaluate_hota(out_dir, vids, tracker_name=name)
        merged = {**{k: v for k, v in met.items() if isinstance(v, (int, float))},
                  **hota}
        res[name] = merged
        print('[%-10s] %5.0f с  ' % (name, sec) + '  '.join(
            f'{k}={merged[k]:.3f}' for k in KEYS if k in merged))
        for k, v in merged.items():
            rows.append({'трекер': name, 'метрика': k,
                         'значение': round(float(v), 4), 'секунд': round(sec, 1)})

    base = res.get('bytetrack')
    if base:
        print('\n=== прирост к ByteTrack (та же детекция) ===')
        for name, m in res.items():
            if name == 'bytetrack':
                continue
            print('  %-10s ' % name + '  '.join(
                '%s %+.2f (%+.1f%%)' % (k, m[k] - base[k],
                                        100 * (m[k] - base[k]) / base[k])
                for k in ('HOTA', 'AssA', 'IDF1', 'MOTA')
                if k in m and base.get(k)))

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
