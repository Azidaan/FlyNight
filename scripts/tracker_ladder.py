"""Четвёртая ступень лестницы: что теряет трекер из того, что детектор нашёл.

Зачем. Замер miss_ladder.py показал: ночью детектор при пороге 0.25 покрывает
64.9% эталонных объектов, а трекинговая оценка отчитывалась о 53% найденных.
Около 12 процентных пунктов теряется ПОСЛЕ детектора. Куда именно — не
разделено, а лечение у разных причин разное.

Здесь эти пункты раскладываются. Для каждого эталонного объекта берутся две
независимые характеристики:

    состояние у детектора — в какой полосе уверенности лежит верная рамка:
        ВЫСОКАЯ      >= track_high_thresh, трекер создаёт по ней трек
        СЛАБАЯ       [track_low_thresh, track_high_thresh) — вторая волна
                     ByteTrack: годится продолжать трек, но не создавать
        НИЖЕ ПОЛОСЫ  [0.01, track_low_thresh) — трекер её не видит вовсе
        НЕТ          верной рамки нет

    состояние у трекера — покрыт ли объект рамкой итогового трека верного
        класса с перекрытием не ниже 0.5.

Пересечение двух характеристик и есть разложение. Оно обязано закончиться
РЕШЕНИЕМ, а не числом, и решение тут такое: сколько из слабой полосы вторая
волна ByteTrack уже спасает. Если почти всё — работать над нижним порогом
незачем, мишень занята. Если почти ничего — 22.7% объектов лежат нетронутыми,
и это цель по теме работы.

Ассоциация гоняется оффлайн по кэшу детекций (tracking/offline_association.py):
тот же BYTETracker, что и в model.track(), но без прогона детектора —
секунды на процессоре.

Запуск: python scripts/tracker_ladder.py
        python scripts/tracker_ladder.py --track-low 0.05
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.miss_ladder import iou_mat, load_gt, size_class  # noqa: E402

CACHE_DIR = PROJECT_ROOT / 'runs' / 'det_cache_ctl'
TRACKS = PROJECT_ROOT / 'runs' / 'tracker_ladder'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'tracker_ladder.csv'

BANDS = ['ВЫСОКАЯ', 'СЛАБАЯ', 'НИЖЕ ПОЛОСЫ', 'НЕТ']
IOU_TP = 0.5


def det_band(gt_box, gt_cls, d_box, d_cls, d_conf, hi: float, lo: float):
    """Полоса уверенности лучшей ВЕРНОЙ детекции для каждой эталонной рамки."""
    n = len(gt_box)
    band = np.full(n, 3, np.int32)                       # НЕТ
    if n == 0 or len(d_box) == 0:
        return band
    m = iou_mat(gt_box, d_box)
    same = gt_cls[:, None] == d_cls[None, :]
    ok = (m >= IOU_TP) & same
    best = np.where(ok, d_conf[None, :], -1.0).max(1)
    band[best >= 0.0] = 2                                # НИЖЕ ПОЛОСЫ
    band[best >= lo] = 1                                 # СЛАБАЯ
    band[best >= hi] = 0                                 # ВЫСОКАЯ
    return band


def track_cover(gt_box, gt_cls, t_box, t_cls):
    """Покрыта ли эталонная рамка рамкой итогового трека верного класса."""
    n = len(gt_box)
    if n == 0 or len(t_box) == 0:
        return np.zeros(n, bool)
    m = iou_mat(gt_box, t_box)
    same = gt_cls[:, None] == t_cls[None, :]
    return (np.where(same, m, 0.0).max(1) >= IOU_TP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--track-high', type=float, default=0.25)
    ap.add_argument('--track-low', type=float, default=0.1)
    ap.add_argument('--track-buffer', type=int, default=30)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--keep', action='store_true',
                    help='не перезапускать ассоциацию, взять готовые треки')
    args = ap.parse_args()

    from datasets.bdd100k_loader import CLASS_NAMES
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from tracking.offline_association import make_tracker_args, run_association

    files = sorted((CACHE_DIR / 'val').glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    ids = [f.stem for f in files]

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    cond_of: dict[str, str] = {}
    for cond in ('normal', 'night'):
        for v in filter_videos_by_subset(attrs, ids, cond):
            cond_of.setdefault(v, cond)
    ids = [v for v in ids if v in cond_of]

    if TRACKS.exists() and not args.keep:
        shutil.rmtree(TRACKS)
    print(f'[INFO] ассоциация {len(ids)} видео, порог высокий '
          f'{args.track_high}, низкий {args.track_low}', flush=True)
    run_association(CACHE_DIR, ids, TRACKS, 'val',
                    args=make_tracker_args(track_high_thresh=args.track_high,
                                           track_low_thresh=args.track_low,
                                           track_buffer=args.track_buffer))

    name_to_cls = {v: k for k, v in CLASS_NAMES.items()}
    tally: Counter = Counter()

    for vi, f in enumerate(files, 1):
        vid = f.stem
        cond = cond_of.get(vid)
        if cond is None:
            continue
        d = np.load(f)
        names, fidx = d['frame_names'], d['frame_idx']
        xyxy, conf, cls = d['xyxy'], d['conf'], d['cls'].astype(np.int32)

        tr = json.loads((TRACKS / f'{vid}.json').read_text(encoding='utf-8'))
        by_frame = {fr['frameIndex']: fr['labels'] for fr in tr}

        for fi in np.unique(fidx):
            gt_box, gt_cls = load_gt(vid, str(names[fi]))
            if len(gt_box) == 0:
                continue
            sel = fidx == fi
            band = det_band(gt_box, gt_cls, xyxy[sel], cls[sel], conf[sel],
                            args.track_high, args.track_low)

            labs = by_frame.get(int(fi), [])
            if labs:
                t_box = np.array([[l['box2d']['x1'], l['box2d']['y1'],
                                   l['box2d']['x2'], l['box2d']['y2']]
                                  for l in labs], np.float32)
                t_cls = np.array([name_to_cls.get(l['category'], -1)
                                  for l in labs], np.int32)
            else:
                t_box = np.zeros((0, 4), np.float32)
                t_cls = np.zeros(0, np.int32)
            cov = track_cover(gt_box, gt_cls, t_box, t_cls)
            sz = size_class(gt_box)

            for b, c, z in zip(band, cov, sz):
                tally[(cond, BANDS[b], bool(c))] += 1
                tally[(cond + '/' + str(z), BANDS[b], bool(c))] += 1
        if vi % 10 == 0:
            print(f'[INFO] {vi}/{len(files)} видео', flush=True)

    rows = []
    for cond in ('normal', 'night'):
        tot = sum(v for (c, _, _), v in tally.items() if c == cond)
        if not tot:
            continue
        title = 'день' if cond == 'normal' else 'ночь'
        print(f'\n=== {title}: эталонных объектов {tot} ===')
        print('%-14s %9s %10s %12s %10s' % (
            'полоса у детектора', 'объектов', 'доля', 'в треке', 'спасено'))
        for b in BANDS:
            yes = tally[(cond, b, True)]
            no = tally[(cond, b, False)]
            n = yes + no
            if not n:
                continue
            print('%-14s %9d %9.1f%% %11.1f%% %9d' % (
                b, n, 100 * n / tot, 100 * yes / n, yes))
            rows.append((cond, b, n, yes, no, round(100 * yes / n, 2)))

        in_track = sum(tally[(cond, b, True)] for b in BANDS)
        hi_n = sum(tally[(cond, BANDS[0], s)] for s in (True, False))
        hi_lost = tally[(cond, BANDS[0], False)]
        lo_n = sum(tally[(cond, BANDS[1], s)] for s in (True, False))
        lo_saved = tally[(cond, BANDS[1], True)]
        below = sum(tally[(cond, BANDS[2], s)] for s in (True, False))
        print(f'\n  покрыто треками:            {100 * in_track / tot:5.1f}%')
        print(f'  выдано детектором (высокая): {100 * hi_n / tot:5.1f}%')
        print(f'  из них трекер потерял:       {100 * hi_lost / max(hi_n, 1):5.1f}%'
              f'  ({hi_lost} объектов)')
        print(f'  слабая полоса:               {100 * lo_n / tot:5.1f}% всех объектов')
        print(f'  из неё вторая волна спасла:  {100 * lo_saved / max(lo_n, 1):5.1f}%'
              f'  ({lo_saved} объектов)')
        print(f'  ниже полосы (трекер не видит): {100 * below / tot:5.1f}%')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['условие', 'полоса', 'объектов', 'в_треке', 'не_в_треке', 'доля_%'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
