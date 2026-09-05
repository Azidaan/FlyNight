"""Расцепление: сопоставлять слабыми детекциями, но не создавать по ним треки.

Механика, ради которой всё затевается (проверена по исходнику
ultralytics/trackers/byte_tracker.py, 05.09):

    первое сопоставление идёт против ОБЪЕДИНЁННОГО пула — отслеживаемые ПЛЮС
        потерянные треки (strack_pool = joint_stracks(tracked, lost));
    второе сопоставление — только против тех, у кого state == Tracked
        (r_tracked_stracks), потерянных там НЕТ;
    новые треки создаются ТОЛЬКО из сильного набора (scores >=
        track_high_thresh) и уже внутри него отсекаются по new_track_thresh.

Отсюда два следствия. Первое: слабая детекция НЕ МОЖЕТ воскресить потерянный
трек. Ночью объект тускнеет, уходит в слабый набор, его трек через
track_buffer кадров помечается потерянным — и слабые детекции уже никогда его
не вернут. Второе: опустить new_track_thresh НИЖЕ track_high_thresh
бессмысленно, связывающий порог всё равно главнее.

Значит осмысленно только расцепление ВВЕРХ: опустить track_high_thresh (чтобы
слабые детекции попали в сильный набор и получили право воскрешать потерянные
треки), но ПОДНЯТЬ new_track_thresh над ним, чтобы они при этом не плодили
новые треки. В связанной настройке, где оба порога равны, такой режим
недостижим в принципе, и он ни разу не проверялся.

Это ровно то, на что указали замеры 04-05.09: слабая полоса ночью полна верных
объектов (уверенность занижена на 0.098 в полосе 0.1..0.5), но ByteTrack не
достаёт их конструктивно — вторая волна спасает 0.5%.

Статус замера: РАЗВЕДОЧНЫЙ. Сетка гоняется на всех видео сразу, поэтому выбор
лучшей клетки и её оценка происходят на одних данных. Если клетка найдётся —
подтверждать отдельно на отложенных видео, иначе это прирост-мираж.

Запуск: python scripts/decouple_track_creation.py
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl'
RUNS = PROJECT_ROOT / 'runs' / 'decouple'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'decouple_track_creation.csv'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='val')
    ap.add_argument('--high', nargs='+', type=float, default=[0.1, 0.15, 0.2, 0.25])
    ap.add_argument('--new', nargs='+', type=float, default=[0.25, 0.3, 0.4])
    ap.add_argument('--buffer', type=int, default=30)
    args = ap.parse_args()

    have = sorted(p.stem for p in (CACHE / args.split).glob('*.npz'))
    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / f'bdd100k_labels_images_{args.split}.json')
    attrs = load_video_attributes(str(det_json))
    by_cond = {c: [v for v in filter_videos_by_subset(attrs, have, c) if v in have]
               for c in ('normal', 'night')}

    # Клетки: связанная настройка (new = high) как точка отсчёта, плюс все
    # расцепленные с new > high. Клетки с new < high пропускаются — они
    # неотличимы от связанной, потому что порог создания там не связывает.
    cells = []
    for h in args.high:
        cells.append((h, h))
        for n in args.new:
            if n > h:
                cells.append((h, n))
    cells = sorted(set(cells))

    rows = []
    n_done, total = 0, len(cells) * len(by_cond)
    for cond, vids in by_cond.items():
        for h, nw in cells:
            n_done += 1
            tag = f'{cond}_h{h:g}_n{nw:g}'
            out_dir = RUNS / tag
            run_association(CACHE, vids, out_dir, args.split,
                            args=make_tracker_args(track_high_thresh=h,
                                                   new_track_thresh=nw,
                                                   track_buffer=args.buffer))
            m = evaluate_tracking_bdd(out_dir, vids, out_dir / '_eval.json',
                                      args.split)

            def pick(*keys):
                for k in keys:
                    if k in m and not isinstance(m[k], list):
                        return float(m[k])
                return float('nan')

            rows.append({'условие': cond, 'high': h, 'new': nw,
                         'связано': h == nw,
                         'MOTA': pick('MOTA', 'mMOTA'),
                         'IDF1': pick('IDF1', 'mIDF1')})
            print(f'[INFO] {n_done}/{total} {tag}: MOTA={rows[-1]["MOTA"]:.3f} '
                  f'IDF1={rows[-1]["IDF1"]:.3f}', flush=True)

    print('\n=== таблица: строки — порог сопоставления, столбцы — порог создания ===')
    for cond in ('normal', 'night'):
        title = 'день' if cond == 'normal' else 'ночь'
        print(f'\n{title} (MOTA):')
        cols = sorted({r['new'] for r in rows if r['условие'] == cond})
        print('%8s' % 'high\\new' + ''.join('%9.2f' % c for c in cols))
        for h in args.high:
            line = '%8.2f' % h
            for c in cols:
                r = next((x for x in rows if x['условие'] == cond
                          and x['high'] == h and x['new'] == c), None)
                line += ('%9.3f' % r['MOTA']) if r else '%9s' % '·'
            print(line)

    print('\n=== приговор ===')
    for cond in ('normal', 'night'):
        g = [r for r in rows if r['условие'] == cond]
        if not g:
            continue
        tied = [r for r in g if r['связано']]
        free = [r for r in g if not r['связано']]
        bt = max(tied, key=lambda r: r['MOTA'])
        bf = max(free, key=lambda r: r['MOTA']) if free else None
        spread = max(r['MOTA'] for r in g) - min(r['MOTA'] for r in g)
        title = 'день' if cond == 'normal' else 'ночь'
        print(f'\n  {title}:')
        print(f'    лучшая СВЯЗАННАЯ:    high={bt["high"]:g} -> {bt["MOTA"]:.3f} MOTA')
        if bf is None:
            continue
        gain = bf['MOTA'] - bt['MOTA']
        print(f'    лучшая РАСЦЕПЛЕННАЯ: high={bf["high"]:g}, new={bf["new"]:g}'
              f' -> {bf["MOTA"]:.3f} MOTA')
        print(f'    прирост расцепления: {gain:+.3f}   (разброс по сетке {spread:.3f})')
        if gain > 0.2 * spread and gain > 0:
            print('    Расцепление даёт заметный прирост — проверять на отложенных.')
        elif gain > 0:
            print('    Прирост есть, но мал относительно разброса сетки.')
        else:
            print('    Расцепление НЕ помогает: связанная настройка не хуже.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
