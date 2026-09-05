"""Доходит ли прирост от разрешения 960 до трекинга — или тонет в нём.

Зачем. Разрешение 960 на применении даёт +0.037 ночной mAP50 бесплатно, без
переобучения (замер 03.09). Но через трекер это ни разу не прогонялось: план
трактовал 960 как настройку стенда, одинаковую для всех конфигураций, то есть
как то, что разницы не даёт. Для работы, которой нужен положительный результат,
это ошибка — прирост детекции есть, а в целевой метрике он не измерен.

Почему ожидание большое. Ночью пропуски составляют 85% всех потерь трекинга
(60 003 против 6 026 ложных и 4 692 перебросов), то есть отношение вклада
обнаружения к вкладу связывания примерно 14 к 1. Полнота при 960 растёт с
0.3317 до 0.3777 — на 13.9% относительно. Если пропуски упадут пропорционально,
это порядка 8 000 объектов из 127 687, то есть единицы пунктов MOTA — на
порядок больше всего, что удалось выжать из порогов (0.3 и 0.06).

Две постановки сравнения, обе нужны:

    ОДИН ПОРОГ — оба разрешения при умолчании 0.25. Консервативно и честно:
        ничего не подбирается, сравнение прямое.
    СВОЙ ЛУЧШИЙ — каждое разрешение при своём оптимальном пороге. На 960
        распределение баллов сдвигается, и оптимум может уехать; сравнивать
        при чужом оптимуме значило бы занижать. Но лучший порог здесь
        выбирается на тех же данных, на которых меряется, поэтому это ВЕРХНЯЯ
        оценка, а не результат для защиты. В текст идёт первая постановка.

Запуск: python scripts/resolution_tracking.py
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

CACHES = {640: PROJECT_ROOT / 'runs' / 'det_cache_ctl',
          960: PROJECT_ROOT / 'runs' / 'det_cache_ctl_960'}
RUNS = PROJECT_ROOT / 'runs' / 'res_track'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'resolution_tracking.csv'
DEFAULT_THR = 0.25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='val')
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument('--buffer', type=int, default=30)
    args = ap.parse_args()

    # Список видео — ПЕРЕСЕЧЕНИЕ двух кэшей: сравнивать можно только то, что
    # посчитано обоими, иначе разница окажется разницей выборок.
    sets = {r: {p.stem for p in (c / args.split).glob('*.npz')}
            for r, c in CACHES.items()}
    for r, s in sets.items():
        print(f'[INFO] кэш {r}: {len(s)} видео')
    common = sorted(sets[640] & sets[960])
    if not common:
        raise SystemExit('нет общих видео — постройте кэш 960')
    print(f'[INFO] общих: {len(common)}')

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / f'bdd100k_labels_images_{args.split}.json')
    attrs = load_video_attributes(str(det_json))
    by_cond = {c: [v for v in filter_videos_by_subset(attrs, common, c) if v in common]
               for c in ('normal', 'night')}

    rows = []
    n, total = 0, len(CACHES) * len(by_cond) * len(args.thresholds)
    for res, cache in CACHES.items():
        for cond, vids in by_cond.items():
            for thr in args.thresholds:
                n += 1
                tag = f'{cond}_r{res}_t{thr:g}'
                out_dir = RUNS / tag
                run_association(cache, vids, out_dir, args.split,
                                args=make_tracker_args(track_high_thresh=thr,
                                                       track_buffer=args.buffer))
                m = evaluate_tracking_bdd(out_dir, vids, out_dir / '_eval.json',
                                          args.split)

                def pick(*keys):
                    for k in keys:
                        if k in m and not isinstance(m[k], list):
                            return float(m[k])
                    return float('nan')

                rows.append({'условие': cond, 'разрешение': res, 'порог': thr,
                             'видео': len(vids), 'MOTA': pick('MOTA', 'mMOTA'),
                             'IDF1': pick('IDF1', 'mIDF1'),
                             'MOTP': pick('MOTP', 'mMOTP')})
                print(f'[INFO] {n}/{total} {tag}: MOTA={rows[-1]["MOTA"]:.3f} '
                      f'IDF1={rows[-1]["IDF1"]:.3f}', flush=True)

    def get(cond, res, thr):
        return next((r for r in rows if r['условие'] == cond
                     and r['разрешение'] == res and r['порог'] == thr), None)

    print('\n=== кривые по порогу ===')
    for cond in ('normal', 'night'):
        title = 'день' if cond == 'normal' else 'ночь'
        print(f'\n{title}:')
        print('%7s %11s %11s %11s %11s' % (
            'порог', 'MOTA 640', 'MOTA 960', 'IDF1 640', 'IDF1 960'))
        for thr in args.thresholds:
            a, b = get(cond, 640, thr), get(cond, 960, thr)
            if not (a and b):
                continue
            print('%7.2f %11.3f %11.3f %11.3f %11.3f' % (
                thr, a['MOTA'], b['MOTA'], a['IDF1'], b['IDF1']))

    print('\n=== приговор ===')
    for cond in ('normal', 'night'):
        title = 'день' if cond == 'normal' else 'ночь'
        a, b = get(cond, 640, DEFAULT_THR), get(cond, 960, DEFAULT_THR)
        if not (a and b):
            continue
        print(f'\n  {title}:')
        print(f'    при одном пороге {DEFAULT_THR:g}: '
              f'{a["MOTA"]:.3f} -> {b["MOTA"]:.3f} MOTA ({b["MOTA"]-a["MOTA"]:+.3f}), '
              f'{a["IDF1"]:.3f} -> {b["IDF1"]:.3f} IDF1 ({b["IDF1"]-a["IDF1"]:+.3f})')
        ba = max((r for r in rows if r['условие'] == cond and r['разрешение'] == 640),
                 key=lambda r: r['MOTA'])
        bb = max((r for r in rows if r['условие'] == cond and r['разрешение'] == 960),
                 key=lambda r: r['MOTA'])
        print(f'    каждое при своём лучшем: {ba["MOTA"]:.3f} (порог {ba["порог"]:g}) '
              f'-> {bb["MOTA"]:.3f} (порог {bb["порог"]:g}) '
              f'= {bb["MOTA"]-ba["MOTA"]:+.3f} MOTA')

    print('\n  Для сравнения масштаба: всё, что удалось выжать из порогов —')
    print('  0.275 MOTA (оптимум ночью) и 0.063 (расцепление проходов).')
    print('  Числа выше надо читать на этом фоне.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
