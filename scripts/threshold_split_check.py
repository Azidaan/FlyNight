"""Устойчив ли разный оптимум порога днём и ночью — проверка разбиением.

Зачем. Сетка порогов 05.09 дала оптимум 0.25 днём и 0.15 ночью на 25 видео
каждого условия. Но 25 видео — малая выборка, и оптимум может оказаться
свойством именно этих видео, а не условий. Разница между соседними точками
кривой около 0.3 MOTA, то есть мала; такое обязано проверяться, иначе на
защите это первый же вопрос.

Метод. Видео каждого условия делятся НА ДВЕ НЕЗАВИСИМЫЕ ПОЛОВИНЫ и вся сетка
считается на каждой отдельно. Если оптимум ночью держится ниже дневного на
обеих половинах — это свойство условий. Если половины расходятся — выборка.

Деление через одно, а не пополам по списку: идентификаторы видео BDD100K
отсортированы, и соседние часто снимались одной поездкой. Разбиение первые
против последних сложило бы в одну половину похожие сцены.

Считается целиком на процессоре по готовому кэшу детекций: ассоциация
секунды на конфигурацию, дальше официальная оценка BDD100K.

Запуск: python scripts/threshold_split_check.py
        python scripts/threshold_split_check.py --thresholds 0.1 0.2 0.3
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
RUNS = PROJECT_ROOT / 'runs' / 'thr_split'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'threshold_split_check.csv'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='val')
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument('--buffer', type=int, default=30)
    args = ap.parse_args()

    have = sorted(p.stem for p in (CACHE / args.split).glob('*.npz'))
    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / f'bdd100k_labels_images_{args.split}.json')
    attrs = load_video_attributes(str(det_json))

    halves: dict[tuple[str, int], list[str]] = {}
    for cond in ('normal', 'night'):
        vids = [v for v in filter_videos_by_subset(attrs, have, cond) if v in have]
        halves[(cond, 0)] = vids[0::2]
        halves[(cond, 1)] = vids[1::2]
        print(f'[INFO] {cond}: {len(vids)} видео -> '
              f'{len(vids[0::2])} и {len(vids[1::2])}')

    rows = []
    n, total = 0, len(halves) * len(args.thresholds)
    for (cond, h), vids in halves.items():
        if not vids:
            continue
        for thr in args.thresholds:
            n += 1
            tag = f'{cond}_h{h}_t{thr:g}'
            out_dir = RUNS / tag
            run_association(CACHE, vids, out_dir, args.split,
                            args=make_tracker_args(track_high_thresh=thr,
                                                   track_buffer=args.buffer))
            m = evaluate_tracking_bdd(out_dir, vids, out_dir / '_eval.json',
                                      args.split)
            rows.append({'условие': cond, 'половина': h, 'порог': thr,
                         'видео': len(vids),
                         'MOTA': float(m.get('mMOTA', float('nan'))),
                         'IDF1': float(m.get('mIDF1', float('nan')))})
            # mMOTA/mIDF1 — усреднение по классам; в общей сетке писались
            # сводные MOTA/IDF1, поэтому здесь берём обе, что вернул оценщик
            for k in ('MOTA', 'IDF1'):
                if k in m and not isinstance(m[k], list):
                    rows[-1][k] = float(m[k])
            print(f'[INFO] {n}/{total} {tag}: MOTA={rows[-1]["MOTA"]:.3f} '
                  f'IDF1={rows[-1]["IDF1"]:.3f}', flush=True)

    print('\n=== кривые по половинам ===')
    for cond in ('normal', 'night'):
        title = 'день' if cond == 'normal' else 'ночь'
        print(f'\n{title}:')
        print('%7s %12s %12s %12s %12s' % (
            'порог', 'MOTA пол.0', 'MOTA пол.1', 'IDF1 пол.0', 'IDF1 пол.1'))
        for thr in args.thresholds:
            g = {h: next((r for r in rows if r['условие'] == cond
                          and r['половина'] == h and r['порог'] == thr), None)
                 for h in (0, 1)}
            if not all(g.values()):
                continue
            print('%7.2f %12.3f %12.3f %12.3f %12.3f' % (
                thr, g[0]['MOTA'], g[1]['MOTA'], g[0]['IDF1'], g[1]['IDF1']))

    print('\n=== приговор ===')
    opt: dict[tuple[str, int], dict] = {}
    for cond in ('normal', 'night'):
        for h in (0, 1):
            g = [r for r in rows if r['условие'] == cond and r['половина'] == h]
            if not g:
                continue
            opt[(cond, h)] = {
                'MOTA': max(g, key=lambda r: r['MOTA'])['порог'],
                'IDF1': max(g, key=lambda r: r['IDF1'])['порог'],
            }
            print(f'  {"день" if cond == "normal" else "ночь"} половина {h}: '
                  f'оптимум по MOTA {opt[(cond, h)]["MOTA"]:g}, '
                  f'по IDF1 {opt[(cond, h)]["IDF1"]:g}')

    ok = []
    for metric in ('MOTA', 'IDF1'):
        for h in (0, 1):
            d = opt.get(('normal', h), {}).get(metric)
            nn = opt.get(('night', h), {}).get(metric)
            if d is None or nn is None:
                continue
            ok.append(nn < d)
    if ok and all(ok):
        print('\n  ОПТИМУМ НОЧЬЮ НИЖЕ ДНЕВНОГО НА ОБЕИХ ПОЛОВИНАХ И ПО ОБЕИМ')
        print('  МЕТРИКАМ. Это свойство условий, а не выборки: порог, зависящий')
        print('  от условий, получает подтверждённое основание.')
    elif ok and sum(ok) >= len(ok) / 2:
        print(f'\n  Знак держится в {sum(ok)} случаях из {len(ok)} — направление')
        print('  живо, но выборки мало. Нужны все видео поднаборов, не 25.')
    else:
        print('\n  ЗНАК НЕ ДЕРЖИТСЯ. Разный оптимум был свойством выборки.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
