"""Гейт Б на детекциях YOLO26 и с корректной оценкой устойчивости.

Два возражения к первому прогону гейта Б, которые здесь закрываются.

Возражение 1: измеряли не то. Первый прогон брал центры ЭТАЛОННЫХ рамок из
box_track_20 — то есть предсказание по идеально точным наблюдениям. Заявлялось
же удаление шума детектора. Шума детектора в том опыте не было вовсе. Здесь на
вход подаются центры реальных детекций YOLO26 (кэш runs/det_cache_ctl, модель
weights/baseline_seed42), сопоставленных с эталонными треками по IoU; цель
остаётся эталонной. Это и есть постановка «очищаем ошибки детектора».

Возражение 2: мера неопределённости выбрана неверно. Первый прогон сравнивал
прирост с РАЗБРОСОМ МЕЖДУ БАЗИСАМИ и делал вывод «неотличимо от случайного
выбора базиса». Разброс между базисами — это различие методов, а не
погрешность оценки: добавь заведомо плохой базис, и «неопределённость»
вырастет, хотя сравнение лучшего варианта с Калманом не изменится. Здесь
устойчивость считается парным бутстрапом с кластеризацией ПО ВИДЕО: окна одной
траектории и траектории одного видео зависимы, независимая единица —
видеозапись. Даётся 95%-й доверительный интервал относительного улучшения.

Оба источника наблюдений (эталон и детекции) считаются в одном прогоне, чтобы
разницу между ними нельзя было списать на другие условия опыта.

Запуск: .venv/Scripts/python.exe scripts/wavelet_gate_b_dets.py
        ... --videos 50 --conf 0.25 --boot 2000
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.wavelet_gate_b import (BASES, TRACKS, extrapolate,  # noqa: E402
                                    kalman_cv, wavelet_clean)
from tracking.detection_cache import load_video_cache  # noqa: E402

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'wavelet_gate_b_dets.csv'
OUT_BOOT = PROJECT_ROOT / 'experiments' / 'results' / 'wavelet_gate_b_bootstrap.csv'


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU каждой рамки a (N,4) с каждой рамкой b (M,4), формат xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-9)


def gt_boxes(path: Path) -> dict[int, tuple[list[str], np.ndarray]]:
    """кадр -> (список id, рамки (N,4))."""
    data = json.loads(path.read_text(encoding='utf-8'))
    per_frame = {}
    for fr in data:
        ids, boxes = [], []
        for l in fr.get('labels') or []:
            b = l.get('box2d')
            if not b:
                continue
            ids.append(l['id'])
            boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
        per_frame[fr['frameIndex']] = (ids, np.array(boxes, np.float64).reshape(-1, 4))
    return per_frame


def build_pairs(vid: str, conf_thr: float, iou_thr: float):
    """Наблюдения двух видов для каждого эталонного трека одного видео.

    Возвращает список кусков подряд идущих кадров:
        (эталонные центры (N,2), центры детекций (N,2))
    Кусок обрывается на кадре, где эталонному объекту не нашлось детекции с
    IoU >= iou_thr и уверенностью >= conf_thr: именно там трекер и теряет
    объект, а предсказание работает по последним успешным наблюдениям.
    """
    gt = gt_boxes(TRACKS / f'{vid}.json')
    cache = load_video_cache(CACHE, 'val', vid)
    fidx_all, xyxy_all, conf_all = cache['frame_idx'], cache['xyxy'], cache['conf']

    obs: dict[str, list] = defaultdict(list)   # id -> (кадр, эталон xy, детекция xy)
    for f in sorted(gt):
        ids, boxes = gt[f]
        if len(boxes) == 0:
            continue
        m = (fidx_all == f) & (conf_all >= conf_thr)
        det = xyxy_all[m].astype(np.float64)
        if len(det) == 0:
            continue
        M = iou_matrix(boxes, det)
        best = M.argmax(axis=1)
        for i, oid in enumerate(ids):
            if M[i, best[i]] < iou_thr:
                continue
            d = det[best[i]]
            obs[oid].append((f,
                             ((boxes[i, 0] + boxes[i, 2]) / 2,
                              (boxes[i, 1] + boxes[i, 3]) / 2),
                             ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)))
    out = []
    for pts in obs.values():
        pts.sort()
        cur = [pts[0]]
        for p in pts[1:]:
            if p[0] == cur[-1][0] + 1:
                cur.append(p)
            else:
                out.append(cur)
                cur = [p]
        out.append(cur)
    return [(np.array([q[1] for q in c]), np.array([q[2] for q in c]))
            for c in out]


def predict_all(win: np.ndarray, gap: int, fit: int) -> dict[str, np.ndarray]:
    methods = {}
    p_mnk, p_kal = np.zeros(2), np.zeros(2)
    pw = {b: np.zeros(2) for b in BASES}
    pk = {b: np.zeros(2) for b in BASES}
    for ax in (0, 1):
        s = win[:, ax]
        p_mnk[ax] = extrapolate(s, fit, gap)
        p_kal[ax] = kalman_cv(s, gap)
        for b in BASES:
            c = wavelet_clean(s, b)
            pw[b][ax] = extrapolate(c, fit, gap)
            pk[b][ax] = kalman_cv(c, gap)
    methods['МНК (база)'] = p_mnk
    methods['Калман CV'] = p_kal
    for b in BASES:
        methods[f'вейвлет {b}'] = pw[b]
        methods[f'Калман+{b}'] = pk[b]
    return methods


def boot_ci(per_video: dict[str, tuple[list, list]], n_boot: int, seed: int = 0):
    """Парный кластерный бутстрап: относительное улучшение медианы ошибки.

    per_video: видео -> (ошибки метода A, ошибки метода B) равной длины,
    случаи идут в одном порядке (парность). Единица ресемплинга — видео.
    """
    vids = list(per_video)
    rng = np.random.default_rng(seed)
    a_all = np.concatenate([per_video[v][0] for v in vids])
    b_all = np.concatenate([per_video[v][1] for v in vids])
    point = 100 * (np.median(a_all) - np.median(b_all)) / np.median(a_all)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(vids), len(vids), replace=True)
        a = np.concatenate([per_video[vids[j]][0] for j in pick])
        b = np.concatenate([per_video[vids[j]][1] for j in pick])
        stats[i] = 100 * (np.median(a) - np.median(b)) / np.median(a)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--fit', type=int, default=8)
    ap.add_argument('--gaps', nargs='+', type=int, default=[2, 3, 5])
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='порог уверенности детекции (рабочая точка трекера)')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    W, F = args.window, min(args.fit, args.window)
    vids = [p.stem for p in sorted(TRACKS.glob('*.json'))
            if (CACHE / 'val' / f'{p.stem}.npz').exists()][:args.videos]
    if not vids:
        raise SystemExit(f'нет кэша детекций: {CACHE}')
    print(f'[INFO] видео с кэшем детекций: {len(vids)}')

    need = W + max(args.gaps) + 1
    # ошибки: (источник, разрыв, метод) -> видео -> список
    err: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_runs = 0
    resid = []          # |детекция - эталон|, шум детектора в пикселях
    for vid in vids:
        for gt_run, det_run in build_pairs(vid, args.conf, args.iou):
            if len(gt_run) < need:
                continue
            n_runs += 1
            resid.append(np.hypot(*(det_run - gt_run).T))
            n = len(gt_run)
            for gap in args.gaps:
                for end in range(W, n - gap):
                    target = gt_run[end + gap - 1]
                    for src, run in (('эталон', gt_run), ('детекции', det_run)):
                        preds = predict_all(run[end - W:end], gap, F)
                        for m, p in preds.items():
                            err[(src, gap, m)][vid].append(
                                float(np.hypot(*(p - target))))

    if not n_runs:
        raise SystemExit('нет пригодных кусков траекторий')
    resid = np.concatenate(resid)
    print(f'[INFO] кусков траекторий {n_runs}, '
          f'шум детектора: медиана {np.median(resid):.2f} px, '
          f'сред. {resid.mean():.2f} px')

    methods = list(next(iter(err.values())) and
                   [m for (s, g, m) in err if s == 'эталон' and g == args.gaps[0]])
    flat = {k: np.concatenate([np.array(v) for v in d.values()])
            for k, d in err.items()}
    med = {k: float(np.median(v)) for k, v in flat.items()}
    ncase = len(flat[('эталон', args.gaps[0], 'Калман CV')])
    print(f'[INFO] случаев на клетку {ncase}\n')

    rows = []
    for src in ('эталон', 'детекции'):
        print(f'=== источник наблюдений: {src} ===')
        print('%-16s' % 'метод' + ''.join('%12s' % f'разрыв {g}' for g in args.gaps))
        for m in methods:
            line = '%-16s' % m
            for g in args.gaps:
                v = med[(src, g, m)]
                line += '%12.3f' % v
                rows.append({'источник': src, 'разрыв': g, 'метод': m,
                             'медиана_ошибки': round(v, 4),
                             'случаев': len(flat[(src, g, m)])})
            print(line)
        print('')

    # --- парный бутстрап по видео ---
    print('=== устойчивость: парный бутстрап с кластеризацией по видео ===')
    print('(единица ресемплинга — видеозапись, %d повторов, 95%% интервал)\n'
          % args.boot)
    # пары читаются как «насколько B лучше A», знак положителен, когда B лучше
    comps = [('Калман CV', 'Калман+db3', 'очищение db3 лучше штатного Калмана'),
             ('Калман CV', 'Калман+db2', 'очищение db2 лучше штатного Калмана'),
             ('Калман+haar', 'Калман+db2', 'db2 лучше Хаара (ось новизны)'),
             ('Калман+haar', 'Калман+db3', 'db3 лучше Хаара (ось новизны)'),
             ('МНК (база)', 'Калман CV', 'Калман лучше МНК (контроль шкалы)')]
    brows = []
    for src in ('эталон', 'детекции'):
        print(f'-- {src} --')
        for a, b, label in comps:
            for g in args.gaps:
                pv = {v: (np.array(err[(src, g, a)][v]),
                          np.array(err[(src, g, b)][v]))
                      for v in err[(src, g, a)]}
                point, lo, hi = boot_ci(pv, args.boot)
                sign = 'значимо' if lo > 0 or hi < 0 else 'НЕ значимо'
                print('   %-38s разрыв %d: %+.2f%%  [%+.2f, %+.2f]  %s'
                      % (label, g, point, lo, hi, sign))
                brows.append({'источник': src, 'разрыв': g, 'сравнение': label,
                              'метод_A': a, 'метод_B': b,
                              'B_лучше_A_%': round(point, 3),
                              'ци_низ': round(lo, 3), 'ци_верх': round(hi, 3),
                              'значимо': lo > 0 or hi < 0, 'видео': len(pv)})
        print('')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_BOOT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0]))
        w.writeheader()
        w.writerows(brows)
    print(f'[OK ] {OUT}\n[OK ] {OUT_BOOT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
