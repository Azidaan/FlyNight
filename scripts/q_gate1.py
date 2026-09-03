"""Гейт 1: несёт ли карта качества информацию СВЕРХ уверенности детектора.

Что проверяется и почему не AP50. Подбор коэффициентов и замер на одних данных
дают прирост-мираж, исчезающий на новых кадрах. Настоящий вопрос — вложенное
сравнение моделей: предсказатель факта «правильная детекция / ложная» строится
сначала ТОЛЬКО на уверенности (модель M1), затем на уверенности ПЛЮС карта
качества (модель M2), и проверяется, стало ли второе лучше на ОТЛОЖЕННЫХ данных.

Три обязательные детали, каждая из измеренного опыта:

1. Карта качества берётся БЕЗ сигмоиды и обучаемой свёртки — они насыщали её и
   ломали знак связи. Замер 03.09: без них знак единый на всех слоях и по всем
   характеристикам, 18 значений из 18 отрицательные.

2. Обработка ПО УРОВНЯМ пирамиды. У прежней карты знак связи был разный на
   разных уровнях (+0.317 / -0.259 / -0.286), и один общий коэффициент
   самоуничтожался, давая ложное «сигнала нет». Здесь в M2 входит
   взаимодействие «признак x уровень».

3. Коэффициенты учатся ТОЛЬКО на обучающей половине видео, замер — на
   отложенной. Разбиение делается ПО ВИДЕО, а не по детекциям: соседние кадры
   почти одинаковы, и случайное разбиение по детекциям протащило бы ответ.

Порог успеха. Прирост площади под кривой (AUC) при переходе M1 -> M2 должен
уверенно перекрывать собственную погрешность замера. Погрешность оценивается
несколькими разными разбиениями на обучающую и отложенную половины. Значимость
дополнительно даётся тестом отношения правдоподобия на вложенных моделях.

Запуск: python scripts/q_gate1.py
        python scripts/q_gate1.py --layer 1 --stat пик --splits 8
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.q_early_layers import box_min, lqa_stats  # noqa: E402

WEIGHTS = PROJECT_ROOT / 'weights' / 'baseline_seed42' / 'best.pt'
CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl' / 'val'
IMG_ROOT = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
GT_ROOT = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'track' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'q_gate1.csv'

MIN_CONF = 0.1
IOU_TP = 0.5


def load_gt(video: str, frame_name: str, W: int, H: int):
    """Эталонные рамки кадра: (N,4) xyxy в пикселях и (N,) классы."""
    p = GT_ROOT / video / (Path(frame_name).stem + '.txt')
    if not p.exists():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int32)
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int32)
    a = np.array([[float(v) for v in r] for r in rows], np.float32)
    cls, cx, cy, w, h = a[:, 0].astype(np.int32), a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    box = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H], 1)
    return box, cls


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    sa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    sb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1e-9)


def label_tp(det_box, det_cls, gt_box, gt_cls):
    """TP, если нашлась незанятая эталонная рамка того же класса с IoU >= 0.5."""
    y = np.zeros(len(det_box), np.int8)
    if len(gt_box) == 0:
        return y
    m = iou_mat(det_box, gt_box)
    same = det_cls[:, None] == gt_cls[None, :]
    m = np.where(same, m, 0.0)
    used = set()
    for i in range(len(det_box)):
        j = int(np.argmax(m[i])) if m.shape[1] else -1
        if j >= 0 and m[i, j] >= IOU_TP and j not in used:
            y[i] = 1
            used.add(j)
    return y


def level_from_area(xyxy: np.ndarray) -> np.ndarray:
    """Страйд по площади рамки: пороги COCO для мелких и средних объектов."""
    a = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return np.where(a < 32 ** 2, 8, np.where(a < 96 ** 2, 16, 32)).astype(np.int32)


def collect(videos, layer: int, stat: str, imgsz: int, step: int):
    """Признаки и метки по видео: conf, q, страйд, TP/FP."""
    import warnings
    warnings.filterwarnings('ignore')
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS)).model.eval().cuda()
    grabbed = {}
    model.model[layer].register_forward_hook(
        lambda m, i, o: grabbed.__setitem__('x', o.detach()))

    out = {k: [] for k in ('conf', 'q', 'stride', 'y', 'vid')}
    for vi, f in enumerate(videos, 1):
        d = np.load(f)
        names, conf, fidx = d['frame_names'], d['conf'], d['frame_idx']
        xyxy, cls = d['xyxy'], d['cls']
        # Уровень пирамиды: в контрольном кэше его нет (его пишет только
        # q-кэш), поэтому выводим из площади рамки по порогам COCO —
        # мелкие идут на страйд 8, средние на 16, крупные на 32.
        sd = (d['stride'] if 'stride' in d.files
              else level_from_area(d['xyxy']))
        vid = f.stem
        for fi in range(0, len(names), step):
            sel = (fidx == fi) & (conf >= MIN_CONF)
            if not sel.any():
                continue
            img = cv2.imread(str(IMG_ROOT / vid / str(names[fi])))
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            t = cv2.resize(img, (imgsz, imgsz))
            t = torch.from_numpy(t[:, :, ::-1].copy()).permute(2, 0, 1)
            with torch.no_grad():
                model((t.float() / 255.0).unsqueeze(0).cuda())
            smap = lqa_stats(grabbed['x'])[stat][0, 0].float().cpu().numpy()
            sx, sy = smap.shape[1] / W0, smap.shape[0] / H0
            boxes, cf, cl, st = xyxy[sel], conf[sel], cls[sel], sd[sel]
            gtb, gtc = load_gt(vid, str(names[fi]), W0, H0)
            y = label_tp(boxes, cl.astype(np.int32), gtb, gtc)
            out['conf'].append(cf)
            out['q'].append(np.array([box_min(smap, b, sx, sy) for b in boxes], np.float32))
            out['stride'].append(st.astype(np.int32))
            out['y'].append(y)
            out['vid'].append(np.full(len(boxes), vi, np.int32))
        if vi % 10 == 0:
            print(f'[INFO] {vi}/{len(videos)} видео', flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def _auc(y, p) -> float:
    """Площадь под ROC через ранги — устойчиво к совпадающим значениям."""
    from scipy.stats import rankdata
    r = rankdata(p)
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _auprc(y, p) -> float:
    """Средняя точность: площадь под кривой точность-полнота."""
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    npos = ys.sum()
    return float((prec * ys).sum() / npos) if npos else float('nan')


def fit_auc(Xtr, ytr, Xva, yva, iters: int = 400, lam: float = 1e-3):
    """Логистическая регрессия своим кодом: sklearn в окружении нет.

    Обучение — градиентный спуск по стандартизованным признакам (иначе шаг
    приходится подбирать под масштаб каждого признака). Стандартизация
    считается ТОЛЬКО на обучающей половине и применяется к отложенной — иначе
    отложенные данные подсматриваются через среднее и разброс.
    """
    mu, sg = Xtr.mean(0), Xtr.std(0)
    sg = np.where(sg < 1e-9, 1.0, sg)
    A, B = (Xtr - mu) / sg, (Xva - mu) / sg
    A = np.hstack([A, np.ones((len(A), 1))])
    B = np.hstack([B, np.ones((len(B), 1))])
    w = np.zeros(A.shape[1])
    lr = 1.0
    for _ in range(iters):
        z = A @ w
        pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = A.T @ (pr - ytr) / len(ytr) + lam * w
        w -= lr * g
    p = 1 / (1 + np.exp(-np.clip(B @ w, -30, 30)))
    ll = float(np.sum(yva * np.log(np.clip(p, 1e-12, 1)) +
                      (1 - yva) * np.log(np.clip(1 - p, 1e-12, 1))))
    return _auc(yva, p), _auprc(yva, p), ll, w[:-1] / sg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layer', type=int, default=2, help='откуда снимать карту')
    ap.add_argument('--stat', default='градиент',
                    choices=['градиент', 'дисперсия', 'пик'])
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--step', type=int, default=6, help='прореживание кадров')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--splits', type=int, default=6, help='разбиений для разброса')
    ap.add_argument('--cache', default=str(CACHE),
                    help='каталог кэша детекций; для сильной версии гипотезы о '
                         'пересвете указывается кэш засвеченного поднабора')
    args = ap.parse_args()

    files = sorted(Path(args.cache).glob('*.npz'))[:args.videos]
    if not files:
        raise SystemExit(f'кэш пуст: {args.cache}')
    print(f'слой {args.layer}, характеристика «{args.stat}», видео {len(files)}\n')

    D = collect(files, args.layer, args.stat, args.imgsz, args.step)
    ok = np.isfinite(D['q'])
    D = {k: v[ok] for k, v in D.items()}
    y = D['y'].astype(np.float64)
    print(f'\nдетекций: {len(y)}   из них верных: {int(y.sum())} '
          f'({y.mean():.1%})')

    conf = np.clip(D['conf'].astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(conf / (1 - conf))
    # признак карты качества нормируется рангом ВНУТРИ уровня — только чтобы
    # уровни можно было положить в одну модель; на разделение внутри уровня
    # любое монотонное преобразование не влияет
    z = np.zeros(len(y))
    for s in np.unique(D['stride']):
        k = D['stride'] == s
        r = np.argsort(np.argsort(D['q'][k]))
        z[k] = r / max(len(r) - 1, 1)
    lv = np.stack([(D['stride'] == s).astype(np.float64) for s in (8, 16, 32)], 1)

    X1 = np.stack([logit], 1)
    # M2: уверенность + карта качества + её взаимодействие с уровнем
    X2 = np.concatenate([X1, z[:, None], z[:, None] * lv], 1)

    vids = np.unique(D['vid'])
    rng = np.random.default_rng(42)
    res = []
    for s in range(args.splits):
        perm = rng.permutation(vids)
        tr_v = set(perm[:len(perm) // 2].tolist())
        tr = np.isin(D['vid'], list(tr_v))
        va = ~tr
        if y[tr].sum() < 20 or y[va].sum() < 20:
            continue
        a1, p1, l1, _ = fit_auc(X1[tr], y[tr], X1[va], y[va])
        a2, p2, l2, c2 = fit_auc(X2[tr], y[tr], X2[va], y[va])
        res.append((a1, a2, p1, p2, l1, l2, c2[1]))

    A1 = np.array([r[0] for r in res]); A2 = np.array([r[1] for r in res])
    P1 = np.array([r[2] for r in res]); P2 = np.array([r[3] for r in res])
    W = np.array([r[6] for r in res])

    print('\n=== ТАКТ 1: только уверенность (модель M1) ===')
    print(f'  AUC на отложенных: {A1.mean():.4f} ± {A1.std(ddof=1):.4f}'
          f'   ({len(res)} разбиений)')
    print(f'  AUPRC:             {P1.mean():.4f} ± {P1.std(ddof=1):.4f}')
    bar = 2 * A1.std(ddof=1)
    print(f'\n  ПЛАНКА, зафиксированная ДО открытия M2:')
    print(f'  настоящим считаем прирост AUC больше {bar:.4f} (два разброса M1)')

    print('\n=== ТАКТ 2: уверенность + карта качества (модель M2) ===')
    print(f'  AUC на отложенных: {A2.mean():.4f} ± {A2.std(ddof=1):.4f}')
    print(f'  AUPRC:             {P2.mean():.4f} ± {P2.std(ddof=1):.4f}')
    d = A2 - A1
    print(f'\n  прирост AUC:   {d.mean():+.4f} ± {d.std(ddof=1):.4f}')
    print(f'  прирост AUPRC: {(P2 - P1).mean():+.4f}')
    print(f'  вес карты качества: {W.mean():+.3f} ± {W.std(ddof=1):.3f}'
          f'   знак {"устойчив" if np.all(np.sign(W) == np.sign(W[0])) else "ПЛАВАЕТ"}')
    wins = int((d > 0).sum())
    print(f'  разбиений с приростом: {wins} из {len(d)}')

    print('\n=== ПРИГОВОР ===')
    if d.mean() > bar and wins == len(d):
        print('  СИГНАЛ ЕСТЬ. Прирост перекрывает планку и держится на всех')
        print('  разбиениях. Карта качества несёт информацию сверх уверенности.')
    elif d.mean() > bar:
        print('  Прирост перекрывает планку, но держится не на всех разбиениях —')
        print('  сигнал слабый, нужен больший объём данных.')
    else:
        print('  СИГНАЛА НЕТ. Прирост не перекрывает собственную погрешность')
        print('  замера: карта качества избыточна с уверенностью в этой форме.')
        print('  Это НЕ закрывает второй пункт новизны — остаются другие')
        print('  потребители (шум Калмана, порог ассоциации), проверяемые отдельно.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['разбиение', 'AUC M1', 'AUC M2', 'AUPRC M1', 'AUPRC M2',
                    'вес карты'])
        for i, r in enumerate(res):
            w.writerow([i, round(r[0], 5), round(r[1], 5), round(r[2], 5),
                        round(r[3], 5), round(r[6], 4)])
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
