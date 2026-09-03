"""Оживает ли карта качества, если снимать её из РАННИХ слоёв, а не из шеи.

Решающий замер для второго пункта новизны.

Предыстория. Признак деградации умирает к третьему слою: отношение размаха
сигнала ночь/день на входе 0.489, на слое 0 — 0.651, слое 1 — 0.724, слое 2 —
0.896, слое 3 — уже 0.987, и до конца сети около единицы. Карта качества всё это
время считалась из признаков ШЕИ (слои 16/19/22), то есть из мёртвой зоны.
Отсюда её почти постоянное значение 0.9766 +- 0.0091 и скачущий по уровням знак
связи с уверенностью (+0.317 / -0.259 / -0.286).

Вопрос: если снять те же три характеристики со слоёв 0-2, окрепнет ли связь?

Отличия от прежней постановки, сделанные намеренно:
    сигмоиды и обучаемой свёртки НЕТ — именно они насыщали карту (три входа
    неотрицательны, веса и смещение положительны, аргумент сигмоиды не мог
    стать отрицательным). Здесь три характеристики докладываются ПОРОЗНЬ, в
    сыром виде. Насыщаться нечему;
    для сопоставимости те же слои шеи считаются тем же кодом.

Агрегация по рамке — минимум: замер 9.5 показал, что он даёт единый знак связи
на всех уровнях, а усреднение нет.

Запуск: python scripts/q_early_layers.py
        python scripts/q_early_layers.py --videos 20
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS = PROJECT_ROOT / 'weights' / 'baseline_seed42' / 'best.pt'
CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl' / 'val'
IMG_ROOT = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'q_early_layers.csv'

PROBE = [0, 1, 2, 16, 19, 22]     # ранние слои и шея для сравнения
MIN_CONF = 0.1


def lqa_stats(x):
    """Три характеристики LQA, каждая порознь, без свёртки и без сигмоиды."""
    import torch
    import torch.nn.functional as F
    k, pad = 3, 1
    avg = F.avg_pool2d(x, k, stride=1, padding=pad)
    avg_sq = F.avg_pool2d(x * x, k, stride=1, padding=pad)
    disp = (avg_sq - avg * avg).clamp_min(0).mean(1, keepdim=True)
    peak = (F.max_pool2d(x, k, stride=1, padding=pad) - avg).mean(1, keepdim=True)
    c = x.shape[1]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype)
    sob = torch.stack([kx, kx.t()]).unsqueeze(1).repeat(c, 1, 1, 1)
    g = F.conv2d(x, sob, padding=1, groups=c).view(x.shape[0], c, 2, *x.shape[-2:])
    grad = torch.sqrt(g[:, :, 0] ** 2 + g[:, :, 1] ** 2 + 1e-12).mean(1, keepdim=True)
    return {'дисперсия': disp, 'пик': peak, 'градиент': grad}


def box_min(m: np.ndarray, box, sx: float, sy: float) -> float:
    """Минимум карты внутри рамки; рамка переводится в клетки карты."""
    h, w = m.shape
    x1 = int(np.clip(box[0] * sx, 0, w - 1)); x2 = int(np.clip(box[2] * sx, 0, w - 1))
    y1 = int(np.clip(box[1] * sy, 0, h - 1)); y2 = int(np.clip(box[3] * sy, 0, h - 1))
    patch = m[y1:y2 + 1, x1:x2 + 1]
    return float(patch.min()) if patch.size else float('nan')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--frame-step', type=int, default=4,
                    help='прореживание кадров внутри видео — замер, не оценка')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    import cv2
    import torch
    from ultralytics import YOLO

    files = sorted(CACHE.glob('*.npz'))[:args.videos]
    if not files:
        raise SystemExit(f'кэш пуст: {CACHE} — запустите scripts/build_day_cache.py')

    model = YOLO(str(WEIGHTS)).model.eval().cuda()
    grabbed: dict[int, torch.Tensor] = {}
    for idx in PROBE:
        model.model[idx].register_forward_hook(
            lambda m, i, o, idx=idx: grabbed.__setitem__(idx, o.detach()))

    acc: dict[tuple, list] = {}
    n_det = 0
    for vi, f in enumerate(files, 1):
        d = np.load(f)
        names, conf = d['frame_names'], d['conf']
        fidx, xyxy = d['frame_idx'], d['xyxy']
        vid = f.stem
        for fi in range(0, len(names), args.frame_step):
            sel = (fidx == fi) & (conf >= MIN_CONF)
            if not sel.any():
                continue
            img_p = IMG_ROOT / vid / str(names[fi])
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            t = cv2.resize(img, (args.imgsz, args.imgsz))
            t = torch.from_numpy(t[:, :, ::-1].copy()).permute(2, 0, 1)
            t = (t.float() / 255.0).unsqueeze(0).cuda()
            with torch.no_grad():
                model(t)
            boxes, cf = xyxy[sel], conf[sel]
            n_det += len(boxes)
            for idx in PROBE:
                fm = grabbed[idx]
                for sname, smap in lqa_stats(fm).items():
                    m = smap[0, 0].float().cpu().numpy()
                    sx, sy = m.shape[1] / W0, m.shape[0] / H0
                    vals = [box_min(m, b, sx, sy) for b in boxes]
                    acc.setdefault((idx, sname), []).append(
                        np.stack([np.array(vals, np.float32), cf]))
        if vi % 10 == 0:
            print(f'[INFO] {vi}/{len(files)} видео, детекций {n_det}', flush=True)

    print(f'\nдетекций всего: {n_det}\n')
    print('%-8s %-12s %10s %12s %12s' % ('слой', 'характеристика', 'связь',
                                         'сред', 'разброс'))
    rows = []
    best = {}
    for (idx, sname), chunks in sorted(acc.items()):
        a = np.concatenate(chunks, axis=1)
        q, cf = a[0], a[1]
        ok = np.isfinite(q)
        q, cf = q[ok], cf[ok]
        if len(q) < 100:
            continue
        r = float(np.corrcoef(q, cf)[0, 1])
        rows.append((idx, sname, len(q), round(r, 4),
                     round(float(q.mean()), 5), round(float(q.std()), 5)))
        best.setdefault(idx, []).append(abs(r))
        print('%-8d %-12s %10.3f %12.4f %12.4f' % (idx, sname, r, q.mean(), q.std()))

    print('\n=== приговор: сила связи по слоям (средний модуль) ===')
    early = [np.mean(best[i]) for i in (0, 1, 2) if i in best]
    neck = [np.mean(best[i]) for i in (16, 19, 22) if i in best]
    for i in PROBE:
        if i in best:
            tag = 'ранний ствол' if i < 3 else 'шея'
            print('  слой %-3d %-14s %.3f' % (i, tag, float(np.mean(best[i]))))
    if early and neck:
        e, n = float(np.mean(early)), float(np.mean(neck))
        print(f'\n  ранние слои: {e:.3f}   шея: {n:.3f}   отношение {e / n:.2f}')
        if e > n * 1.3:
            print('\n  СИГНАЛ ОЖИВАЕТ В РАННИХ СЛОЯХ. Перенос карты качества из шеи')
            print('  оправдан замером — второй пункт новизны получает опору.')
        elif e > n * 1.05:
            print('\n  Слабое улучшение. Перенос помогает, но не решает.')
        else:
            print('\n  ПЕРЕНОС НЕ ПОМОГАЕТ. Дело не в месте: характеристики LQA меряют')
            print('  не то, что нужно, где бы их ни снимать.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['слой', 'характеристика', 'детекций', 'связь', 'сред', 'разброс'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
