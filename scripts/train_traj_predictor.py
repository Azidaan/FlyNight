"""Гейт 1 ветки WTConv1d: бьёт ли обучаемая модель движения фильтр Калмана.

Постановка записана заранее в dissertation/Wavelet/Wavelet.md, п. 7.3.

Что сравнивается. На одних и тех же окнах наблюдений и одних и тех же разрывах:

    Калман CV      — модель постоянной скорости, как в штатном ByteTrack;
    Калман + db3   — лучшее ручное вейвлетное очищение из гейта Б (для масштаба);
    WTConv1d       — обучаемая модель движения (models/wtconv1d.py);
    обычная свёртка — та же сеть с глубинной свёрткой вместо вейвлетного слоя.

Последняя строка обязательна: без неё выигрыш нельзя приписать вейвлетной
конструкции, а не самому факту обучения.

Данные. Обучение — траектории train-сплита box_track_20 (1400 видео), проверка
на отложенных видео того же сплита, итоговый замер — val-сплит, те же 50 видео,
на которых считались гейты А и Б. Видео не пересекаются ни в одной паре
множеств: единица разделения — видеозапись, а не окно.

Наблюдения на итоговом замере берутся двух видов, как в гейте Б: эталонные
центры разметки и центры детекций YOLO26 из кэша. Цель всегда эталонная.

Вход модели — приращения координат по кадрам (а не сами координаты): так модель
не зависит от места объекта в кадре. Выход — смещение от последнего наблюдения
до положения через каждый из разрывов. Разметку модель не видит нигде, кроме
обучения на прошлом самих траекторий, — как и Калман.

Обучение идёт с добавлением шума к наблюдениям (по умолчанию 1.5 px на ось):
на итоговом замере вход приходит из детектора, у которого измеренный шум
1.9 px по медиане. Без этого модель училась бы на идеально гладких рядах.

Запуск: .venv/Scripts/python.exe scripts/train_traj_predictor.py
        ... --videos-train 400 --epochs 30 --wt-type db3
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.wtconv1d import build_model  # noqa: E402
from scripts.wavelet_gate_b import kalman_cv, load_runs, wavelet_clean  # noqa: E402
from scripts.wavelet_gate_b_dets import CACHE, boot_ci, build_pairs  # noqa: E402

TRAIN_LABELS = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'train'
VAL_LABELS = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'wtconv1d_gate1.csv'
OUT_BOOT = PROJECT_ROOT / 'experiments' / 'results' / 'wtconv1d_gate1_bootstrap.csv'
CKPT = PROJECT_ROOT / 'runs' / 'wtconv1d'
SCALE = 10.0          # пиксели -> единицы модели


def windows_from_run(run: np.ndarray, W: int, gaps: tuple):
    """Окна наблюдений и цели для одного непрерывного куска траектории.

    Требуется, чтобы для окна были определены ВСЕ разрывы сразу: тогда набор
    случаев одинаков для всех разрывов и сравнения между ними парные.
    """
    n, gmax = len(run), max(gaps)
    xs, ys = [], []
    for end in range(W, n - gmax + 1):
        win = run[end - W:end]
        xs.append(win)
        ys.append([run[end + g - 1] - win[-1] for g in gaps])
    return xs, ys


def build_dataset(files, W: int, gaps: tuple, stride: int = 1):
    """(окна (N,W,2), цели (N,G,2), номера видео (N,)) по списку файлов разметки."""
    need = W + max(gaps)
    X, Y, V = [], [], []
    for vi, p in enumerate(files):
        for run in load_runs(p, need):
            xs, ys = windows_from_run(run, W, gaps)
            X += xs[::stride]
            Y += ys[::stride]
            V += [vi] * len(xs[::stride])
    if not X:
        raise SystemExit('пустой набор данных')
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(V, np.int32))


def to_input(win: np.ndarray, noise: float, rng=None) -> np.ndarray:
    """Окно (N,W,2) -> приращения (N,2,W) в единицах модели, с шумом наблюдений."""
    w = win.copy()
    if noise > 0 and rng is not None:
        w = w + rng.normal(0.0, noise, w.shape).astype(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.transpose(d, (0, 2, 1)) / SCALE


def train(model, Xtr, Ytr, Xva, Yva, args, device) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(Xtr) // args.batch))
    lossf = nn.SmoothL1Loss(beta=0.5)
    rng = np.random.default_rng(0)
    best, best_state = float('inf'), None
    Xva_t = torch.tensor(to_input(Xva, 0.0)).to(device)
    Yva_t = torch.tensor(Yva / SCALE).to(device)

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot = nb = 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.tensor(to_input(Xtr[idx], args.noise, rng)).to(device)
            yb = torch.tensor(Ytr[idx] / SCALE).to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            err = torch.linalg.norm(model(Xva_t) - Yva_t, dim=-1) * SCALE
            med = err.median(dim=0).values.cpu().numpy()
        score = float(med.mean())
        mark = ''
        if score < best:
            best, best_state = score, {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
            mark = ' <- лучшее'
        print('  эпоха %2d/%d  обучение %.4f  проверка (медиана px): %s  сред %.3f%s'
              % (ep + 1, args.epochs, tot / max(nb, 1),
                 ' / '.join('%.2f' % v for v in med), score, mark), flush=True)
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_windows(vids, W: int, gaps: tuple, conf: float, iou: float):
    """Окна итогового замера по val: источник -> (окна, цели, видео)."""
    data = {s: {'X': [], 'Y': [], 'V': []} for s in ('эталон', 'детекции')}
    need = W + max(gaps)
    n_runs = 0
    for vid in vids:
        for gt_run, det_run in build_pairs(vid, conf, iou):
            if len(gt_run) < need:
                continue
            n_runs += 1
            xs_gt, ys = windows_from_run(gt_run, W, gaps)
            xs_dt, _ = windows_from_run(det_run, W, gaps)
            # цели всегда эталонные; отличается только источник наблюдений
            data['эталон']['X'] += xs_gt
            data['детекции']['X'] += xs_dt
            for s in data:
                data[s]['Y'] += ys
                data[s]['V'] += [vid] * len(xs_gt)
    for s in data:
        data[s]['X'] = np.asarray(data[s]['X'], np.float32)
        data[s]['Y'] = np.asarray(data[s]['Y'], np.float32)
    print(f'[INFO] итоговый замер: кусков {n_runs}, '
          f'окон {len(data["эталон"]["X"])}')
    return data


def kalman_errors(X, Y, gaps, clean: str | None = None,
                  q: float = 1.0, r: float = 4.0):
    """Ошибки предсказания Калмана (и, если задано, с очищением) — (N,G)."""
    out = np.empty((len(X), len(gaps)), np.float64)
    for i, win in enumerate(X):
        sig = [win[:, 0].astype(np.float64), win[:, 1].astype(np.float64)]
        if clean:
            sig = [wavelet_clean(s, clean) for s in sig]
        for gi, g in enumerate(gaps):
            p = np.array([kalman_cv(sig[0], g, q, r), kalman_cv(sig[1], g, q, r)])
            out[i, gi] = np.hypot(*(p - (win[-1] + Y[i, gi])))
    return out


def tune_kalman(X, Y, gaps, noise: float, n_max: int = 3000, seed: int = 0):
    """Подбор (q, r) фильтра по отложенным видео ОБУЧАЮЩЕГО сплита.

    Зачем. Значения q=1, r=4 из гейта Б никто не настраивал, а обучаемая модель
    настроена по определению. Сравнение с ненастроенной базой даёт ложный
    выигрыш и на защите не держится — поэтому базе даётся та же возможность
    подобрать свои параметры, и на тех же данных, что видела модель.
    Наблюдения зашумляются так же, как при обучении.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_max, len(X)), replace=False)
    Xs = X[idx] + rng.normal(0.0, noise, X[idx].shape).astype(np.float32)
    Ys = Y[idx]
    best, best_qr = float('inf'), (1.0, 4.0)
    for q in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for r in (0.25, 1.0, 4.0, 9.0, 25.0):
            e = kalman_errors(Xs, Ys, gaps, q=q, r=r)
            score = float(np.mean([np.median(e[:, gi]) for gi in range(len(gaps))]))
            if score < best:
                best, best_qr = score, (q, r)
    print('[INFO] настройка Калмана на отложенных видео: q=%.2f r=%.2f '
          '(средняя медиана %.3f px, у ненастроенного q=1 r=4)'
          % (*best_qr, best))
    return best_qr


def model_errors(model, X, Y, device) -> np.ndarray:
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.tensor(to_input(X[i:i + 4096], 0.0)).to(device)
            pred = model(xb).cpu().numpy() * SCALE
            errs.append(np.linalg.norm(pred - Y[i:i + 4096], axis=-1))
    return np.concatenate(errs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--gaps', nargs='+', type=int, default=[2, 3, 5])
    ap.add_argument('--videos-train', type=int, default=400)
    ap.add_argument('--videos-val', type=int, default=50)
    ap.add_argument('--holdout', type=int, default=40, help='видео на проверку')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--noise', type=float, default=1.5, help='шум наблюдений, px')
    ap.add_argument('--wt-type', default='db3')
    ap.add_argument('--wt-levels', type=int, default=2)
    ap.add_argument('--channels', type=int, default=48)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--models', nargs='+', default=['wtconv', 'plain'])
    args = ap.parse_args()

    gaps = tuple(args.gaps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] устройство {device}, базис {args.wt_type}, '
          f'окно {args.window}, разрывы {gaps}')

    files = sorted(TRAIN_LABELS.glob('*.json'))[:args.videos_train]
    tr_files, va_files = files[args.holdout:], files[:args.holdout]
    t0 = time.time()
    Xtr, Ytr, _ = build_dataset(tr_files, args.window, gaps)
    Xva, Yva, _ = build_dataset(va_files, args.window, gaps)
    print(f'[INFO] обучение: {len(Xtr)} окон из {len(tr_files)} видео; '
          f'проверка: {len(Xva)} окон из {len(va_files)} видео '
          f'({time.time() - t0:.0f} с)')

    vids = [p.stem for p in sorted(VAL_LABELS.glob('*.json'))
            if (CACHE / 'val' / f'{p.stem}.npz').exists()][:args.videos_val]
    ev = eval_windows(vids, args.window, gaps, args.conf, args.iou)

    # --- базы сравнения: считаются один раз на тех же окнах ---
    q_t, r_t = tune_kalman(Xva, Yva, gaps, args.noise)
    base_err = {}
    for src in ('эталон', 'детекции'):
        t0 = time.time()
        base_err[(src, 'Калман CV')] = kalman_errors(
            ev[src]['X'], ev[src]['Y'], gaps)
        base_err[(src, 'Калман настроенный')] = kalman_errors(
            ev[src]['X'], ev[src]['Y'], gaps, q=q_t, r=r_t)
        base_err[(src, 'Калман+db3')] = kalman_errors(
            ev[src]['X'], ev[src]['Y'], gaps, clean='db3')
        print(f'[INFO] базы посчитаны для «{src}» ({time.time() - t0:.0f} с)')

    # --- обучение моделей ---
    CKPT.mkdir(parents=True, exist_ok=True)
    for mixer in args.models:
        name = f'WTConv1d {args.wt_type}' if mixer == 'wtconv' else 'обычная свёртка'
        print(f'\n=== обучение: {name} ===')
        torch.manual_seed(42)
        model = build_model(mixer=mixer, wt_type=args.wt_type, gaps=gaps,
                            channels=args.channels, blocks=args.blocks,
                            wt_levels=args.wt_levels).to(device)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  обучаемых параметров {n_par}')
        model = train(model, Xtr, Ytr, Xva, Yva, args, device)
        torch.save(model.state_dict(), CKPT / f'{mixer}_{args.wt_type}.pt')
        for src in ('эталон', 'детекции'):
            base_err[(src, name)] = model_errors(
                model, ev[src]['X'], ev[src]['Y'], device)

    methods = ['Калман CV', 'Калман настроенный', 'Калман+db3'] + [
        (f'WTConv1d {args.wt_type}' if m == 'wtconv' else 'обычная свёртка')
        for m in args.models]

    # --- таблица ---
    rows = []
    print('\n=== ошибка предсказания центра, пиксели (медиана) ===')
    for src in ('эталон', 'детекции'):
        print(f'-- {src} --')
        print('%-22s' % 'метод' + ''.join('%12s' % f'разрыв {g}' for g in gaps))
        for m in methods:
            e = base_err[(src, m)]
            line = '%-22s' % m
            for gi, g in enumerate(gaps):
                v = float(np.median(e[:, gi]))
                line += '%12.3f' % v
                rows.append({'источник': src, 'разрыв': g, 'метод': m,
                             'медиана_ошибки': round(v, 4), 'случаев': len(e)})
            print(line)
        print('')

    # --- парный бутстрап по видео ---
    # база сравнения — ЛУЧШИЙ из вариантов Калмана: сравнение с худшим
    # завысило бы выигрыш обучения
    base_name = min(('Калман CV', 'Калман настроенный'),
                    key=lambda m: np.mean([np.median(base_err[('детекции', m)][:, gi])
                                           for gi in range(len(gaps))]))
    print('=== устойчивость: парный бутстрап с кластеризацией по видео ===')
    print(f'(база сравнения — {base_name})')
    brows = []
    for src in ('эталон', 'детекции'):
        vid_arr = np.asarray(ev[src]['V'])
        print(f'-- {src} --')
        for m in methods[1:]:
            for gi, g in enumerate(gaps):
                pv = {}
                for v in set(vid_arr):
                    mask = vid_arr == v
                    pv[v] = (base_err[(src, base_name)][mask, gi],
                             base_err[(src, m)][mask, gi])
                point, lo, hi = boot_ci(pv, args.boot)
                sign = 'значимо' if lo > 0 or hi < 0 else 'НЕ значимо'
                print('   %-22s лучше базы, разрыв %d: %+.2f%% [%+.2f, %+.2f] %s'
                      % (m, g, point, lo, hi, sign))
                brows.append({'источник': src, 'разрыв': g, 'метод': m,
                              'база': base_name,
                              'лучше_Калмана_%': round(point, 3),
                              'ци_низ': round(lo, 3), 'ци_верх': round(hi, 3),
                              'значимо': lo > 0 or hi < 0, 'видео': len(pv)})
        print('')

    # --- приговор по записанному заранее критерию ---
    gmax = max(gaps)
    gi = gaps.index(gmax)
    wt_name = f'WTConv1d {args.wt_type}'
    verdict = None
    if wt_name in methods:
        row = [b for b in brows if b['источник'] == 'детекции'
               and b['разрыв'] == gmax and b['метод'] == wt_name][0]
        print('=== гейт 1 (критерий записан заранее) ===')
        print('  требуется: на разрыве %d кадров лучше НАСТРОЕННОГО Калмана'
              % gmax)
        print('  на 15%% и более,')
        print('  доверительный интервал не накрывает ноль.')
        print('  получено: %+.2f%% [%+.2f, %+.2f]'
              % (row['лучше_Калмана_%'], row['ци_низ'], row['ци_верх']))
        passed = row['лучше_Калмана_%'] >= 15.0 and row['ци_низ'] > 0
        verdict = 'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'
        print(f'  ГЕЙТ 1 {verdict}')
        if not passed:
            print('  В трекер не идём: по шкале гейта А такой прирост даст')
            print('  доли пункта IDF1. Ветка закрывается с измеренным числом.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT_BOOT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0]))
        w.writeheader(); w.writerows(brows)
    print(f'\n[OK ] {OUT}\n[OK ] {OUT_BOOT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
