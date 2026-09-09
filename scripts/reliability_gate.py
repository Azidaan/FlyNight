"""Гейты A и B: несёт ли состояние сцены информацию о наблюдении.

Проверяются ДВЕ РАЗНЫЕ величины, и в этом главное отличие от прежней постановки.
Одна «надёжность» отвечала бы сразу на два разных вопроса, а они не совпадают:
рамка может быть настоящим объектом верного класса и при этом плохо
локализованной, и наоборот.

    ГЕЙТ A — допуск. q_det = вероятность того, что детекция соответствует
    реальному объекту, при данных: балл уверенности s, класс c, геометрия
    рамки g, состояние сцены z. Именно эта величина логично заменяет сырой
    балл в порогах трекера, включая порог создания трека. Целится в измеренный
    condition-зависимый разрыв: 21.0% объектов проваливаются под порог ночью
    против 15.5% днём.

    ГЕЙТ B — локализация. u_loc = ожидаемая геометрическая ошибка рамки при тех
    же условиях. Эта величина, и только она, логично идёт в шум наблюдения
    фильтра Калмана.

Лестницы признаков (ключевое сравнение — предпоследняя со последней):

    Q0: сырой балл, без подгонки          U0: балл + класс + геометрия
    Q1: калибровка балла                  U1: то же + состояние сцены
    Q2: балл + класс + геометрия
    Q3: то же + состояние сцены
    Q3b: Q3 + число детекций в кадре (отдельной абляцией)

Почему геометрия входит уже в Q2. Состояние сцены коррелирует с составом
объектов: ночью другой набор классов и размеров. Если размер попадёт только в
Q3 вместе со сценой, эффект размера будет засчитан как эффект освещения.

Почему число детекций вынесено в отдельную абляцию. Оно само зависит от
детектора и порога, то есть частично является функцией предсказываемой
величины; включать его в основной вариант — значит смешивать свойство сцены со
свойством конвейера.

Протокол. Разбиение по видео (5 частей), обучение калибратора на одних видео,
замер на других, предсказания собираются вне обучения. Устойчивость разницы —
парный бутстрап по видео. Критерии записаны в конце файла и печатаются до чисел.

Запуск: .venv/Scripts/python.exe scripts/reliability_gate.py --videos 50
        ... --model gbm        (проверка ёмкости модели)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT, iou_matrix  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402

IMAGES = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
FEATS = PROJECT_ROOT / 'runs' / 'reliability' / 'features.npz'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'reliability_gate.csv'
OUT_BOOT = PROJECT_ROOT / 'experiments' / 'results' / 'reliability_gate_boot.csv'

# состав признаков каждой ступени
SCENE = ['яркость_сред', 'яркость_разброс', 'доля_тёмных', 'доля_пересвета',
         'контраст']
GEOM = ['лог_размер', 'отношение_сторон', 'центр_y', 'высота_отн']
LADDER_A = {
    'Q0': [],                                  # сырой балл, без подгонки
    'Q1': ['балл'],
    'Q2': ['балл'] + GEOM + ['класс'],
    'Q3': ['балл'] + GEOM + ['класс'] + SCENE,
    'Q3b': ['балл'] + GEOM + ['класс'] + SCENE + ['число_детекций'],
}
LADDER_B = {
    'U0': ['балл'] + GEOM + ['класс'],
    'U1': ['балл'] + GEOM + ['класс'] + SCENE,
}


def scene_features(img: np.ndarray) -> list[float]:
    """Дешёвый дескриптор кадра: яркость и её распределение."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32)
    lo, hi = np.percentile(g, [5, 95])
    return [float(g.mean()), float(g.std()),
            float((g < 32).mean()), float((g > 235).mean()), float(hi - lo)]


def build_features(vids: list[str], conf_min: float) -> dict:
    """Признаки и цели по всем детекциям: сопоставление с разметкой по IoU."""
    import cv2

    name_to_id = {v: k for k, v in CLASS_NAMES.items()}
    rows, ys, ious, cerr, vid_idx, row_idx, frm_idx = [], [], [], [], [], [], []
    for vi, vid in enumerate(vids):
        gt_data = json.loads((GT / f'{vid}.json').read_text(encoding='utf-8'))
        per_frame = {}
        for fr in gt_data:
            ids, boxes, cats = [], [], []
            for l in fr.get('labels') or []:
                b = l.get('box2d')
                if not b:
                    continue
                ids.append(l['id'])
                cats.append(name_to_id.get(l['category'], -1))
                boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
            per_frame[fr['frameIndex']] = (np.array(cats, int),
                                           np.array(boxes, np.float64).reshape(-1, 4))
        cache = load_video_cache(CACHE, 'val', vid)
        fidx_all = cache['frame_idx']
        for fidx, fname, xyxy, conf, cls in iter_frames(cache):
            keep = conf >= conf_min
            # позиции этих детекций в плоских массивах кэша — по ним трекер
            # потом найдёт свою q для каждой детекции
            rows_here = np.where(fidx_all == fidx)[0][keep]
            if not keep.any():
                continue
            det, dconf, dcls = xyxy[keep], conf[keep], cls[keep].astype(int)
            img = cv2.imread(str(IMAGES / vid / fname))
            if img is None:
                continue
            H, W = img.shape[:2]
            sc = scene_features(img)
            gcls, gbox = per_frame.get(fidx, (np.zeros(0, int), np.zeros((0, 4))))
            M = iou_matrix(det.astype(np.float64), gbox)
            taken: set[int] = set()
            order = np.argsort(-dconf)           # жадное сопоставление по баллу
            match = -np.ones(len(det), int)
            for i in order:
                if not len(gbox):
                    break
                cand = np.where((gcls == dcls[i]) & (M[i] >= 0.5))[0]
                cand = [j for j in cand if j not in taken]
                if cand:
                    j = max(cand, key=lambda j: M[i, j])
                    taken.add(j)
                    match[i] = j
            for i in range(len(det)):
                x1, y1, x2, y2 = det[i]
                w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
                rows.append([float(dconf[i]),
                             float(np.log(np.sqrt(w * h))), float(w / h),
                             float((y1 + y2) / 2 / H), float(h / H),
                             float(dcls[i]), *sc, float(len(det))])
                j = match[i]
                ys.append(1 if j >= 0 else 0)
                if j >= 0:
                    ious.append(float(M[i, j]))
                    gx = (gbox[j, 0] + gbox[j, 2]) / 2
                    gy = (gbox[j, 1] + gbox[j, 3]) / 2
                    dx = (x1 + x2) / 2 - gx
                    dy = (y1 + y2) / 2 - gy
                    cerr.append(float(np.hypot(dx, dy) / np.sqrt(w * h)))
                else:
                    ious.append(np.nan)
                    cerr.append(np.nan)
                vid_idx.append(vi)
                row_idx.append(int(rows_here[i]))
                frm_idx.append(int(fidx))
    cols = (['балл'] + GEOM + ['класс'] + SCENE + ['число_детекций'])
    return {'X': np.array(rows, np.float32), 'y': np.array(ys, np.int8),
            'iou': np.array(ious, np.float32), 'cerr': np.array(cerr, np.float32),
            'video': np.array(vid_idx, np.int32), 'columns': np.array(cols),
            'row': np.array(row_idx, np.int32), 'frame': np.array(frm_idx, np.int32),
            'videos': np.array(vids)}


def design(data: dict, names: list[str]) -> np.ndarray:
    """Матрица признаков: класс разворачивается в индикаторы."""
    cols = list(data['columns'])
    X = data['X']
    parts = []
    for n in names:
        if n == 'класс':
            c = X[:, cols.index('класс')].astype(int)
            for k in sorted(set(c.tolist())):
                parts.append((c == k).astype(np.float32)[:, None])
        else:
            parts.append(X[:, cols.index(n)][:, None])
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(X), 0), np.float32)


def folds_by_video(video: np.ndarray, k: int = 5, seed: int = 0):
    """Разбиение ПО ВИДЕО: соседние кадры зависимы, делить по ним нельзя."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(video)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq)))          # частей не больше, чем видео
    for ch in np.array_split(uniq, k):
        if not len(ch):
            continue
        test = np.isin(video, ch)
        if test.all() or not test.any():
            continue
        yield ~test, test


def fit_predict(Xd, y, video, model: str, classifier: bool = True) -> np.ndarray:
    """Предсказания вне обучения: калибратор учится на одних видео, метит другие."""
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.zeros(len(y), np.float64)
    for tr, te in folds_by_video(video):
        if model == 'gbm':
            m = (HistGradientBoostingClassifier(max_iter=200, random_state=0)
                 if classifier else
                 HistGradientBoostingRegressor(max_iter=200, random_state=0))
            m.fit(Xd[tr], y[tr])
            out[te] = m.predict_proba(Xd[te])[:, 1] if classifier else m.predict(Xd[te])
        else:
            sc = StandardScaler().fit(Xd[tr])
            if classifier:
                m = LogisticRegression(max_iter=2000, C=1.0)
                m.fit(sc.transform(Xd[tr]), y[tr])
                out[te] = m.predict_proba(sc.transform(Xd[te]))[:, 1]
            else:
                m = LinearRegression().fit(sc.transform(Xd[tr]), y[tr])
                out[te] = m.predict(sc.transform(Xd[te]))
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Площадь под ROC через ранги — без зависимостей."""
    order = np.argsort(p)
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = y == 1, y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Ожидаемая ошибка калибровки: |доля верных − средний балл| по корзинам."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def metrics_cls(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'Brier': float(np.mean((p - y) ** 2)),
            'NLL': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            'ECE': ece(y, p), 'AUC': auc(y, p)}


def metrics_reg(t, p) -> dict:
    from scipy.stats import spearmanr
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return {'R2': 1 - ss_res / ss_tot, 'MAE': float(np.mean(np.abs(t - p))),
            'Spearman': float(spearmanr(t, p).statistic)}


def boot_delta(video, fn_a, fn_b, n_boot=1000, seed=0):
    """Парный бутстрап по видео для разницы качества двух ступеней."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(video)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        m = np.concatenate([np.where(video == v)[0] for v in pick])
        stats[i] = fn_a(m) - fn_b(m)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--conf-min', type=float, default=0.05,
                    help='нижняя граница: допуск касается и слабых детекций')
    ap.add_argument('--model', default='logit', choices=['logit', 'gbm'])
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    print('=== критерии, записанные ДО чисел ===')
    print('  Гейт A пройден, если Q3 лучше Q2 по Brier и NLL, и доверительный')
    print('    интервал разницы (бутстрап по видео) не накрывает ноль.')
    print('    Сравнение с Q0/Q1 к делу не относится: превосходство над сырым')
    print('    баллом — чужой результат.')
    print('  Гейт B пройден, если U1 лучше U0 по R2 на отложенных видео с тем')
    print('    же условием на интервал. Если нет — condition-aware шум Калмана')
    print('    не реализуется, но гейт A от этого не зависит.\n')

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]

    if FEATS.exists() and not args.rebuild:
        d = np.load(FEATS, allow_pickle=True)
        data = {k: d[k] for k in d.files}
        print(f'[INFO] признаки взяты из кэша {FEATS.name}')
    else:
        t0 = time.time()
        data = build_features(vids, args.conf_min)
        FEATS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FEATS, **data)
        print(f'[INFO] признаки построены за {time.time() - t0:.0f} с')

    y, video = data['y'].astype(np.float64), data['video']
    print(f'[INFO] детекций {len(y)}, из них верных {int(y.sum())} '
          f'({100 * y.mean():.1f}%), видео {len(np.unique(video))}, '
          f'модель {args.model}')

    rows, preds = [], {}
    print('\n=== ГЕЙТ A: допуск, q_det ===')
    print('%-5s %9s %9s %9s %9s' % ('шаг', 'Brier', 'NLL', 'ECE', 'AUC'))
    for name, feats in LADDER_A.items():
        if name == 'Q0':
            p = data['X'][:, 0].astype(np.float64)      # сырой балл как есть
        else:
            p = fit_predict(design(data, feats), y, video, args.model, True)
        preds[name] = p
        m = metrics_cls(y, p)
        print('%-5s %9.5f %9.5f %9.5f %9.5f'
              % (name, m['Brier'], m['NLL'], m['ECE'], m['AUC']))
        rows += [{'гейт': 'A', 'ступень': name, 'метрика': k,
                  'значение': round(v, 6)} for k, v in m.items()]

    brows = []
    for a, b in (('Q2', 'Q3'), ('Q1', 'Q2'), ('Q3', 'Q3b')):
        for metric in ('Brier', 'NLL', 'AUC'):
            f = (lambda idx, p=None: metrics_cls(y[idx], p[idx])[metric])
            lo, hi = boot_delta(video,
                                lambda idx: metrics_cls(y[idx], preds[a][idx])[metric],
                                lambda idx: metrics_cls(y[idx], preds[b][idx])[metric],
                                args.boot)
            point = metrics_cls(y, preds[a])[metric] - metrics_cls(y, preds[b])[metric]
            better = 'да' if (lo > 0 or hi < 0) else 'НЕТ'
            sign = '+' if metric == 'AUC' else '-'
            print('  %s против %s по %-5s: разница %+.5f [%+.5f, %+.5f] значимо: %s'
                  % (a, b, metric, point, lo, hi, better))
            brows.append({'гейт': 'A', 'сравнение': f'{a} vs {b}', 'метрика': metric,
                          'разница': round(point, 6), 'ци_низ': round(lo, 6),
                          'ци_верх': round(hi, 6), 'значимо': lo > 0 or hi < 0})

    print('\n=== ГЕЙТ B: локализация, u_loc ===')
    ok = np.isfinite(data['iou'])
    sub = {k: (v[ok] if isinstance(v, np.ndarray) and v.shape[:1] == y.shape else v)
           for k, v in data.items()}
    for target_name, target in (('1-IoU', 1 - data['iou'][ok].astype(np.float64)),
                                ('ошибка_центра', data['cerr'][ok].astype(np.float64))):
        print(f'-- цель: {target_name}, случаев {ok.sum()} --')
        pr = {}
        for name, feats in LADDER_B.items():
            p = fit_predict(design(sub, feats), target, sub['video'],
                            args.model, False)
            pr[name] = p
            m = metrics_reg(target, p)
            print('   %-3s R2=%+.4f  MAE=%.4f  Spearman=%+.4f'
                  % (name, m['R2'], m['MAE'], m['Spearman']))
            rows += [{'гейт': f'B:{target_name}', 'ступень': name, 'метрика': k,
                      'значение': round(v, 6)} for k, v in m.items()]
        lo, hi = boot_delta(sub['video'],
                            lambda idx: metrics_reg(target[idx], pr['U1'][idx])['R2'],
                            lambda idx: metrics_reg(target[idx], pr['U0'][idx])['R2'],
                            args.boot)
        point = metrics_reg(target, pr['U1'])['R2'] - metrics_reg(target, pr['U0'])['R2']
        print('   U1 лучше U0 по R2: %+.5f [%+.5f, %+.5f] значимо: %s'
              % (point, lo, hi, 'да' if lo > 0 else 'НЕТ'))
        brows.append({'гейт': f'B:{target_name}', 'сравнение': 'U1 vs U0',
                      'метрика': 'R2', 'разница': round(point, 6),
                      'ци_низ': round(lo, 6), 'ци_верх': round(hi, 6),
                      'значимо': lo > 0})

    # связь допуска с геометрией: годится ли одна величина на оба механизма
    print('\n=== несёт ли q_det геометрическую информацию ===')
    q = preds['Q3'][ok]
    err = 1 - data['iou'][ok]
    dec = np.clip((q * 10).astype(int), 0, 9)
    print('   дециль q_det :', ' '.join('%5.2f' % (dec == d).mean() for d in range(10)))
    print('   средняя 1-IoU:', ' '.join(
        '%5.3f' % err[dec == d].mean() if (dec == d).any() else '  —  '
        for d in range(10)))

    # q_det вне обучения — именно эти значения пойдут в трекер: калибратор
    # никогда не видел то видео, к которому применяется
    qpath = FEATS.parent / 'q_det.npz'
    np.savez_compressed(qpath, video=video, row=data['row'], frame=data['frame'],
                        videos=data['videos'], conf=data['X'][:, 0],
                        **{k: preds[k] for k in ('Q1', 'Q2', 'Q3')})
    print(f'[OK ] {qpath}  (q_det по каждой детекции, вне обучения)')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT_BOOT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0])); w.writeheader(); w.writerows(brows)
    print(f'\n[OK ] {OUT}\n[OK ] {OUT_BOOT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
