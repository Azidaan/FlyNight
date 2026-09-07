"""Гейт Б: работает ли выбор базиса по нулевым моментам на РЕАЛЬНЫХ треках.

Зачем. Замысел вейвлетной модели движения обосновывает выбор базиса
соответствием числа нулевых моментов порядку модели движения: вейвлет с N
нулевыми моментами обнуляет полиномы степени ниже N, модель постоянной
скорости — полином первой степени, значит db2 обнуляет ровно её и оставляет в
деталях только отклонение. Хаар с одним моментом пропускает равномерное
движение в полосу деталей.

На синтетике это подтвердилось точно (энергия деталей на равномерном движении:
Хаар 1.768, db2 2.46e-14). Но синтетика делалась при 30 кадрах в секунду, а
разметка BDD100K идёт при 5.05 — и полоса дрожания 3-8 Гц там выше Найквиста,
то есть ненаблюдаема. Значит на этих данных вейвлету остаётся отделять
равномерное движение только от ШУМА ДЕТЕКЦИИ, а синтетика говорит, что без
дрожания выигрыш падает до +3%.

Вопрос гейта, ровно два:

    1. Бьёт ли вейвлетное очищение честную базу (МНК-сглаживание) на реальных
       треках 5 Гц?
    2. Бьёт ли db2 Хаара? Если нет — ось новизны на реальных данных не
       работает, сколько бы она ни работала на синтетике.

Второй вопрос важнее первого: он проверяет сам заявляемый вклад.

Честность сравнения. База — МНК-сглаживание, НЕ разность двух последних точек:
двухточечная разность убивается любым шумом, проигрывает любому сглаживанию, и
сравнение с ней даёт ложный выигрыш (в песочнице оно давало +70%, причём +49%
даже при нулевом дрожании, чего быть не могло). Единственное отличие
вейвлетного плеча от базы — шаг очищения; экстраполяция у них одна и та же.

Параметры пересчитаны под 5 Гц. Окно 12 кадров, а не 64: медиана длины трека 13
кадров, и окно из исходного замысла длиннее 92% треков. Разрывы 2, 3, 5 кадров
(0.4-1.0 с), а не 5-30 (1-6 с).

Обязательное по ловушкам исходного замысла: краевой режим symmetric, снятие
линейного тренда до разложения и возврат после.

Запуск: python scripts/wavelet_gate_b.py
        python scripts/wavelet_gate_b.py --window 16 --videos 100
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pywt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRACKS = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'wavelet_gate_b.csv'
BASES = ['haar', 'db2', 'db3', 'db4', 'coif1']


def load_runs(path: Path, min_len: int):
    """Непрерывные куски траекторий: (кадры подряд) -> массив центров (N,2)."""
    data = json.loads(path.read_text(encoding='utf-8'))
    per_id: dict[str, list] = defaultdict(list)
    for fr in data:
        fi = fr['frameIndex']
        for l in fr.get('labels') or []:
            b = l.get('box2d')
            if not b:
                continue
            per_id[l['id']].append((fi, (b['x1'] + b['x2']) / 2,
                                    (b['y1'] + b['y2']) / 2))
    runs = []
    for pts in per_id.values():
        pts.sort()
        cur = [pts[0]]
        for p in pts[1:]:
            if p[0] == cur[-1][0] + 1:      # только подряд идущие кадры
                cur.append(p)
            else:
                if len(cur) >= min_len:
                    runs.append(np.array([[q[1], q[2]] for q in cur], np.float64))
                cur = [p]
        if len(cur) >= min_len:
            runs.append(np.array([[q[1], q[2]] for q in cur], np.float64))
    return runs


def extrapolate(sig: np.ndarray, fit_n: int, ahead: int) -> float:
    """Прямая по последним fit_n точкам, продолженная на ahead вперёд."""
    y = sig[-fit_n:]
    t = np.arange(len(sig) - fit_n, len(sig), dtype=np.float64)
    a, b = np.polyfit(t, y, 1)
    return float(a * (len(sig) - 1 + ahead) + b)


def wavelet_clean(sig: np.ndarray, wname: str) -> np.ndarray:
    """Снять тренд, обнулить самые мелкие детали, вернуть тренд обратно."""
    n = len(sig)
    t = np.arange(n, dtype=np.float64)
    a, b = np.polyfit(t, sig, 1)
    resid = sig - (a * t + b)
    w = pywt.Wavelet(wname)
    lvl = min(2, max(1, pywt.dwt_max_level(n, w.dec_len)))
    c = pywt.wavedec(resid, w, mode='symmetric', level=lvl)
    c[-1] = np.zeros_like(c[-1])            # самая высокочастотная полоса
    clean = pywt.waverec(c, w, mode='symmetric')[:n]
    return clean + (a * t + b)


def kalman_cv(sig: np.ndarray, ahead: int, q: float = 1.0, r: float = 4.0) -> float:
    """Одномерный фильтр постоянной скорости; общий, НЕ настройки ByteTrack."""
    x = np.array([sig[0], 0.0])
    P = np.diag([r, 10.0])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
    H = np.array([[1.0, 0.0]])
    for z in sig[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + r
        K = (P @ H.T / S).ravel()
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H)) @ P
    for _ in range(ahead):
        x = F @ x
    return float(x[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=int, default=12, help='кадров наблюдения')
    ap.add_argument('--fit', type=int, default=8, help='кадров под МНК-прямую')
    ap.add_argument('--gaps', nargs='+', type=int, default=[2, 3, 5])
    ap.add_argument('--videos', type=int, default=50)
    args = ap.parse_args()

    W, F = args.window, min(args.fit, args.window)
    files = sorted(TRACKS.glob('*.json'))[:args.videos]
    if not files:
        raise SystemExit(f'нет разметки: {TRACKS}')

    need = W + max(args.gaps)
    runs = []
    for p in files:
        runs += load_runs(p, need + 1)
    print(f'[INFO] видео {len(files)}, пригодных кусков траекторий {len(runs)}')
    if not runs:
        raise SystemExit('нет достаточно длинных непрерывных треков')

    # Вейвлет как ЗАМЕНА экстраполятора проиграл Калману 21% (прогон 06.09),
    # значит вопрос поставлен неверно. Правильный — помогает ли очищение
    # САМОМУ Калману: очищенный сигнал идёт в его штатный вход, модель
    # движения не заменяется.
    methods = (['МНК (база)', 'Калман CV']
               + [f'вейвлет {b}' for b in BASES]
               + [f'Калман+{b}' for b in BASES])
    err: dict[tuple, list] = defaultdict(list)

    for run in runs:
        n = len(run)
        for gap in args.gaps:
            # все допустимые позиции разрыва внутри куска
            for end in range(W, n - gap):
                win = run[end - W:end]          # окно наблюдения (W, 2)
                target = run[end + gap - 1]     # истинное положение после разрыва
                pred = {m: np.zeros(2) for m in methods}
                for ax in (0, 1):
                    s = win[:, ax]
                    pred['МНК (база)'][ax] = extrapolate(s, F, gap)
                    pred['Калман CV'][ax] = kalman_cv(s, gap)
                    for bname in BASES:
                        c = wavelet_clean(s, bname)
                        pred[f'вейвлет {bname}'][ax] = extrapolate(c, F, gap)
                        pred[f'Калман+{bname}'][ax] = kalman_cv(c, gap)
                for m in methods:
                    err[(gap, m)].append(float(np.hypot(*(pred[m] - target))))

    print(f'\nокно {W} кадров, прямая по {F}, разрыв в кадрах')
    print('Ошибка предсказания центра рамки, пиксели (медиана)\n')
    print('%-16s' % 'метод' + ''.join('%12s' % f'разрыв {g}' for g in args.gaps))
    rows = []
    med = {}
    for m in methods:
        line = '%-16s' % m
        for g in args.gaps:
            v = float(np.median(err[(g, m)]))
            med[(g, m)] = v
            line += '%12.3f' % v
            rows.append({'разрыв': g, 'метод': m, 'медиана_ошибки': round(v, 4),
                         'случаев': len(err[(g, m)])})
        print(line)
    print('\nслучаев на клетку: %d' % len(err[(args.gaps[0], methods[0])]))

    print('')
    print('=== вопрос 1: помогает ли очищение самому Калману ===')
    kal = {g: med[(g, 'Калман CV')] for g in args.gaps}
    for g in args.gaps:
        best = min(BASES, key=lambda b: med[(g, 'Калман+' + b)])
        bv = med[(g, 'Калман+' + best)]
        print('  разрыв %d: Калман %.3f, лучший Калман+%s %.3f -> %+.1f%%'
              % (g, kal[g], best, bv, 100 * (kal[g] - bv) / kal[g]))

    print('')
    print('=== вопрос 2: держится ли порядок базисов (ось новизны) ===')
    for g in args.gaps:
        h, d = med[(g, 'Калман+haar')], med[(g, 'Калман+db2')]
        print('  разрыв %d: Калман+Хаар %.3f, Калман+db2 %.3f -> %+.1f%% за db2'
              % (g, h, d, 100 * (h - d) / h))

    print('')
    print('=== приговор ===')
    gains = [100 * (kal[g] - min(med[(g, 'Калман+' + b)] for b in BASES)) / kal[g]
             for g in args.gaps]
    spread = [100 * (max(med[(g, 'Калман+' + b)] for b in BASES)
                     - min(med[(g, 'Калман+' + b)] for b in BASES)) / kal[g]
              for g in args.gaps]
    wins = sum(med[(g, 'Калман+db2')] < med[(g, 'Калман+haar')] for g in args.gaps)
    print('  прирост к штатному Калману: %+.1f%%' % np.mean(gains))
    print('  разброс между базисами:     %.1f%%' % np.mean(spread))
    print('  db2 лучше Хаара: %d из %d' % (wins, len(args.gaps)))
    if np.mean(gains) > max(1.0, np.mean(spread)) and wins == len(args.gaps):
        print('')
        print('  ОЧИЩЕНИЕ ПОМОГАЕТ КАЛМАНУ, и порядок базисов держится.')
        print('  Ось новизны получает применение, не спорящее с трекером.')
    elif np.mean(gains) > 0:
        print('')
        print('  Прирост есть, но не больше разброса между базисами —')
        print('  то есть неотличим от выбора базиса наугад.')
    else:
        print('')
        print('  ОЧИЩЕНИЕ КАЛМАНУ НЕ ПОМОГАЕТ. Фильтр уже сглаживает сам, и')
        print('  вторичное сглаживание отнимает информацию. На 5 Гц линия')
        print('  закрыта: всё упирается в кадры 30 Гц.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
