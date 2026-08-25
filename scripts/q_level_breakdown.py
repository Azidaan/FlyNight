"""Разбор карты качества по уровням пирамиды (дополнение к подэтапу 9.5).

Зачем отдельно от correlation_experiment.py. Тот считает ОДНО число на
поднабор — Pearson по всем детекциям сразу. Выяснилось, что это число
вводит в заблуждение: связь q с уверенностью имеет РАЗНЫЙ ЗНАК на разных
уровнях пирамиды, и общая корреляция складывается из противоположных
слагаемых. Считать её осмысленно только внутри уровня.

Второе, что проверяется здесь, — пригодность q для формулы плана
    tau_i = tau_base - k * (1 - q_i).
Формула писалась в расчёте на q из диапазона [0, 1]. Если q фактически
занимает узкую полосу у единицы, поправка вырождается в общий сдвиг
порога, то есть режим 3 совпадает с контрольным режимом 3к — а план
требует, чтобы режим 3 контроль ОБОШЁЛ.

Использование:
    python scripts/q_level_breakdown.py
    python scripts/q_level_breakdown.py --agg min --min-conf 0.25
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q'
TAU_BASE = 0.5
K = 0.2


def load(split: str, agg: str) -> dict[str, np.ndarray]:
    """Все детекции q-кэша одним набором массивов."""
    files = sorted((Q_CACHE / split).glob('*.npz'))
    if not files:
        raise SystemExit(f'q-кэш пуст: {Q_CACHE / split}')
    cols: dict[str, list] = {k: [] for k in ('q', 'conf', 'stride', 'area')}
    for f in files:
        d = np.load(f)
        key = f'q_{agg}'
        if key not in d:
            raise SystemExit(f'{f.name}: нет {key} — модель без DRB?')
        xyxy = d['xyxy']
        cols['q'].append(d[key])
        cols['conf'].append(d['conf'])
        cols['stride'].append(d['stride'])
        cols['area'].append((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out['n_videos'] = len(files)
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float('nan')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='val')
    ap.add_argument('--agg', default='mean', choices=['mean', 'min', 'median'])
    ap.add_argument('--min-conf', type=float, default=0.1)
    args = ap.parse_args()

    d = load(args.split, args.agg)
    m = d['conf'] >= args.min_conf
    q, cf, sd = d['q'][m], d['conf'][m], d['stride'][m]
    n_vid = d['n_videos']

    print(f'\nвидео: {n_vid}, детекций после порога conf>={args.min_conf}: {len(q)}')
    print(f'агрегация q по рамке: {args.agg}\n')

    print('=== диапазон q ===')
    print(f'  сред {q.mean():.4f}  разброс {q.std():.4f}  '
          f'от {q.min():.4f} до {q.max():.4f}')
    print(f'  формула ждёт q из [0, 1], фактически занято '
          f'{(q.max() - q.min()) * 100:.1f}% диапазона')

    print('\n=== по уровням пирамиды ===')
    print(f'  {"страйд":<8}{"детекций":>10}{"q сред":>10}{"q разброс":>12}'
          f'{"связь с conf":>15}')
    rows = []
    for s in sorted(set(int(x) for x in sd)):
        k = sd == s
        if k.sum() < 20:
            continue
        r = corr(q[k], cf[k])
        rows.append((s, int(k.sum()), q[k].mean(), q[k].std(), r))
        print(f'  {s:<8}{k.sum():>10}{q[k].mean():>10.4f}{q[k].std():>12.4f}'
              f'{r:>+15.3f}')
    r_all = corr(q, cf)
    print(f'  {"ВСЕ":<8}{len(q):>10}{q.mean():>10.4f}{q.std():>12.4f}'
          f'{r_all:>+15.3f}')

    signs = {np.sign(r) for *_, r in rows}
    if len(signs) > 1:
        print('\n  ВНИМАНИЕ: знак связи МЕНЯЕТСЯ между уровнями.')
        print('  Общее число складывается из противоположных слагаемых —')
        print('  трактовать его как «связь q с уверенностью» нельзя.')

    print('\n=== что даст формула плана tau = %.1f - %.1f*(1-q) ===' % (TAU_BASE, K))
    tau = TAU_BASE - K * (1 - q)
    print(f'  порог: сред {tau.mean():.4f}  разброс {tau.std():.4f}  '
          f'от {tau.min():.4f} до {tau.max():.4f}')
    print(f'  весь размах порога: {(tau.max() - tau.min()):.4f}')
    print(f'  типичное отличие двух соседних детекций: {tau.std() * 1.41:.4f}')
    verdict = ('порог практически ОДИНАКОВ для всех — индивидуальной адаптации '
               'не возникает,\n  режим 3 совпадёт с контрольным режимом 3к'
               if tau.std() < 0.01 else 'разброс порога заметный')
    print(f'  ПРИГОВОР: {verdict}')

    print('\n=== если нормировать q ВНУТРИ уровня (ранг в [0,1]) ===')
    qn = np.zeros_like(q)
    for s in sorted(set(int(x) for x in sd)):
        k = sd == s
        if k.sum() < 2:
            qn[k] = 0.5
            continue
        order = np.argsort(np.argsort(q[k]))
        qn[k] = order / max(len(order) - 1, 1)
    tau_n = TAU_BASE - K * (1 - qn)
    print(f'  порог: сред {tau_n.mean():.4f}  разброс {tau_n.std():.4f}  '
          f'от {tau_n.min():.4f} до {tau_n.max():.4f}')
    print(f'  размах вырос в {(tau_n.max()-tau_n.min())/(tau.max()-tau.min()):.0f} раз')
    print(f'  связь нормированного q с уверенностью: {corr(qn, cf):+.3f}')

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f'q_level_breakdown_{args.agg}.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stride', 'n', 'q_mean', 'q_std', 'corr_with_conf'])
        for s, n, mu, sg, r in rows:
            w.writerow([s, n, f'{mu:.5f}', f'{sg:.5f}', f'{r:.4f}'])
        w.writerow(['all', len(q), f'{q.mean():.5f}', f'{q.std():.5f}',
                    f'{r_all:.4f}'])
    print(f'\n[OK ] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
