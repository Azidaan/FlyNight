"""Осмысленнее ли карта качества там, где текстура ВЫБИТА пересветом?

Гипотеза (записана до замера). Три характеристики LQA — дисперсия, пик, градиент
— меряют насыщенность текстурой, а не трудность. Ночью это бесполезно: текстура
ослаблена, но присутствует. В пересвеченных областях текстура уничтожена
физически, восстанавливать нечего, и там оценка честна.

Проверка на уже готовых данных, без единого прогона. У нас есть:
    q по каждой детекции ночного кэша (129 836 штук);
    доля выбитых точек по каждому видео (scripts/build_overexposed_subset.py).

Делим ночные видео на две половины по доле выбитых и сравниваем, насколько
сильнее связь q с уверенностью детектора в засвеченной половине.

Оговорка, обязательная к упоминанию. Ночной поднабор — самый ТЁМНЫЙ из всех
(средняя доля выбитых 0.0068 против 0.0185 днём), поэтому здесь проверяется
слабая версия гипотезы: работает ли разница ВНУТРИ ночи. Сильная версия требует
q-кэша на дневном пересвеченном поднаборе, и это отдельный прогон.

Связь считается ВНУТРИ уровня пирамиды: знак связи на уровнях разный, общий
коэффициент самоуничтожается (замер 9.5).

Запуск: python scripts/q_vs_overexposure.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q' / 'val'
SCAN = PROJECT_ROOT / 'experiments' / 'results' / 'overexposure_scan.csv'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'q_vs_overexposure.csv'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agg', default='min', choices=['mean', 'min', 'median'],
                    help='агрегация q по рамке; min даёт единый знак (замер 9.5)')
    ap.add_argument('--min-conf', type=float, default=0.1)
    args = ap.parse_args()

    if not SCAN.exists():
        raise SystemExit(f'нет скана пересвета: {SCAN} — запустите '
                         f'scripts/build_overexposed_subset.py')
    sat = {r['видео']: float(r['доля выбитых'])
           for r in csv.DictReader(open(SCAN, encoding='utf-8'))}

    files = sorted(Q_CACHE.glob('*.npz'))
    have = [(f, sat[f.stem]) for f in files if f.stem in sat]
    if len(have) < 10:
        raise SystemExit(f'слишком мало видео с обеими величинами: {len(have)}')
    have.sort(key=lambda x: x[1])
    half = len(have) // 2
    groups = {'мало выбитых': have[:half], 'много выбитых': have[half:]}

    key = f'q_{args.agg}'
    print(f'видео с q и сканом: {len(have)}   агрегация: {args.agg}')
    print('порог выбитых между половинами: %.5f\n' % have[half][1])

    rows = []
    print('%-16s %8s %10s %12s %12s %12s' % (
        'группа', 'страйд', 'детекций', 'q сред', 'q разброс', 'связь conf'))
    stats: dict[str, dict[int, float]] = {}
    for gname, sel in groups.items():
        q, cf, sd = [], [], []
        for f, _ in sel:
            d = np.load(f)
            if key not in d:
                continue
            m = d['conf'] >= args.min_conf
            q.append(d[key][m]); cf.append(d['conf'][m]); sd.append(d['stride'][m])
        q, cf, sd = np.concatenate(q), np.concatenate(cf), np.concatenate(sd)
        stats[gname] = {}
        for s in (8, 16, 32):
            k = sd == s
            if k.sum() < 200:
                continue
            r = float(np.corrcoef(q[k], cf[k])[0, 1])
            stats[gname][s] = r
            print('%-16s %8d %10d %12.4f %12.4f %12.3f' % (
                gname, s, k.sum(), q[k].mean(), q[k].std(), r))
            rows.append((gname, s, int(k.sum()), round(float(q[k].mean()), 5),
                         round(float(q[k].std()), 5), round(r, 4)))
        print()

    print('=== приговор ===')
    lo, hi = stats.get('мало выбитых', {}), stats.get('много выбитых', {})
    common = sorted(set(lo) & set(hi))
    if not common:
        print('  не с чем сравнивать')
    else:
        print('  %8s %14s %14s %10s' % ('страйд', 'мало выбитых', 'много выбитых',
                                        'по модулю'))
        gains = []
        for s in common:
            g = abs(hi[s]) - abs(lo[s])
            gains.append(g)
            print('  %8d %14.3f %14.3f %+10.3f' % (s, lo[s], hi[s], g))
        mean_gain = float(np.mean(gains))
        signs_hi = {np.sign(hi[s]) for s in common}
        print(f'\n  средний прирост связи по модулю: {mean_gain:+.3f}')
        print(f'  знак в засвеченной половине: '
              f'{"ЕДИНЫЙ" if len(signs_hi) == 1 else "по-прежнему РАЗНЫЙ"}')
        if mean_gain > 0.05 and len(signs_hi) == 1:
            print('\n  ГИПОТЕЗА ПОДДЕРЖАНА: где выбито больше, связь крепче и знак')
            print('  устойчив. Стоит строить отдельный гейт на дневном пересвете.')
        elif mean_gain > 0.05:
            print('\n  Связь крепче, но знак всё ещё скачет. Частичная поддержка:')
            print('  без починки по уровням пользоваться нельзя.')
        else:
            print('\n  ГИПОТЕЗА НЕ ПОДДЕРЖАНА на этих данных. Оговорка в силе: ночь —')
            print('  самый тёмный поднабор, сильная версия требует дневного кэша.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['группа', 'страйд', 'детекций', 'q сред', 'q разброс', 'связь'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
