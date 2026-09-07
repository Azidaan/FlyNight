"""Контроль к гейту Б: чем на самом деле объясняется победа db3.

Возражение, которое проверяется. В гейте Б лучшим оказался db3, и это было
прочитано как «данные говорят, что эффективный порядок движения второй». Вывод
не следует из опыта по трём причинам, и каждая проверяется отдельно.

    1. Число нулевых моментов меняется ОДНОВРЕМЕННО с длиной фильтра (db2 -> 4
       отсчёта, db3 -> 6, db4 -> 8). Значит наблюдалась смесь двух факторов.
       Разделяем: берём базисы, где длина совпадает при разном числе моментов
       (db3 и coif1 — оба 6 отсчётов, моментов 3 и 2), и где число моментов
       совпадает при разной длине (db2 и coif1 — оба 2 момента, 4 и 6).

    2. Перед разложением снимается ЛИНЕЙНЫЙ тренд, то есть та самая полиномиальная
       часть, обнулять которую и есть отличие db2 от Хаара. Проверяем два
       дополнительных режима снятия тренда — без него и квадратичный. Прямое
       предсказание из утверждения «порядок движения второй»: при квадратичном
       снятии тренда db2 должен догнать db3, потому что квадратичная часть уже
       снята и обнулять её базису больше не нужно. Если не догонит, объяснение
       через порядок движения неверно.

    3. Никакого не-вейвлетного контроля не было. Очищение = подавление верхней
       полосы, то есть просто сглаживание. Добавляем сглаживания той же силы
       без всякой вейвлетной конструкции: скользящее среднее и Савицкий-Голей.
       Если они дают тот же выигрыш, то измерена сила сглаживания, а не
       соответствие базиса порядку модели движения.

Остальное — как в гейте Б: эталонные траектории 5 Гц, окно 12 кадров, разрывы
2/3/5, краевой режим symmetric, метрика — медиана ошибки положения центра.

Запуск: .venv/Scripts/python.exe scripts/wavelet_basis_control.py
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.wavelet_gate_b import TRACKS, kalman_cv, load_runs  # noqa: E402

OUT = PROJECT_ROOT / 'experiments' / 'results' / 'wavelet_basis_control.csv'

# базис -> (число нулевых моментов, длина фильтра разложения)
WAVELETS = ['haar', 'db2', 'db3', 'db4', 'db5',
            'sym3', 'sym4', 'coif1', 'coif2']
DETRENDS = {'линейный': 1, 'без тренда': None, 'квадратичный': 2}


def wavelet_clean(sig: np.ndarray, wname: str, deg: int | None) -> np.ndarray:
    """Снять тренд степени deg, обнулить мелкие детали, вернуть тренд."""
    n = len(sig)
    t = np.arange(n, dtype=np.float64)
    if deg is None:
        trend = np.zeros(n)
    else:
        trend = np.polyval(np.polyfit(t, sig, deg), t)
    resid = sig - trend
    w = pywt.Wavelet(wname)
    lvl = min(2, max(1, pywt.dwt_max_level(n, w.dec_len)))
    c = pywt.wavedec(resid, w, mode='symmetric', level=lvl)
    c[-1] = np.zeros_like(c[-1])
    return pywt.waverec(c, w, mode='symmetric')[:n] + trend


def moving_average(sig: np.ndarray, k: int) -> np.ndarray:
    """Скользящее среднее ширины k с краями по типу symmetric."""
    pad = k // 2
    ext = np.concatenate([sig[pad:0:-1], sig, sig[-2:-2 - pad:-1]])
    out = np.convolve(ext, np.ones(k) / k, mode='same')
    return out[pad:pad + len(sig)]


def build_cleaners() -> list[tuple[str, str, int | None, callable]]:
    """(имя, семейство, число нулевых моментов, функция очищения)."""
    out = [('без очищения', 'контроль', None, lambda s: s)]
    for name, deg in DETRENDS.items():
        for wn in WAVELETS:
            w = pywt.Wavelet(wn)
            label = f'{wn} [{name}]'
            out.append((label, 'вейвлет', w.vanishing_moments_psi,
                        lambda s, wn=wn, deg=deg: wavelet_clean(s, wn, deg)))
    for k in (2, 3, 4, 5):
        out.append((f'скольз. среднее {k}', 'контроль', None,
                    lambda s, k=k: moving_average(s, k)))
    for win in (5, 7, 9):
        for po in (1, 2):
            out.append((f'Савицкий-Голей {win}/{po}', 'контроль', None,
                        lambda s, win=win, po=po: savgol_filter(s, win, po,
                                                                mode='nearest')))
    return out


def filt_len(name: str) -> int | None:
    base = name.split(' [')[0]
    try:
        return pywt.Wavelet(base).dec_len
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--gaps', nargs='+', type=int, default=[2, 3, 5])
    ap.add_argument('--videos', type=int, default=50)
    args = ap.parse_args()

    W = args.window
    files = sorted(TRACKS.glob('*.json'))[:args.videos]
    need = W + max(args.gaps)
    runs = []
    for p in files:
        runs += [(p.stem, r) for r in load_runs(p, need + 1)]
    print(f'[INFO] видео {len(files)}, кусков траекторий {len(runs)}')

    cleaners = build_cleaners()
    print(f'[INFO] способов очищения {len(cleaners)}')
    err: dict[tuple, list] = defaultdict(list)

    for _, run in runs:
        n = len(run)
        for gap in args.gaps:
            for end in range(W, n - gap):
                win = run[end - W:end]
                target = run[end + gap - 1]
                for cname, _, _, fn in cleaners:
                    p = np.array([kalman_cv(fn(win[:, ax]), gap)
                                  for ax in (0, 1)])
                    err[(gap, cname)].append(float(np.hypot(*(p - target))))

    med = {k: float(np.median(v)) for k, v in err.items()}
    base = {g: med[(g, 'без очищения')] for g in args.gaps}

    rows = []
    for cname, fam, mom, _ in cleaners:
        for g in args.gaps:
            rows.append({'разрыв': g, 'очищение': cname, 'семейство': fam,
                         'нулевых_моментов': mom, 'длина_фильтра': filt_len(cname),
                         'медиана_ошибки': round(med[(g, cname)], 4),
                         'к_штатному_Калману_%':
                             round(100 * (base[g] - med[(g, cname)]) / base[g], 3),
                         'случаев': len(err[(g, cname)])})

    def gain(cname):     # средний по разрывам прирост к штатному Калману, %
        return float(np.mean([100 * (base[g] - med[(g, cname)]) / base[g]
                              for g in args.gaps]))

    print('\nокно %d кадров, разрывы %s, случаев на клетку %d'
          % (W, args.gaps, len(err[(args.gaps[0], 'без очищения')])))
    print('Прирост к штатному Калману, %% (среднее по разрывам)\n')
    print('%-26s%10s%8s%9s' % ('очищение', 'прирост%', 'моментов', 'длина'))
    for cname, fam, mom, _ in cleaners:
        print('%-26s%10.2f%8s%9s' % (cname, gain(cname), mom if mom else '-',
                                     filt_len(cname) or '-'))

    print('\n=== 1. разделение факторов: моменты против длины фильтра ===')
    for tag in ('линейный',):
        print(f'  снятие тренда: {tag}')
        pairs = [('равная длина 6, моменты 3 против 2', f'db3 [{tag}]',
                  f'coif1 [{tag}]'),
                 ('равная длина 6, моменты 3 против 3', f'db3 [{tag}]',
                  f'sym3 [{tag}]'),
                 ('равные моменты 2, длина 4 против 6', f'db2 [{tag}]',
                  f'coif1 [{tag}]'),
                 ('равные моменты 4, длина 8 против 12', f'db4 [{tag}]',
                  f'coif2 [{tag}]')]
        for label, a, b in pairs:
            print('    %-38s %s %.2f%%  против  %s %.2f%%'
                  % (label, a, gain(a), b, gain(b)))

    print('\n=== 2. предсказание «порядок движения второй» ===')
    print('  если так, при квадратичном снятии тренда db2 догоняет db3')
    for tag in DETRENDS:
        d2, d3 = gain(f'db2 [{tag}]'), gain(f'db3 [{tag}]')
        print('    тренд %-14s db2 %+.2f%%, db3 %+.2f%%, разница %+.2f%%'
              % (tag, d2, d3, d3 - d2))

    print('\n=== 3. не-вейвлетный контроль ===')
    best_w = max([c[0] for c in cleaners if c[1] == 'вейвлет'], key=gain)
    ctrls = [c[0] for c in cleaners if c[1] == 'контроль' and c[0] != 'без очищения']
    best_c = max(ctrls, key=gain)
    print('  лучший вейвлет:  %-24s %+.2f%%' % (best_w, gain(best_w)))
    print('  лучший контроль: %-24s %+.2f%%' % (best_c, gain(best_c)))
    if gain(best_c) >= gain(best_w) - 0.2:
        print('  Обычное сглаживание повторяет результат вейвлета: измерена')
        print('  сила сглаживания, а не соответствие базиса модели движения.')
    else:
        print('  Вейвлет обходит обычное сглаживание — конструкция что-то даёт.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
