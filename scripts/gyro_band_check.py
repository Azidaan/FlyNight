"""Гейт В: видно ли дрожание камеры в полосе 3-8 Гц по инерциальным данным.

Зачем. Вся вейвлетная линия держится на предпосылке, что в движении камеры
есть отдельная высокочастотная полоса (дрожание подвески/дороги, 3-8 Гц),
которую вейвлет отделит от полезного движения. На разметке 5.05 Гц эта полоса
выше Найквиста и ненаблюдаема — поэтому гейт Б её не проверял и не мог. Но
BDD100K отдаёт инерциальные данные с частотой порядка 50 Гц для тех же самых
видео: по ним полоса 3-8 Гц наблюдаема с запасом.

Проверяются ровно два вопроса:

    1. Есть ли в спектре гироскопа ИЗБЫТОК мощности в полосе 3-8 Гц над
       собственным фоном спектра? Просто «мощность в полосе не ноль» ничего не
       значит: у любого спада 1/f мощность есть везде. Фон оценивается степенной
       моделью по log-log вне полосы, избыток считается относительно неё.
    2. Какова АМПЛИТУДА этого дрожания в пикселях изображения? Если полосное
       дрожание даёт доли пикселя, отделять его незачем, сколько бы кадров в
       секунду ни было: оно тонет в ошибке детектора (единицы пикселей).

Пиксели считаются из угловой скорости: смещение точки на матрице при повороте
камеры на малый угол равно f_px * dtheta, где f_px = W / (2 tan(FOV/2)).
Для BDD100K W = 1280, FOV по горизонтали принята 60 градусов (типовая для
регистратора), f_px ~ 1108 px/rad. Угол получается интегрированием полосно
отфильтрованной угловой скорости.

Запуск: .venv/Scripts/python.exe scripts/gyro_band_check.py
        ... --videos 200 --fov 60
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRACKS = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'val'
INFO = PROJECT_ROOT / 'data' / 'bdd100k' / 'info' / '100k'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'gyro_band_check.csv'

BAND = (3.0, 8.0)          # заявленная полоса дрожания
FIT_LO, FIT_HI = 0.3, 20.0  # диапазон оценки фонового спада спектра
IMG_W = 1280


def find_info(vid: str) -> Path | None:
    for split in ('val', 'train'):
        p = INFO / split / f'{vid}.json'
        if p.exists():
            return p
    return None


def gyro_series(path: Path) -> tuple[np.ndarray, float] | None:
    """(N,3) угловая скорость и частота дискретизации по медиане интервалов."""
    d = json.loads(path.read_text(encoding='utf-8'))
    g = d.get('gyro') or []
    if len(g) < 256:
        return None
    g = sorted(g, key=lambda s: s['timestamp'])
    ts = np.array([s['timestamp'] for s in g], np.float64) / 1000.0  # мс -> с
    xyz = np.array([[s.get('x', 0.0), s.get('y', 0.0), s.get('z', 0.0)]
                    for s in g], np.float64)
    dt = np.median(np.diff(ts))
    if not np.isfinite(dt) or dt <= 0:
        return None
    # неравномерность отсчётов: приводим к равномерной сетке линейно
    n = int((ts[-1] - ts[0]) / dt) + 1
    grid = ts[0] + np.arange(n) * dt
    res = np.stack([np.interp(grid, ts, xyz[:, k]) for k in range(3)], axis=1)
    return res, 1.0 / dt


def band_excess(f: np.ndarray, pxx: np.ndarray) -> tuple[float, float, float]:
    """Доля мощности в полосе, ожидание по фону и превышение (в разах).

    Фон: прямая в log-log по частотам диапазона [FIT_LO, FIT_HI] ИСКЛЮЧАЯ саму
    полосу. Если в полосе есть настоящий подъём, точки полосы лягут выше этой
    прямой; если спектр просто спадает, отношение будет около единицы.
    """
    ok = (f >= FIT_LO) & (f <= FIT_HI) & (pxx > 0)
    inb = ok & (f >= BAND[0]) & (f <= BAND[1])
    out = ok & ~inb
    if inb.sum() < 3 or out.sum() < 5:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(np.log10(f[out]), np.log10(pxx[out]), 1)
    pred = 10 ** (a * np.log10(f[inb]) + b)
    obs = pxx[inb]
    frac = float(np.trapz(pxx[inb], f[inb]) / np.trapz(pxx[ok], f[ok]))
    return frac, float(a), float(np.median(obs / pred))


def band_pixels(sig3: np.ndarray, fs: float, fov_deg: float) -> float:
    """СКО смещения изображения (пиксели) от полосно-ограниченного дрожания."""
    f_px = IMG_W / (2 * np.tan(np.radians(fov_deg) / 2))
    ny = fs / 2
    hi = min(BAND[1], ny * 0.95)
    if hi <= BAND[0]:
        return np.nan
    sos = signal.butter(4, [BAND[0] / ny, hi / ny], btype='band', output='sos')
    # горизонтальное смещение даёт рыскание, вертикальное — тангаж; берём
    # квадратичную сумму двух осей, наибольших по мощности
    var = sig3.var(axis=0)
    axes = np.argsort(var)[-2:]
    tot = 0.0
    for k in axes:
        w = signal.sosfiltfilt(sos, sig3[:, k])          # рад/с в полосе
        ang = np.cumsum(w) / fs                          # рад
        ang = ang - signal.savgol_filter(ang, min(len(ang) // 2 * 2 - 1, 101), 1)
        tot += float(np.var(ang * f_px))
    return float(np.sqrt(tot))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=200)
    ap.add_argument('--fov', type=float, default=60.0, help='горизонт. FOV, град')
    args = ap.parse_args()

    vids = [p.stem for p in sorted(TRACKS.glob('*.json'))][:args.videos]
    rows, fracs, ratios, pxs, fss = [], [], [], [], []
    missing = 0
    for vid in vids:
        p = find_info(vid)
        if p is None:
            missing += 1
            continue
        got = gyro_series(p)
        if got is None:
            missing += 1
            continue
        sig3, fs = got
        fss.append(fs)
        nper = int(min(len(sig3), max(256, fs * 8)))
        # спектр суммарной угловой скорости по трём осям
        pxx_sum = None
        for k in range(3):
            f, pxx = signal.welch(sig3[:, k], fs=fs, nperseg=nper)
            pxx_sum = pxx if pxx_sum is None else pxx_sum + pxx
        frac, slope, ratio = band_excess(f, pxx_sum)
        px = band_pixels(sig3, fs, args.fov)
        rows.append({'video': vid, 'fs_hz': round(fs, 3),
                     'доля_мощности_3_8': round(frac, 5),
                     'наклон_логлог': round(slope, 3),
                     'превышение_над_фоном': round(ratio, 4),
                     'ско_дрожания_px': round(px, 4)})
        fracs.append(frac)
        ratios.append(ratio)
        pxs.append(px)

    if not rows:
        raise SystemExit('нет инерциальных данных')

    fracs, ratios, pxs = map(np.array, (fracs, ratios, pxs))
    print(f'[INFO] видео с гироскопом {len(rows)}, без данных {missing}')
    print(f'частота дискретизации: медиана {np.median(fss):.2f} Гц, '
          f'мин {np.min(fss):.2f}, макс {np.max(fss):.2f} — '
          f'Найквист {np.median(fss) / 2:.2f} Гц')
    print(f'\nполоса {BAND[0]}-{BAND[1]} Гц, фон — степенной спад по '
          f'{FIT_LO}-{FIT_HI} Гц вне полосы\n')
    print('  доля мощности в полосе:   медиана %.4f  (%.4f..%.4f, 10-90%%)'
          % (np.nanmedian(fracs), *np.nanpercentile(fracs, [10, 90])))
    print('  наклон спектра log-log:   медиана %.2f' % np.nanmedian(
        [r['наклон_логлог'] for r in rows]))
    print('  ПРЕВЫШЕНИЕ над фоном:     медиана %.3f  (%.3f..%.3f, 10-90%%)'
          % (np.nanmedian(ratios), *np.nanpercentile(ratios, [10, 90])))
    print('  видео с превышением >1.5: %d из %d'
          % (int(np.nansum(ratios > 1.5)), len(ratios)))
    print('\n  амплитуда дрожания в кадре (СКО, пиксели при FOV %.0f):' % args.fov)
    print('     медиана %.3f  (%.3f..%.3f, 10-90%%)'
          % (np.nanmedian(pxs), *np.nanpercentile(pxs, [10, 90])))

    print('\n=== приговор гейта В ===')
    if np.nanmedian(ratios) > 1.5:
        print('  Подъём в полосе 3-8 Гц ЕСТЬ — предпосылка о дрожании стоит.')
    else:
        print('  Отдельного подъёма в полосе 3-8 Гц НЕТ: спектр в ней ложится')
        print('  на собственный фоновый спад. Отделять как отдельную')
        print('  компоненту нечего.')
    if np.nanmedian(pxs) < 1.0:
        print('  Амплитуда полосного дрожания меньше пикселя — она заведомо')
        print('  ниже ошибки детектора (единицы пикселей), и 30 кадров в')
        print('  секунду её не сделают полезной мишенью.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
