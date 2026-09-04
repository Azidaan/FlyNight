"""Занижена ли уверенность детектора ночью — или она честна.

Зачем. Лестница потерь показала: ночью 22.7% эталонных объектов найдены верно,
но получают балл ниже рабочего порога, и внутри каждого размера потери на
пороге ночью заметно выше дневных (средние 11.5% -> 21.0%). Отсюда соблазн
сказать «ночью уверенность занижена, надо снизить порог». Но у наблюдения два
разных объяснения, и лечение у них противоположное:

    ЧЕСТНАЯ уверенность. Ночные объекты действительно труднее, и низкий балл
        им положен: доля верных среди детекций с баллом 0.2 ночью такая же,
        как днём. Тогда снижение порога ночью тянет ровно столько же ложных,
        сколько днём, и никакого «бесплатного» выигрыша нет.

    ЗАНИЖЕННАЯ уверенность. Ночная детекция с баллом 0.2 оказывается верной
        ЧАЩЕ, чем дневная с тем же баллом. Тогда один и тот же порог
        выбрасывает ночью непропорционально много верных объектов, и порог
        обязан зависеть от условий. Это ошибка калибровки, и её исправляют.

Различает их кривая надёжности: детекции разбиваются по баллу на корзины, в
каждой считается фактическая доля верных. Сравниваются две кривые.

Три обязательных контроля, каждый из измеренного опыта.

1. РАЗМЕР. Ночью мелких вдвое меньше (26% против 41%), а мелкие сами по себе
   труднее и менее уверенны. Без этого контроля разница окажется составом
   сцен. Работа уже допускала эту ошибку в агрегированных метриках.

2. КЛАСС. Ночью может быть больше машин (лёгкий класс) и меньше пешеходов и
   велосипедистов (трудные). Разбиение по размеру этого не снимает.

3. СКУЧЕННОСТЬ КАДРА. Днём 11.9 объектов на кадр, ночью 9.0. В плотном кадре
   при том же балле выше шанс, что детекция окажется дубликатом или ложной.

Первый прогон (04.09) дал разрыв +0.08..+0.18 в рабочей полосе 0.1..0.5 при
контроле только по размеру. Оба оставшихся конфаундера правдоподобно толкают
в ту же сторону, поэтому решает ПОСЛОЙНАЯ сводка: разность день-ночь считается
внутри ячейки (класс x скученность x размер x корзина), где все три контроля
действуют одновременно, и лишь затем усредняется. Если разрыв переживает её —
он свойство условий, а не состава.

Разметка верности — жадное сопоставление ОДИН К ОДНОМУ внутри кадра: детекция
верна, если нашлась незанятая эталонная рамка того же класса с перекрытием не
ниже 0.5. Переиспользование запрещено, иначе доля верных будет завышена.

Запуск: python scripts/confidence_calibration.py
        python scripts/confidence_calibration.py --min-cell 50
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.miss_ladder import iou_mat, load_gt, size_class  # noqa: E402

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'confidence_calibration.csv'
IOU_TP = 0.5
SIZES = ['мелкие', 'средние', 'крупные']
CROWD_EDGES = [7, 12]                 # объектов в кадре: разреженный/обычный/плотный
CROWDS = ['разреженный', 'обычный', 'плотный']
BAND = (0.1, 0.5)                     # рабочая полоса, где стоит порог


def label_tp(d_box, d_cls, d_conf, gt_box, gt_cls) -> np.ndarray:
    """Верна ли детекция: жадно один к одному, по убыванию уверенности."""
    y = np.zeros(len(d_box), np.int8)
    if len(gt_box) == 0 or len(d_box) == 0:
        return y
    m = iou_mat(d_box, gt_box)
    same = d_cls[:, None] == gt_cls[None, :]
    m = np.where(same, m, 0.0)
    used = np.zeros(len(gt_box), bool)
    for i in np.argsort(-d_conf):
        row = np.where(used, -1.0, m[i])
        j = int(np.argmax(row))
        if row[j] >= IOU_TP:
            y[i] = 1
            used[j] = True
    return y


def _table(acc, key_fn, keys, edges, nbins, title, min_n):
    """Печать кривых надёжности по срезу; возвращает строки для CSV."""
    rows = []
    for k in keys:
        head = False
        for bi in range(nbins):
            a = {c: key_fn(acc, c, k, bi) for c in ('normal', 'night')}
            nd, nn = a['normal'][1], a['night'][1]
            if nd < min_n or nn < min_n:
                continue
            if not head:
                print(f'=== {title}: {k} ===')
                print('%-12s %9s %10s %9s %10s %9s' % (
                    'корзина', 'день N', 'день верн', 'ночь N', 'ночь верн', 'разница'))
                head = True
            pd_, pn = a['normal'][0] / nd, a['night'][0] / nn
            print('%.2f-%.2f %9d %9.3f %9d %9.3f %9.3f' % (
                edges[bi], edges[bi + 1], nd, pd_, nn, pn, pn - pd_))
            rows.append((title, str(k), round(edges[bi], 3), round(edges[bi + 1], 3),
                         int(nd), round(pd_, 4), int(nn), round(pn, 4),
                         round(pn - pd_, 4)))
        if head:
            print()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--min-cell', type=int, default=30,
                    help='минимум детекций в ячейке с каждой стороны')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from datasets.bdd100k_loader import CLASS_NAMES
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes

    files = sorted(CACHE.glob('*.npz'))
    if args.limit:
        files = files[:args.limit]
    ids = [f.stem for f in files]

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    cond_of: dict[str, str] = {}
    for cond in ('normal', 'night'):
        for v in filter_videos_by_subset(attrs, ids, cond):
            cond_of.setdefault(v, cond)

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    # полная ячейка: (условие, класс, скученность, размер, корзина) -> [верных, всего]
    cell: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(2, np.float64))

    for vi, f in enumerate(files, 1):
        cond = cond_of.get(f.stem)
        if cond is None:
            continue
        d = np.load(f)
        names, fidx = d['frame_names'], d['frame_idx']
        xyxy, cls, conf = d['xyxy'], d['cls'].astype(np.int32), d['conf']
        for fi in np.unique(fidx):
            gt_box, gt_cls = load_gt(f.stem, str(names[fi]))
            sel = fidx == fi
            db, dc, df = xyxy[sel], cls[sel], conf[sel]
            if len(db) == 0:
                continue
            y = label_tp(db, dc, df, gt_box, gt_cls)
            sz = size_class(db)
            cw = int(np.digitize(len(gt_box), CROWD_EDGES))   # 0/1/2
            b = np.clip(np.digitize(df, edges) - 1, 0, args.bins - 1)
            for yi, zi, bi, ci in zip(y, sz, b, dc):
                a = cell[(cond, int(ci), cw, int(zi), int(bi))]
                a[0] += yi
                a[1] += 1
        if vi % 10 == 0:
            print(f'[INFO] {vi}/{len(files)} видео', flush=True)

    def marg(acc, cond, key, bi, axis):
        """Сумма по всем ячейкам с заданным значением одной оси."""
        out = np.zeros(2)
        for (c, cl, cw, sz, b), v in acc.items():
            if c != cond or b != bi:
                continue
            val = {'class': cl, 'crowd': cw, 'size': sz}[axis]
            if val == key:
                out += v
        return out

    print('\nКривая надёжности: доля верных среди детекций корзины.\n')
    rows = []
    rows += _table(cell, lambda a, c, k, b: marg(a, c, k, b, 'size'),
                   [0, 1, 2], edges, args.bins, 'размер', args.min_cell)
    rows += _table(cell, lambda a, c, k, b: marg(a, c, k, b, 'crowd'),
                   [0, 1, 2], edges, args.bins, 'скученность', args.min_cell)
    cls_present = sorted({k[1] for k in cell})
    rows += _table(cell, lambda a, c, k, b: marg(a, c, k, b, 'class'),
                   cls_present, edges, args.bins, 'класс', args.min_cell)

    # ---- послойная сводка: все три контроля одновременно ----
    print('=== ПОСЛОЙНАЯ СВОДКА: класс x скученность x размер x корзина ===')
    print('Разность считается ВНУТРИ ячейки, где все три контроля совпадают.\n')
    keys = {k[1:] for k in cell}
    band_d, band_w, all_d, all_w = [], [], [], []
    n_cells = 0
    for (cl, cw, sz, bi) in sorted(keys):
        a = cell[('normal', cl, cw, sz, bi)]
        b = cell[('night', cl, cw, sz, bi)]
        if a[1] < args.min_cell or b[1] < args.min_cell:
            continue
        diff = b[0] / b[1] - a[0] / a[1]
        w = min(a[1], b[1])
        n_cells += 1
        all_d.append(diff); all_w.append(w)
        if BAND[0] <= edges[bi] < BAND[1]:
            band_d.append(diff); band_w.append(w)

    def summarize(ds, ws, name):
        if not ds:
            print(f'  {name}: ячеек нет')
            return None
        ds, ws = np.array(ds), np.array(ws)
        m = float(np.average(ds, weights=ws))
        pos = float((ds > 0).mean())
        print(f'  {name}: ячеек {len(ds)}, средняя разность '
              f'{m:+.4f}, доля ячеек с плюсом {pos:.2f}')
        return m

    print(f'  всего сопоставимых ячеек: {n_cells}')
    m_all = summarize(all_d, all_w, 'все корзины  ')
    m_band = summarize(band_d, band_w, f'полоса {BAND[0]}-{BAND[1]}')

    print('\n=== приговор ===')
    if m_band is None:
        print('  Данных в рабочей полосе не хватило — контроль не состоялся.')
    elif m_band > 0.02:
        print(f'  Разрыв ПЕРЕЖИЛ все три контроля: {m_band:+.4f} в рабочей полосе.')
        print('  Уверенность ночью занижена как свойство УСЛОВИЙ, а не состава.')
        print('  Порог, зависящий от условий, получает измеренное основание.')
    elif m_band > 0:
        print(f'  Разрыв съёжился до {m_band:+.4f} — знак прежний, величина мала.')
        print('  Большая часть исходного эффекта была составом сцен.')
    else:
        print(f'  Разрыв ИСЧЕЗ или сменил знак: {m_band:+.4f}.')
        print('  Исходный эффект был составом сцен. Направление закрыто.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['срез', 'значение', 'корзина_от', 'корзина_до', 'день_N',
                    'день_доля', 'ночь_N', 'ночь_доля', 'разница'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
