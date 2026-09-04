"""Из чего состоят пропуски: лестница потерь по ступеням.

Зачем. Крупнейшее число работы — 47% ненайденных объектов ночью (пропуски
трекинга 60 003 из 127 687). Причина ни разу не была названа: разрешение
объяснило только +0.025 при удвоении, а дальше стояло общее «не находит».
Но «не находит» — это не одна причина, а цепочка, и объект может выпасть на
любой ступени:

    СЛЕПОТА      детектор не выдал рамку даже с уверенностью 0.01
    ЛОКАЛИЗАЦИЯ  рамка есть и на месте, но перекрытие 0.3..0.5 — засчитано в
                 пропуск, хотя объект найден
    КЛАСС        рамка на месте с перекрытием >= 0.5, но класс другой
    ПОРОГ        всё верно, но уверенность ниже рабочего порога
    НАЙДЕН       дошёл до выхода детектора

Лечится каждая ступень по-разному, и ни одно лечение не помогает остальным.
Частотный блок бьёт по слепоте, голова регрессии — по локализации, порог —
по порогу. Пока ступени не разделены, любая стройка идёт вслепую.

Почему замер возможен без видеокарты. Кэш детекций собран с нижней границей
уверенности 0.01, и 80% детекций в нём лежат ниже рабочего порога 0.25.
То есть хвост, который обычно отбрасывается, сохранён — и «не найден» от
«отсечён порогом» отличить можно. Кадры BDD100K все 1280x720, поэтому
изображения читать не нужно вовсе: разметка переводится в пиксели константами.

Сопоставление намеренно оптимистичное. Для каждой эталонной рамки берётся
ЛУЧШАЯ детекция кадра, без запрета переиспользования. Это завышает
найденность, а значит занижает пропуски — и полученные потери тем самым
являются нижней оценкой. Для диагностики это верная сторона запаса: вопрос
не «сколько детектор набрал», а «видел ли он объект вообще».

Запуск: python scripts/miss_ladder.py
        python scripts/miss_ladder.py --conf 0.25 --iou-loc 0.3
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl' / 'val'
GT_ROOT = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'track' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'miss_ladder.csv'

W, H = 1280, 720                  # BDD100K: размер кадра постоянный
STEPS = ['НАЙДЕН', 'ПОРОГ', 'КЛАСС', 'ЛОКАЛИЗАЦИЯ', 'СЛЕПОТА']
SIZES = ['мелкие', 'средние', 'крупные']


def load_gt(video: str, frame_name: str):
    """Эталон кадра в пикселях: (N,4) xyxy и (N,) классы."""
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
    """Попарные перекрытия: строки — эталон, столбцы — детекции."""
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


def size_class(box: np.ndarray) -> np.ndarray:
    """Пороги COCO по площади рамки."""
    a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return np.where(a < 32 ** 2, 0, np.where(a < 96 ** 2, 1, 2)).astype(np.int32)


def classify_frame(gt_box, gt_cls, d_box, d_cls, d_conf, conf_thr, iou_loc):
    """Ступень выпадения для каждой эталонной рамки кадра.

    Порядок проверок задаёт смысл: сперва полностью верная детекция, затем
    верное место с чужим классом, затем верное место со слабым перекрытием.
    Объект, у которого рядом есть рамка с перекрытием >= 0.5 чужого класса,
    считается потерянным КЛАССОМ, а не локализацией: место найдено верно.
    """
    n = len(gt_box)
    out = np.full(n, 4, np.int32)                 # по умолчанию СЛЕПОТА
    best_conf = np.zeros(n, np.float32)
    if n == 0 or len(d_box) == 0:
        return out, best_conf

    m = iou_mat(gt_box, d_box)
    same = gt_cls[:, None] == d_cls[None, :]
    m_same = np.where(same, m, 0.0)

    hit_same = m_same.max(1) >= 0.5               # верный класс, верное место
    hit_any = m.max(1) >= 0.5                     # верное место, класс любой
    near = m.max(1) >= iou_loc                    # место примерно верное

    # уверенность лучшей верной детекции — по ней отделяем порог от находки
    conf_grid = np.where(m_same >= 0.5, d_conf[None, :], -1.0)
    best_conf = conf_grid.max(1)

    out[near] = 3                                 # ЛОКАЛИЗАЦИЯ
    out[hit_any] = 2                              # КЛАСС
    out[hit_same] = 1                             # ПОРОГ
    out[hit_same & (best_conf >= conf_thr)] = 0   # НАЙДЕН
    return out, best_conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='рабочий порог уверенности, ниже которого детекция '
                         'не доходит до трекера')
    ap.add_argument('--iou-loc', type=float, default=0.3,
                    help='нижняя граница «рамка примерно на месте»')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes

    files = sorted(CACHE.glob('*.npz'))
    if not files:
        raise SystemExit(f'кэш пуст: {CACHE} — запустите scripts/build_day_cache.py')
    if args.limit:
        files = files[:args.limit]
    have = {f.stem: f for f in files}

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(have)
    cond_of: dict[str, str] = {}
    for cond in ('normal', 'night'):
        for v in filter_videos_by_subset(attrs, all_ids, cond):
            cond_of.setdefault(v, cond)

    # счётчики: (условие, размер, ступень) -> объектов
    tally: Counter = Counter()
    # уверенности объектов, потерянных порогом — сколько вернёт его снижение
    thr_conf: dict[str, list] = {'normal': [], 'night': []}
    n_frames = 0

    for vi, f in enumerate(files, 1):
        cond = cond_of.get(f.stem)
        if cond is None:
            continue
        d = np.load(f)
        names, fidx = d['frame_names'], d['frame_idx']
        xyxy, conf, cls = d['xyxy'], d['conf'], d['cls'].astype(np.int32)
        for fi in np.unique(fidx):
            gt_box, gt_cls = load_gt(f.stem, str(names[fi]))
            if len(gt_box) == 0:
                continue
            sel = fidx == fi
            step, bconf = classify_frame(gt_box, gt_cls, xyxy[sel], cls[sel],
                                         conf[sel], args.conf, args.iou_loc)
            sz = size_class(gt_box)
            for s, z in zip(step, sz):
                tally[(cond, SIZES[z], STEPS[s])] += 1
            thr_conf[cond] += list(bconf[step == 1])
            n_frames += 1
        if vi % 10 == 0:
            print(f'[INFO] {vi}/{len(files)} видео, кадров {n_frames}', flush=True)

    print(f'\nкадров обработано: {n_frames}')
    print(f'рабочий порог: {args.conf}, граница локализации: {args.iou_loc}\n')

    rows = []
    for cond in ('normal', 'night'):
        tot_c = sum(v for (c, _, _), v in tally.items() if c == cond)
        if not tot_c:
            continue
        print(f'=== {"день" if cond == "normal" else "ночь"}: '
              f'эталонных объектов {tot_c} ===')
        print('%-10s %9s %8s %8s %8s %8s %8s' % (
            'размер', 'объектов', *STEPS))
        for sz in SIZES + ['ВСЕ']:
            if sz == 'ВСЕ':
                cnt = [sum(tally[(cond, s, st)] for s in SIZES) for st in STEPS]
            else:
                cnt = [tally[(cond, sz, st)] for st in STEPS]
            tot = sum(cnt)
            if not tot:
                continue
            print('%-10s %9d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%' % (
                sz, tot, *[100 * c / tot for c in cnt]))
            for st, c in zip(STEPS, cnt):
                rows.append((cond, sz, st, c, round(100 * c / tot, 2)))
        print()

    print('=== что даёт снижение порога ===')
    for cond in ('normal', 'night'):
        cc = np.array(thr_conf[cond], np.float32)
        if not len(cc):
            continue
        tot_c = sum(v for (c, _, _), v in tally.items() if c == cond)
        print(f'  {"день" if cond == "normal" else "ночь"}: потеряно порогом '
              f'{len(cc)} ({100 * len(cc) / tot_c:.1f}% всех объектов)')
        for t in (0.10, 0.05, 0.01):
            back = int((cc >= t).sum())
            print(f'      порог {t:.2f} вернул бы {back} '
                  f'(+{100 * back / tot_c:.1f} п.п. полноты)')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['условие', 'размер', 'ступень', 'объектов', 'доля_%'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
