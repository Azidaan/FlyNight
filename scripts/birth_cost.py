"""Цена первого появления: как детектор видит объект в начале трека и позже.

Зачем. Лестница потерь (`miss_ladder.py`) меряет, СКОЛЬКО объектов теряется и по
какой причине, но не меряет, ЧЕГО стоит потеря. Для трекинга потери неравноценны:
пропуск объекта в середине живущего трека закрывается предсказанием Калмана,
пропуск при первом появлении не закрывается ничем — трека ещё нет.

Гипотеза. Доля найденных на возрасте 0 (первый кадр, где объект присутствует в
разметке) заметно ниже, чем на возрасте 10 и старше.

Контроли, без которых вывод не читается:

  1. Размер. Объекты входят в сцену издалека и на возрасте 0 в среднем мельче.
     Поэтому решающий ответ берётся ВНУТРИ размерной группы, а не из общей
     строки. Узкие полосы площади считает `occl_scan.py`.
  2. Возраст считается по разметке, а не по треку детектора: это свойство
     сцены, а не конвейера.
  3. Пороги трекера. В стенде `track_high_thresh = new_track_thresh = 0.25`, а
     `track_low_thresh = 0.1`: продолжить трек можно детекцией от 0.1, родить —
     только от 0.25. Значит часть эффекта возраста создаёт политика трекера, а
     не сеть. Раздел 6 отделяет одно от другого: считает, сколько объектов
     ступени ПОРОГ попадает в полосу [0.1, 0.25) — их зрелый трек удержал бы
     второй волной ассоциации, а новый не родит.

Сравнение по порогу рождения ведётся при РАВНОЙ точности рождений, а не при
равном пороге: снижение `new_track_thresh` плодит ложные треки. Точка отсчёта —
точность рождений 87.64% (`qdet_admission.csv`).

Запуск:
    python scripts/birth_cost.py
    python scripts/birth_cost.py --cache runs/det_cache_ctl_960/val --suffix _960
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl' / 'val'
GT_JSON = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'birth_cost.csv'

STEPS = ['НАЙДЕН', 'ПОРОГ', 'КЛАСС', 'ЛОКАЛИЗАЦИЯ', 'СЛЕПОТА']
SIZES = ['мелкие', 'средние', 'крупные']
CATS = {'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7}
AGE_BINS = [(0, 0, 'возраст 0'), (1, 1, 'возраст 1'), (2, 2, 'возраст 2'),
            (3, 4, 'возраст 3-4'), (5, 9, 'возраст 5-9'),
            (10, 10 ** 9, 'возраст 10+')]
AGE_LABELS = [b[2] for b in AGE_BINS]


def age_label(age: int) -> str:
    for lo, hi, name in AGE_BINS:
        if lo <= age <= hi:
            return name
    return AGE_LABELS[-1]


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    """Ступень выпадения каждой эталонной рамки кадра.

    Порядок проверок тот же, что в `miss_ladder.py`: сперва верное место с
    чужим классом, затем верное место со слабым перекрытием, затем верная
    детекция ниже порога, затем найденная. Возвращает ступени и лучшую
    уверенность верной детекции (нужна для разбора полосы порогов).
    """
    n = len(gt_box)
    out = np.full(n, 4, np.int32)                  # по умолчанию СЛЕПОТА
    best_conf = np.zeros(n, np.float32)
    if n == 0 or len(d_box) == 0:
        return out, best_conf
    m = iou_mat(gt_box, d_box)
    same = gt_cls[:, None] == d_cls[None, :]
    m_same = np.where(same, m, -1.0)
    near = m.max(1) >= iou_loc
    hit_same = m_same.max(1) >= 0.5
    best_conf = np.where(m_same >= 0.5, d_conf[None, :], -1.0).max(1)
    out[near] = 3                                  # ЛОКАЛИЗАЦИЯ
    out[m.max(1) >= 0.5] = 2                       # КЛАСС
    out[hit_same] = 1                              # ПОРОГ
    out[hit_same & (best_conf >= conf_thr)] = 0    # НАЙДЕН
    return out, best_conf


def load_gt_tracks(vid: str):
    """(покадровый эталон, кадры каждого трека, возраст каждого появления)."""
    data = json.loads((GT_JSON / f'{vid}.json').read_text(encoding='utf-8'))
    per_frame, seen = {}, defaultdict(list)
    for fr in sorted(data, key=lambda x: x['frameIndex']):
        fi = fr['frameIndex']
        boxes, cls, tids = [], [], []
        for l in fr.get('labels') or []:
            b, c = l.get('box2d'), l.get('category')
            if not b or c not in CATS:
                continue
            boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
            cls.append(CATS[c])
            tids.append(l['id'])
        per_frame[fi] = (np.array(boxes, np.float32).reshape(-1, 4),
                         np.array(cls, np.int32), tids)
        for t in tids:
            seen[t].append(fi)
    age_of = {(t, fi): k
              for t, frames in seen.items()
              for k, fi in enumerate(sorted(frames))}
    return per_frame, seen, age_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.25,
                    help='порог рождения трека (new_track_thresh стенда)')
    ap.add_argument('--conf-low', type=float, default=0.1,
                    help='порог второй волны ассоциации (track_low_thresh)')
    ap.add_argument('--iou-loc', type=float, default=0.3,
                    help='нижняя граница «рамка примерно на месте»')
    ap.add_argument('--videos', type=int, default=0, help='0 — все из кэша')
    ap.add_argument('--cache', default=None,
                    help='каталог кэша детекций; стенд применения — 960')
    ap.add_argument('--suffix', default='', help='приписка к имени выходного файла')
    args = ap.parse_args()

    global CACHE, OUT
    if args.cache:
        CACHE = Path(args.cache)
    if args.suffix:
        OUT = OUT.with_name(OUT.stem + args.suffix + OUT.suffix)

    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes

    files = sorted(CACHE.glob('*.npz'))
    if not files:
        raise SystemExit(f'кэш пуст: {CACHE}')
    if args.videos:
        files = files[:args.videos]
    attrs = load_video_attributes(
        str(PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
            / 'bdd100k_labels_images_val.json'))
    cond_of: dict[str, str] = {}
    for cond in ('normal', 'night'):
        for v in filter_videos_by_subset(attrs, sorted(f.stem for f in files), cond):
            cond_of.setdefault(v, cond)

    # критерии печатаются ДО чисел
    print(f'кэш: {CACHE}')
    print(f'порог рождения трека: {args.conf}')
    print(f'порог второй волны ассоциации: {args.conf_low}')
    print(f'граница локализации: {args.iou_loc}')
    print(f'видео в кэше: {len(files)}\n', flush=True)

    by_age = Counter()        # (условие, возраст, ступень) -> объектов
    by_age_size = Counter()   # (возраст, размер, ступень) -> объектов
    band = Counter()          # (возраст, «в полосе»?) -> объектов ступени ПОРОГ
    birth_delay = defaultdict(list)
    never_found = Counter()
    n_tracks = Counter()
    n_frames = n_videos = 0

    for f in files:
        cond = cond_of.get(f.stem)
        if cond is None:
            continue
        n_videos += 1
        per_frame, seen, age_of = load_gt_tracks(f.stem)
        d = np.load(f)
        names, fidx = d['frame_names'], d['frame_idx']
        xyxy, conf, cls = d['xyxy'], d['conf'], d['cls'].astype(np.int32)
        found_ages = defaultdict(list)

        for fi in range(len(names)):
            gt = per_frame.get(fi)
            if gt is None or len(gt[0]) == 0:
                continue
            gt_box, gt_cls, tids = gt
            sel = fidx == fi
            step, bconf = classify_frame(gt_box, gt_cls, xyxy[sel], cls[sel],
                                         conf[sel], args.conf, args.iou_loc)
            sz = size_class(gt_box)
            for k, t in enumerate(tids):
                lab = age_label(age_of[(t, fi)])
                by_age[(cond, lab, STEPS[step[k]])] += 1
                by_age_size[(lab, SIZES[sz[k]], STEPS[step[k]])] += 1
                if step[k] == 1:                   # ПОРОГ: виден, но недооценён
                    band[(lab, bool(bconf[k] >= args.conf_low))] += 1
                if step[k] == 0:
                    found_ages[t].append(age_of[(t, fi)])
            n_frames += 1

        for t in seen:
            n_tracks[cond] += 1
            if found_ages[t]:
                birth_delay[cond].append(min(found_ages[t]))
            else:
                never_found[cond] += 1

        if n_videos % 10 == 0:
            print(f'[INFO] {n_videos}/{len(files)} видео, кадров {n_frames}',
                  flush=True)

    rows: list[tuple] = []
    tot_obj = sum(by_age.values())
    print(f'\nвидео обработано: {n_videos}')
    print(f'кадров обработано: {n_frames}')
    print(f'эталонных появлений объектов: {tot_obj}')
    print(f'эталонных треков: {sum(n_tracks.values())}\n')

    print('=== 1. Доля найденных по возрасту объекта в треке ===')
    print('%-14s %9s %8s %9s %8s' % ('возраст', 'день N', 'день %',
                                     'ночь N', 'ночь %'))
    for lab in AGE_LABELS:
        cells = []
        for cond in ('normal', 'night'):
            tot = sum(v for (c, a, _), v in by_age.items()
                      if c == cond and a == lab)
            hit = by_age[(cond, lab, 'НАЙДЕН')]
            pct = 100.0 * hit / tot if tot else 0.0
            cells += [tot, pct]
            rows.append(('возраст', cond, lab, '', 'объектов', tot))
            rows.append(('возраст', cond, lab, '', 'доля_найденных_%',
                         round(pct, 2)))
        print('%-14s %9d %7.2f%% %9d %7.2f%%' % (lab, *cells))
    print()

    print('=== 2. Контроль на размер: доля найденных внутри размерной группы ===')
    print('%-14s %11s %11s %11s' % ('возраст', 'мелкие %', 'средние %',
                                    'крупные %'))
    for lab in AGE_LABELS:
        pcts = []
        for s in SIZES:
            tot = sum(v for (a, z, _), v in by_age_size.items()
                      if a == lab and z == s)
            hit = by_age_size[(lab, s, 'НАЙДЕН')]
            pct = 100.0 * hit / tot if tot else 0.0
            pcts.append(pct)
            rows.append(('возраст_размер', '', lab, s, 'объектов', tot))
            rows.append(('возраст_размер', '', lab, s, 'доля_найденных_%',
                         round(pct, 2)))
        print('%-14s %10.2f%% %10.2f%% %10.2f%%' % (lab, *pcts))
    print()

    print('=== 3. Ступени выпадения по возрасту ===')
    print('%-14s %9s %8s %8s %8s %11s %8s' % ('возраст', 'объектов', *STEPS))
    for lab in AGE_LABELS:
        tot = sum(v for (a, _, _), v in by_age_size.items() if a == lab)
        pcts = []
        for st in STEPS:
            n = sum(v for (a, _, s), v in by_age_size.items()
                    if a == lab and s == st)
            pct = 100.0 * n / tot if tot else 0.0
            pcts.append(pct)
            rows.append(('ступени', '', lab, '', f'{st}_%', round(pct, 2)))
        print('%-14s %9d %7.2f%% %7.2f%% %7.2f%% %10.2f%% %7.2f%%'
              % (lab, tot, *pcts))
    print()

    print('=== 4. Задержка рождения трека ===')
    for cond in ('normal', 'night'):
        d = np.array(birth_delay[cond], np.float64)
        tot, nf = n_tracks[cond], never_found[cond]
        stats = {'треков': tot, 'ни_разу_не_найден': nf,
                 'доля_ни_разу_не_найден_%': round(100.0 * nf / tot, 2) if tot else 0.0}
        if d.size:
            stats.update({
                'задержка_медиана': float(np.median(d)),
                'задержка_среднее': round(float(d.mean()), 2),
                'рождений_без_задержки_%': round(100.0 * float((d == 0).mean()), 2),
                'задержка_3_и_более_%': round(100.0 * float((d >= 3).mean()), 2),
                'потеряно_кадров_на_рождениях': int(d.sum()),
            })
        for k, v in stats.items():
            rows.append(('рождение', cond, '', '', k, v))
            print(f'  {cond:7s} {k:32s} {v}')
    print()

    print('=== 5. Доля промахов, приходящаяся на молодые объекты ===')
    young = ('возраст 0', 'возраст 1')
    miss_all = sum(v for (_, _, s), v in by_age.items() if s != 'НАЙДЕН')
    miss_young = sum(v for (_, a, s), v in by_age.items()
                     if s != 'НАЙДЕН' and a in young)
    obj_young = sum(v for (_, a, _), v in by_age.items() if a in young)
    share_miss = 100.0 * miss_young / miss_all if miss_all else 0.0
    share_obj = 100.0 * obj_young / tot_obj if tot_obj else 0.0
    for k, v in (('промахов_всего', miss_all),
                 ('промахов_на_возрасте_0_1', miss_young),
                 ('доля_промахов_на_молодых_%', round(share_miss, 2)),
                 ('доля_молодых_среди_объектов_%', round(share_obj, 2)),
                 ('перепредставленность_раз',
                  round(share_miss / share_obj, 2) if share_obj else 0.0)):
        rows.append(('промахи', '', '', '', k, v))
        print(f'  {k:34s} {v}')
    print()

    print(f'=== 6. Полоса порогов [{args.conf_low}, {args.conf}) — вклад трекера ===')
    print('Объекты ступени ПОРОГ, чья верная детекция набрала не меньше порога')
    print('второй волны: зрелый трек их удержал бы, новый — не родит.')
    print('%-14s %10s %12s %10s' % ('возраст', 'ПОРОГ N', 'в полосе N', 'в полосе %'))
    for lab in AGE_LABELS:
        inb, out_ = band[(lab, True)], band[(lab, False)]
        tot = inb + out_
        pct = 100.0 * inb / tot if tot else 0.0
        rows.append(('полоса', '', lab, '', 'порог_объектов', tot))
        rows.append(('полоса', '', lab, '', 'в_полосе_объектов', inb))
        rows.append(('полоса', '', lab, '', 'в_полосе_%', round(pct, 2)))
        print('%-14s %10d %12d %9.2f%%' % (lab, tot, inb, pct))
    print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['таблица', 'условие', 'возраст', 'размер', 'метрика', 'значение'])
        w.writerows(rows)
    print(f'[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
