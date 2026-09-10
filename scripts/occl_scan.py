"""Атрибуты разметки как измерительный инструмент: окклюзия, усечение, толпа.

Разметка BDD100K MOT несёт покадровые атрибуты `occluded` / `truncated` /
`crowd`, и проектом они до сих пор не использовались. Здесь меряется доля
найденных детектором объектов в разрезе этих атрибутов — там, где лежат резервы
того же порядка, что и «первое появление» (`birth_cost.py`), и крупнее.

Разделы:

  1. доля найденных по каждому атрибуту, день против ночи;
  2. сочетания окклюзии и усечения — какая комбинация чего стоит;
  3. контроль к идее первого появления: возраст в УЗКИХ полосах площади
     (градация COCO слишком груба: «мелкие» это всё меньше 1024 px²);
  4. контроль: выживает ли эффект возраста среди неусечённых объектов;
  5. смерть трека — симметрия к рождению, для проверки её величины;
  6. пересечение мишеней «молодой» и «заслонён»: если молодые объекты в
     основном заслонены, два механизма взвешивания будут тянуть одни и те же
     объекты, и наслоение даст меньше суммы.

Молодым считается объект возраста 0 или 1 в своём треке; возраст берётся из
разметки, а не из трека детектора.

Запуск:
    python scripts/occl_scan.py
    python scripts/occl_scan.py --cache runs/det_cache_ctl_960/val --suffix _960
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
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'occl_scan.csv'

from scripts.birth_cost import STEPS, classify_frame  # noqa: E402

CATS = {'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7}
ATTRS = ['occluded', 'не occluded', 'truncated', 'не truncated',
         'crowd', 'не crowd']
BANDS = [(0, 256), (256, 512), (512, 1024), (1024, 2048),
         (2048, 4096), (4096, 9216), (9216, 10 ** 9)]
YOUNG_MAX_AGE = 1          # молодой = возраст 0 или 1
DYING_TAIL = 1             # уходящий = последние два кадра трека


def band_label(a: float) -> str:
    for lo, hi in BANDS:
        if lo <= a < hi:
            return f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
    return 'прочее'


BAND_LABELS = [f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+' for lo, hi in BANDS]


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def found_flags(gt_box, gt_cls, d_box, d_cls, d_conf, thr) -> np.ndarray:
    """НАЙДЕН: та же категория, перекрытие не ниже 0.5, уверенность от порога."""
    if len(gt_box) == 0 or len(d_box) == 0:
        return np.zeros(len(gt_box), bool)
    m = iou_mat(gt_box, d_box)
    same = gt_cls[:, None] == d_cls[None, :]
    return ((m >= 0.5) & same & (d_conf[None, :] >= thr)).any(1)


def pct(hit: int, tot: int) -> float:
    return round(100.0 * hit / tot, 2) if tot else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou-loc', type=float, default=0.3,
                    help='нижняя граница «рамка примерно на месте»')
    ap.add_argument('--videos', type=int, default=0, help='0 — все из кэша')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--suffix', default='')
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

    print(f'кэш: {CACHE}')
    print(f'порог уверенности: {args.conf}')
    print(f'молодой: возраст не выше {YOUNG_MAX_AGE}')
    print(f'видео в кэше: {len(files)}\n', flush=True)

    att = Counter()        # (условие, атрибут, найден) -> N
    combo = Counter()      # ((occ, trunc), найден) -> N
    age_band = Counter()   # (полоса, молодой, найден) -> N
    age_ntr = Counter()    # (условие, молодой, найден) -> N среди неусечённых
    death = Counter()      # (условие, уходящий, найден) -> N
    cross = Counter()      # (молодой, заслонён, найден) -> N
    births = Counter()     # (заслонён на первом кадре, найден) -> N
    stage_occ = Counter()  # (заслонён, ступень) -> N
    n_obj = n_videos = 0

    for f in files:
        cond = cond_of.get(f.stem)
        if cond is None:
            continue
        n_videos += 1
        data = json.loads((GT_JSON / f'{f.stem}.json').read_text(encoding='utf-8'))
        seen = defaultdict(list)
        for fr in data:
            for l in fr.get('labels') or []:
                if l.get('box2d') and l.get('category') in CATS:
                    seen[l['id']].append(fr['frameIndex'])
        first = {t: min(v) for t, v in seen.items()}
        last = {t: max(v) for t, v in seen.items()}

        d = np.load(f)
        names, fidx = d['frame_names'], d['frame_idx']
        xyxy, conf, cls = d['xyxy'], d['conf'], d['cls'].astype(np.int32)
        by_fi = {fr['frameIndex']: fr for fr in data}

        for fi in range(len(names)):
            fr = by_fi.get(fi)
            if fr is None:
                continue
            boxes, clss, occ, tru, crw, tids = [], [], [], [], [], []
            for l in fr.get('labels') or []:
                b, c = l.get('box2d'), l.get('category')
                if not b or c not in CATS:
                    continue
                boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
                clss.append(CATS[c])
                a = l.get('attributes') or {}
                occ.append(bool(a.get('occluded')))
                tru.append(bool(a.get('truncated')))
                crw.append(bool(a.get('crowd')))
                tids.append(l['id'])
            if not boxes:
                continue
            gb = np.array(boxes, np.float32)
            gc = np.array(clss, np.int32)
            sel = fidx == fi
            # полная лестница ступеней, а не только «найден/не найден»:
            # нужно знать, через какой механизм действует заслонение —
            # через недооценку уверенности или через настоящую слепоту
            step, _ = classify_frame(gb, gc, xyxy[sel], cls[sel], conf[sel],
                                     args.conf, args.iou_loc)
            ok = step == 0
            area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            n_obj += len(gb)

            for k, t in enumerate(tids):
                hit = bool(ok[k])
                young = fi - first[t] <= YOUNG_MAX_AGE
                dying = last[t] - fi <= DYING_TAIL
                att[(cond, 'occluded' if occ[k] else 'не occluded', hit)] += 1
                att[(cond, 'truncated' if tru[k] else 'не truncated', hit)] += 1
                att[(cond, 'crowd' if crw[k] else 'не crowd', hit)] += 1
                combo[((occ[k], tru[k]), hit)] += 1
                age_band[(band_label(float(area[k])), young, hit)] += 1
                cross[(young, occ[k], hit)] += 1
                death[(cond, dying, hit)] += 1
                stage_occ[(occ[k], STEPS[step[k]])] += 1
                if not tru[k]:
                    age_ntr[(cond, young, hit)] += 1
                if fi == first[t]:
                    births[(occ[k], hit)] += 1

        if n_videos % 10 == 0:
            print(f'[INFO] {n_videos}/{len(files)} видео', flush=True)

    rows: list[tuple] = []
    print(f'\nвидео обработано: {n_videos}')
    print(f'объектов учтено: {n_obj}\n')

    print('=== 1. Доля найденных по атрибутам разметки ===')
    print('%-16s %9s %8s %9s %8s' % ('атрибут', 'день N', 'день %',
                                     'ночь N', 'ночь %'))
    for a in ATTRS:
        cells = []
        for cond in ('normal', 'night'):
            tot = att[(cond, a, True)] + att[(cond, a, False)]
            p = pct(att[(cond, a, True)], tot)
            cells += [tot, p]
            rows.append(('атрибуты', cond, a, 'объектов', tot))
            rows.append(('атрибуты', cond, a, 'доля_найденных_%', p))
        print('%-16s %9d %7.2f%% %9d %7.2f%%' % (a, *cells))
    print('\nНочной штраф (день минус ночь), процентных пунктов:')
    for a in ATTRS:
        d_ = pct(att[('normal', a, True)],
                 att[('normal', a, True)] + att[('normal', a, False)])
        n_ = pct(att[('night', a, True)],
                 att[('night', a, True)] + att[('night', a, False)])
        rows.append(('ночной_штраф', '', a, 'штраф_пп', round(d_ - n_, 2)))
        print(f'  {a:16s} {d_ - n_:+6.2f}')
    print()

    print('=== 2. Сочетание окклюзии и усечения ===')
    print('%-20s %10s %10s' % ('категория', 'объектов', 'найдено %'))
    for (o, t), lab in [((False, False), 'чистый'),
                        ((True, False), 'только occluded'),
                        ((False, True), 'только truncated'),
                        ((True, True), 'оба')]:
        tot = combo[((o, t), True)] + combo[((o, t), False)]
        p = pct(combo[((o, t), True)], tot)
        rows.append(('сочетания', '', lab, 'объектов', tot))
        rows.append(('сочетания', '', lab, 'доля_найденных_%', p))
        print('%-20s %10d %9.2f%%' % (lab, tot, p))
    print()

    print('=== 3. Контроль к первому появлению: возраст в узких полосах площади ===')
    print('%-12s %9s %9s %9s %9s %9s' % ('полоса px2', 'молод N', 'молод %',
                                         'зрелых N', 'зрелых %', 'разрыв пп'))
    for lab in BAND_LABELS:
        cells, pcts = [], []
        for young in (True, False):
            tot = age_band[(lab, young, True)] + age_band[(lab, young, False)]
            p = pct(age_band[(lab, young, True)], tot)
            cells += [tot, p]
            pcts.append(p)
            tag = 'молодые' if young else 'зрелые'
            rows.append(('полосы_площади', '', f'{lab}|{tag}', 'объектов', tot))
            rows.append(('полосы_площади', '', f'{lab}|{tag}',
                         'доля_найденных_%', p))
        gap = round(pcts[1] - pcts[0], 2)
        rows.append(('полосы_площади', '', lab, 'разрыв_пп', gap))
        print('%-12s %9d %8.2f%% %9d %8.2f%% %9.2f' % (lab, *cells, gap))
    print()

    print('=== 4. Контроль: возраст среди НЕусечённых объектов ===')
    for cond in ('normal', 'night'):
        for young in (True, False):
            tot = age_ntr[(cond, young, True)] + age_ntr[(cond, young, False)]
            p = pct(age_ntr[(cond, young, True)], tot)
            tag = 'молодые' if young else 'зрелые'
            rows.append(('неусечённые', cond, tag, 'объектов', tot))
            rows.append(('неусечённые', cond, tag, 'доля_найденных_%', p))
            print(f'  {cond:7s} {tag:8s} {tot:6d}  {p:6.2f}%')
    print()

    print('=== 5. Смерть трека: последние два кадра против остальных ===')
    for cond in ('normal', 'night'):
        for dying in (True, False):
            tot = death[(cond, dying, True)] + death[(cond, dying, False)]
            p = pct(death[(cond, dying, True)], tot)
            tag = 'уходящие' if dying else 'остальные'
            rows.append(('смерть_трека', cond, tag, 'объектов', tot))
            rows.append(('смерть_трека', cond, tag, 'доля_найденных_%', p))
            print(f'  {cond:7s} {tag:9s} {tot:6d}  {p:6.2f}%')
    print()

    print('=== 6. Пересечение мишеней: молодой на заслонён ===')
    print('%-22s %10s %10s' % ('клетка', 'объектов', 'найдено %'))
    base = 0.0
    cell_tot: dict[tuple, int] = {}
    cell_pct: dict[tuple, float] = {}
    for young in (False, True):
        for o in (False, True):
            tot = cross[(young, o, True)] + cross[(young, o, False)]
            p = pct(cross[(young, o, True)], tot)
            cell_tot[(young, o)], cell_pct[(young, o)] = tot, p
            if not young and not o:
                base = p
            lab = ('молодой' if young else 'зрелый') + \
                  (', заслонён' if o else ', чистый')
            rows.append(('пересечение', '', lab, 'объектов', tot))
            rows.append(('пересечение', '', lab, 'доля_найденных_%', p))
            print('%-22s %10d %9.2f%%' % (lab, tot, p))

    print(f'\nРезерв каждой клетки до уровня «зрелый, чистый» ({base:.2f}%):')
    total_res = 0.0
    for (young, o), tot in cell_tot.items():
        if not young and not o:
            continue
        res = tot * (base - cell_pct[(young, o)]) / 100.0
        total_res += max(res, 0.0)
        lab = ('молодой' if young else 'зрелый') + \
              (', заслонён' if o else ', чистый')
        rows.append(('резерв', '', lab, 'объектов', round(res)))
        print(f'  {lab:22s} {res:8.0f}')
    rows.append(('резерв', '', 'суммарно', 'объектов', round(total_res)))
    print(f'  {"суммарно":22s} {total_res:8.0f}')

    n_young = sum(cell_tot[(True, o)] for o in (False, True))
    n_occ = sum(cell_tot[(y, True)] for y in (False, True))
    n_both = cell_tot[(True, True)]
    for k, v in (('молодых_всего', n_young), ('заслонённых_всего', n_occ),
                 ('молодых_и_заслонённых', n_both),
                 ('доля_заслонённых_среди_молодых_%', pct(n_both, n_young)),
                 ('доля_молодых_среди_заслонённых_%', pct(n_both, n_occ)),
                 ('доля_заслонённых_среди_всех_%', pct(n_occ, n_obj))):
        rows.append(('пересечение_сводка', '', '', k, v))
        print(f'  {k:36s} {v}')
    print()

    print('=== 7. Рождение трека: заслонён ли объект в свой первый кадр ===')
    for o in (False, True):
        tot = births[(o, True)] + births[(o, False)]
        p = pct(births[(o, True)], tot)
        lab = 'заслонён' if o else 'чистый'
        rows.append(('рождение_окклюзия', '', lab, 'треков', tot))
        rows.append(('рождение_окклюзия', '', lab,
                     'найден_на_первом_кадре_%', p))
        print(f'  {lab:10s} треков {tot:6d}  найден на первом кадре {p:6.2f}%')
    print()

    print('=== 8. Через какую ступень действует заслонение ===')
    print('Ключевой вопрос: заслонение отнимает объекты через НЕДООЦЕНКУ')
    print('уверенности или через настоящую СЛЕПОТУ. От ответа зависит,')
    print('подействует ли на них взвешивание при обучении: недооценку')
    print('поднять можно, отсутствующий отклик — нет.')
    print('%-12s %10s %8s %8s %8s %11s %8s'
          % ('категория', 'объектов', *STEPS))
    pct_by = {}
    for o, lab in ((False, 'чистые'), (True, 'заслонённые')):
        tot = sum(v for (oo, _), v in stage_occ.items() if oo == o)
        vals = []
        for st in STEPS:
            p = pct(stage_occ[(o, st)], tot)
            vals.append(p)
            rows.append(('ступени_окклюзия', '', lab, f'{st}_%', p))
        pct_by[o] = dict(zip(STEPS, vals))
        rows.append(('ступени_окклюзия', '', lab, 'объектов', tot))
        print('%-12s %10d %7.2f%% %7.2f%% %7.2f%% %10.2f%% %7.2f%%'
              % (lab, tot, *vals))

    if pct_by:
        gap = pct_by[False]['НАЙДЕН'] - pct_by[True]['НАЙДЕН']
        print(f'\nразрыв по найденным: {gap:.2f} пп. Из чего он складывается:')
        parts = []
        for st in STEPS[1:]:
            d = pct_by[True][st] - pct_by[False][st]
            share = 100.0 * d / gap if gap else 0.0
            parts.append((st, d, share))
            rows.append(('вклад_ступени_окклюзия', '', st, 'вклад_пп',
                         round(d, 2)))
            rows.append(('вклад_ступени_окклюзия', '', st, 'доля_разрыва_%',
                         round(share, 2)))
        for st, d, share in sorted(parts, key=lambda x: -x[1]):
            print('  %-12s %+7.2f пп  %6.2f%% разрыва' % (st, d, share))
        top = max(parts, key=lambda x: x[1])
        print(f'\nВЫВОД: заслонение действует прежде всего через «{top[0]}» '
              f'({top[2]:.0f}% разрыва).')
        if top[0] == 'ПОРОГ':
            print('  Это тот же механизм, что у возраста — недооценка. Значит')
            print('  окклюзия встаёт в ту же цепочку, и взвешивание применимо.')
        else:
            print('  Это НЕ механизм недооценки. Значит взвешивание при')
            print('  обучении на заслонённые объекты подействует слабо, и')
            print('  вариант Г требует другого метода, а не веса.')
    print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['таблица', 'условие', 'группа', 'метрика', 'значение'])
        w.writerows(rows)
    print(f'[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
