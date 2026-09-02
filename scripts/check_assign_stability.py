"""Насколько скачет набор детекций между соседними кадрами — день против ночи.

Зачем. YOLO26 работает без подавления дубликатов: какой отклик оставить на
объект, решает сама сеть назначением «один к одному». В обычном детекторе это
делает NMS — жёсткий алгоритм, одинаковый в любую погоду. Отсюда вопрос,
которого на детекторах с NMS не существует: не начинает ли назначение
колебаться, когда кадр плохой?

Если да, набор детекций прыгает от кадра к кадру, и это прямо даёт перебросы
личности в трекере. Наш блок дал ровно этот эффект — меньше ложных (5535
против 5996-6417) и меньше перебросов (4639 против 4712-4807) при неизменных
пропусках, — но объяснён он был иначе.

Как меряем. Для каждой пары соседних кадров одного видео сопоставляем детекции
по перекрытию рамок (жадно, порог IoU 0.5) и считаем долю несопоставленных.
Кадры видео идут подряд и почти не отличаются, поэтому высокая доля
несопоставленных — это нестабильность назначения, а не движение объектов.

Данные: готовые кэши детекций, обучения и видеокарты не требуется.

Запуск: python scripts/check_assign_stability.py
        python scripts/check_assign_stability.py --conf 0.25
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q' / 'val'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'assign_stability.csv'


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Попарные перекрытия двух наборов рамок xyxy."""
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


def unmatched_fraction(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    """Доля детекций кадра, которым не нашлось пары в соседнем кадре."""
    if len(a) == 0 and len(b) == 0:
        return float('nan')
    m = iou_matrix(a, b)
    matched_a, matched_b = set(), set()
    if m.size:
        order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
        for i, j in order:
            if m[i, j] < thr:
                break
            if i not in matched_a and j not in matched_b:
                matched_a.add(int(i))
                matched_b.add(int(j))
    total = len(a) + len(b)
    return (total - 2 * len(matched_a)) / total if total else float('nan')


def video_stats(npz: Path, conf_min: float, iou_thr: float) -> dict[str, float]:
    d = np.load(npz)
    conf, fidx, xyxy = d['conf'], d['frame_idx'], d['xyxy']
    keep = conf >= conf_min
    conf, fidx, xyxy = conf[keep], fidx[keep], xyxy[keep]
    frames = sorted(set(int(x) for x in fidx))
    per_frame = {f: xyxy[fidx == f] for f in frames}
    counts = np.array([len(per_frame[f]) for f in frames], dtype=np.float32)
    un = [unmatched_fraction(per_frame[frames[k]], per_frame[frames[k + 1]], iou_thr)
          for k in range(len(frames) - 1)]
    un = np.array([u for u in un if np.isfinite(u)], dtype=np.float32)
    # скачки числа детекций между соседними кадрами — вторая, независимая мера
    jump = np.abs(np.diff(counts)) / np.maximum(counts[:-1], 1)
    return {
        'кадров': float(len(frames)),
        'детекций на кадр': float(counts.mean()) if len(counts) else float('nan'),
        'не сопоставлено': float(un.mean()) if len(un) else float('nan'),
        'скачок числа': float(jump.mean()) if len(jump) else float('nan'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='нижний порог уверенности детекций')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0, help='ограничить число видео')
    ap.add_argument('--cache', default=str(Q_CACHE),
                    help='каталог кэша детекций; по умолчанию q-кэш ночного '
                         'поднабора, построенный МОДЕЛЬЮ С БЛОКОМ — для честного '
                         'сравнения условий нужен кэш контрольной модели')
    args = ap.parse_args()
    cache_dir = Path(args.cache)

    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes

    files = sorted(cache_dir.glob('*.npz'))
    if not files:
        raise SystemExit(f'кэш пуст: {cache_dir}')
    have = {f.stem: f for f in files}

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(have)
    by_cond: dict[str, list[Path]] = {}
    for cond in ('normal', 'night', 'rain', 'fog'):
        vids = filter_videos_by_subset(attrs, all_ids, cond)
        sel = [have[v] for v in vids if v in have]
        if args.limit:
            sel = sel[:args.limit]
        if sel:
            by_cond[cond] = sel

    if not by_cond:
        raise SystemExit('ни одно видео кэша не попало в поднаборы')

    print(f'порог уверенности {args.conf}, порог перекрытия {args.iou}\n')
    print('%-10s %7s %10s %16s %14s' % (
        'условие', 'видео', 'кадров', 'не сопоставлено', 'скачок числа'))
    rows = []
    for cond, sel in by_cond.items():
        stats = [video_stats(p, args.conf, args.iou) for p in sel]
        agg = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
        print('%-10s %7d %10.0f %16.4f %14.4f' % (
            cond, len(sel), agg['кадров'], agg['не сопоставлено'], agg['скачок числа']))
        for k, v in agg.items():
            rows.append((cond, len(sel), k, round(v, 5)))

    print('\n=== приговор ===')
    if 'normal' in by_cond and 'night' in by_cond:
        d = float(np.nanmean([video_stats(p, args.conf, args.iou)['не сопоставлено']
                              for p in by_cond['normal']]))
        n = float(np.nanmean([video_stats(p, args.conf, args.iou)['не сопоставлено']
                              for p in by_cond['night']]))
        print(f'  день  не сопоставлено: {d:.4f}')
        print(f'  ночь  не сопоставлено: {n:.4f}   отношение {n / d:.3f}')
        if n > d * 1.15:
            print('\n  НАЗНАЧЕНИЕ НЕСТАБИЛЬНЕЕ НОЧЬЮ. Набор детекций прыгает сильнее,')
            print('  и это прямой путь к перебросам личности. Направление живое.')
        else:
            print('\n  Разницы нет. Назначение ночью не менее стабильно, чем днём —')
            print('  гипотеза о нестабильности назначения не подтверждается.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['условие', 'видео', 'метрика', 'значение'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
