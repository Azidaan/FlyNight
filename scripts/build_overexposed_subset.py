"""Поднабор переэкспонированных видео: есть ли он в BDD100K и как его выделить.

Зачем. Переэкспонирование стоит ПЕРВЫМ словом в теме диссертации, но в
метаданных BDD100K его нет: атрибуты только weather, timeofday, scene. Значит
поднабор надо выделять самим по кадрам — и это же означает, что туда мало кто
ходил.

Что меряем на каждом видео:
    доля насыщенных пикселей — сколько точек упёрлось в потолок яркости;
    доля очень тёмных — для контраста, чтобы отделить пересвет от недосвета;
    средняя яркость.

Насыщенная точка — это физически потерянная информация: текстуры там нет и
восстановить её нечем. Тем и отличается от ночи, где сигнал ослаблен, но есть.
Отсюда и гипотеза, что карта качества (меряющая насыщенность текстурой) на
пересвете осмысленна, в отличие от ночи.

Обучения и видеокарты не нужно, только чтение кадров.

Запуск: python scripts/build_overexposed_subset.py
        python scripts/build_overexposed_subset.py --frames 12 --top 40
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IMG_ROOT = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
OUT_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'overexposure_scan.csv'
OUT_JSON = PROJECT_ROOT / 'experiments' / 'results' / 'overexposed_videos.json'

SAT_LEVEL = 250      # яркость, выше которой точка считается выбитой (0..255)
DARK_LEVEL = 20


def video_stats(vid_dir: Path, n_frames: int) -> dict[str, float]:
    import cv2
    files = sorted(vid_dir.glob('*.jpg'))
    if not files:
        return {}
    step = max(1, len(files) // n_frames)
    sat, dark, mean = [], [], []
    for f in files[::step][:n_frames]:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sat.append(float((img >= SAT_LEVEL).mean()))
        dark.append(float((img <= DARK_LEVEL).mean()))
        mean.append(float(img.mean()) / 255.0)
    if not sat:
        return {}
    return {'доля выбитых': float(np.mean(sat)),
            'доля тёмных': float(np.mean(dark)),
            'яркость': float(np.mean(mean)),
            'пик выбитых': float(np.max(sat))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=int, default=10, help='кадров на видео')
    ap.add_argument('--top', type=int, default=30, help='сколько видео отобрать')
    args = ap.parse_args()

    if not IMG_ROOT.exists():
        raise SystemExit(f'нет каталога кадров: {IMG_ROOT}')
    vids = sorted(p for p in IMG_ROOT.iterdir() if p.is_dir())
    print(f'[INFO] видео в val: {len(vids)}, кадров на видео: {args.frames}',
          flush=True)

    from datasets.subset_filter import load_video_attributes
    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json)) if det_json.exists() else {}

    rows = []
    for i, v in enumerate(vids, 1):
        st = video_stats(v, args.frames)
        if not st:
            continue
        a = attrs.get(v.name, {})
        st['время суток'] = a.get('timeofday', '?')
        st['погода'] = a.get('weather', '?')
        rows.append((v.name, st))
        if i % 40 == 0:
            print(f'[INFO] {i}/{len(vids)}', flush=True)

    rows.sort(key=lambda r: -r[1]['доля выбитых'])

    print('\n=== верх списка по доле выбитых точек ===')
    print('%-24s %12s %11s %10s %10s %s' % (
        'видео', 'выбитых', 'пик', 'тёмных', 'яркость', 'время суток'))
    for name, st in rows[:15]:
        print('%-24s %12.5f %11.5f %10.5f %10.4f %s' % (
            name[:24], st['доля выбитых'], st['пик выбитых'],
            st['доля тёмных'], st['яркость'], st['время суток']))

    sat_all = np.array([st['доля выбитых'] for _, st in rows])
    print('\n=== распределение доли выбитых по всем видео ===')
    for q in (50, 75, 90, 95, 99):
        print('  %2d-й процентиль: %.5f' % (q, float(np.percentile(sat_all, q))))
    print('  максимум:          %.5f' % float(sat_all.max()))

    by_tod: dict[str, list[float]] = {}
    for _, st in rows:
        by_tod.setdefault(st['время суток'], []).append(st['доля выбитых'])
    print('\n=== доля выбитых по времени суток ===')
    for tod, vals in sorted(by_tod.items(), key=lambda x: -np.mean(x[1])):
        print('  %-12s видео %3d   средняя доля выбитых %.5f' % (
            tod, len(vals), float(np.mean(vals))))

    chosen = [name for name, _ in rows[:args.top]]
    thr = float(np.percentile(sat_all, 100 - 100 * args.top / max(len(rows), 1)))
    print('\n=== приговор ===')
    print(f'  порог отбора (доля выбитых): {thr:.5f}')
    print(f'  отобрано видео: {len(chosen)}')
    if sat_all.max() < 0.005:
        print('\n  ПЕРЕСВЕТА В НАБОРЕ ПРАКТИЧЕСКИ НЕТ. Даже у худшего видео выбито')
        print('  меньше половины процента точек — поднабор придётся синтезировать.')
    else:
        print('\n  Пересвет в наборе ЕСТЬ, поднабор выделяется. Дальше — считать на')
        print('  нём метрики и карту качества отдельным гейтом.')

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['видео', 'доля выбитых', 'пик выбитых', 'доля тёмных',
                    'яркость', 'время суток', 'погода'])
        for name, st in rows:
            w.writerow([name, round(st['доля выбитых'], 6),
                        round(st['пик выбитых'], 6), round(st['доля тёмных'], 6),
                        round(st['яркость'], 5), st['время суток'], st['погода']])
    OUT_JSON.write_text(json.dumps({'threshold': thr, 'videos': chosen},
                                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[OK ] {OUT_CSV}')
    print(f'[OK ] {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
