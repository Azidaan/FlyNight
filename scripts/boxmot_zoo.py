"""Трекеры из BoxMOT на нашем замороженном кэше детекций.

Продолжение гейта 0в. В ultralytics встроены шесть трекеров
(scripts/tracker_zoo.py), но в отчётах названы ещё несколько, которых там нет:
BoostTrack и BoostTrack++, OccluBoost, HybridSORT, StrongSORT, SFSORT. Они
живут в пакете BoxMOT, и здесь прогоняются на тех же детекциях YOLO26, тех же
50 видео и оцениваются теми же двумя конвейерами — иначе числа несравнимы.

BoostTrack интересен по существу дела: он поднимает балл слабых детекций,
согласующихся с уже существующими треками. Наша диагностика показала, что
ночью 22.8% объектов лежат в слабой полосе и трекер их конструктивно не
подбирает — то есть механизм BoostTrack целится ровно в измеренную мишень.

ReID нигде не включается: во всех трекерах передаётся with_reid=False, если
параметр поддерживается. Так сравнение остаётся честным по входу — внешность
не участвует ни у одного метода, включая наши.

Кадры читаются с диска: BoxMOT требует изображение даже когда компенсация
движения выключена (проверка входа), а BoostTrack по умолчанию использует
компенсацию движения камеры.

Запуск: .venv/Scripts/python.exe scripts/boxmot_zoo.py --videos 50
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from evaluation.hota_eval import evaluate_hota  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402

IMAGES = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
RUNS = PROJECT_ROOT / 'runs' / 'boxmot_zoo'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'boxmot_zoo.csv'
KEYS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSw', 'MOTP')

# имя -> (путь класса, дополнительные параметры)
# Фильтр формы рамки. BoostTrack и производные по умолчанию выбрасывают
# рамки шире 1.6 по отношению ширины к высоте — наследие пешеходных
# бенчмарков MOT17/MOT20, где объекты вытянуты по вертикали. На дорожной
# съёмке это отбрасывает 21.6% эталонных объектов: 27.7% машин, 28.8%
# автобусов, 25.9% грузовиков (замер по 20 видео val). Поэтому фильтр
# отключается: иначе сравниваются не методы связывания, а разные наборы
# объектов, и BoostTrack проигрывает не механизмом, а настройкой не под тот
# домен. Прогон с включённым фильтром сохранён как boosttrack_wide_filter.
NO_SHAPE_FILTER = {'aspect_ratio_thresh': 1e6, 'min_box_area': 0}

ZOO = {
    'boosttrack': ('boxmot.trackers.bbox.boosttrack.BoostTrack',
                   {'with_reid': False, **NO_SHAPE_FILTER}),
    'boosttrack_pp': ('boxmot.trackers.bbox.boosttrack.BoostTrack',
                      {'with_reid': False, 'use_sb': True, 'use_vt': True,
                       'use_rich_s': True, 's_sim_corr': True,
                       **NO_SHAPE_FILTER}),
    'occluboost': ('boxmot.trackers.bbox.occluboost.OccluBoost',
                   {'with_reid': False, **NO_SHAPE_FILTER}),
    'hybridsort': ('boxmot.trackers.bbox.hybridsort.HybridSort',
                   {'with_reid': False}),
    'sfsort': ('boxmot.trackers.bbox.sfsort.SFSORT', {}),
}


def make_tracker(name: str, conf: float):
    """Создать трекер BoxMOT, подставив только поддерживаемые им параметры."""
    import importlib
    import inspect

    path, extra = ZOO[name]
    module, cls_name = path.rsplit('.', 1)
    cls = getattr(importlib.import_module(module), cls_name)
    ok = set(inspect.signature(cls.__init__).parameters)
    kw = {k: v for k, v in extra.items() if k in ok}
    for k, v in (('det_thresh', conf), ('track_thresh', conf),
                 ('min_conf', conf)):
        if k in ok:
            kw[k] = v
    return cls(**kw)


def run_one(name: str, vids: list[str], conf: float, out_dir: Path) -> float:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for vid in vids:
        cache = load_video_cache(CACHE, 'val', vid)
        tracker = make_tracker(name, conf)
        frames = []
        for fidx, fname, xyxy, cf, cls in iter_frames(cache):
            keep = cf >= conf
            dets = np.concatenate(
                [xyxy[keep].astype(np.float64),
                 cf[keep, None].astype(np.float64),
                 cls[keep, None].astype(np.float64)], axis=1) \
                if keep.any() else np.zeros((0, 6))
            img = cv2.imread(str(IMAGES / vid / fname))
            out = np.asarray(tracker.update(dets, img))
            labels = []
            if out.size:
                for row in np.atleast_2d(out):
                    # формат BoxMOT: x1,y1,x2,y2,id,conf,cls,det_ind
                    x1, y1, x2, y2, tid, score, c = row[:7]
                    labels.append({'id': str(int(tid)),
                                   'category': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                                   'score': float(score),
                                   'box2d': {'x1': float(x1), 'y1': float(y1),
                                             'x2': float(x2), 'y2': float(y2)}})
            frames.append({'videoName': vid, 'name': fname,
                           'frameIndex': fidx, 'labels': labels})
        (out_dir / f'{vid}.json').write_text(json.dumps(frames), encoding='utf-8')
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='порог уверенности — та же рабочая точка, что у наших прогонов')
    ap.add_argument('--trackers', nargs='+', default=list(ZOO))
    args = ap.parse_args()

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    print(f'[INFO] видео {len(vids)}, порог {args.conf}, трекеры {args.trackers}')

    rows, res = [], {}
    for name in args.trackers:
        if name not in ZOO:
            print(f'[ПРОПУСК] {name}: нет в списке')
            continue
        try:
            sec = run_one(name, vids, args.conf, RUNS / name)
        except Exception as e:                       # noqa: BLE001
            print(f'[ОШИБКА] {name}: {type(e).__name__}: {e}')
            continue
        met = evaluate_tracking_bdd(RUNS / name, vids, RUNS / f'{name}.json')
        hota = evaluate_hota(RUNS / name, vids, tracker_name=name)
        merged = {**{k: v for k, v in met.items() if isinstance(v, (int, float))},
                  **hota}
        res[name] = merged
        print('[%-14s] %5.0f с  ' % (name, sec) + '  '.join(
            f'{k}={merged[k]:.3f}' for k in KEYS if k in merged))
        for k, v in merged.items():
            rows.append({'трекер': name, 'метрика': k,
                         'значение': round(float(v), 4), 'секунд': round(sec, 1)})

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f'\n[OK ] {OUT}')
        print('Сравнивать с таблицей experiments/results/tracker_zoo.csv: '
              'ByteTrack IDF1 50.877, BoT-SORT 53.122 на тех же данных.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
