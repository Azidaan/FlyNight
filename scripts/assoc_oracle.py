"""Гейт 0б: какой запас лежит в ассоциации при тех же детекциях.

Зачем. Гейт А измерил запас на ПРЕДСКАЗАНИИ положения (+6.47 IDF1 для
предсказателя центра). Запас на самом РЕШЕНИИ о сопоставлении не измерен ни
разу, а именно туда целится вся линия адаптивной ассоциации. Без этого числа
разработка идёт вслепую: неизвестно, за 6 пунктов борьба или за 0.5.

Как измеряется. Детекции остаются теми же самыми — тот же кэш YOLO26, тот же
порог уверенности. Меняется только присвоение личности: каждая детекция
получает идентификатор того эталонного объекта, с которым она совпадает по IoU.
То есть ассоциация становится безошибочной при неизменном детекторе.

Детекции, которым эталонного объекта не нашлось, получают собственные
одноразовые номера: это ложные срабатывания, и прятать их нельзя — иначе
оракул тихо улучшит ещё и обнаружение, а меряется только связывание.

Верхняя граница получается оптимистичной по построению: она не ограничена ни
буфером треков, ни правилами создания трека, ни порогом сопоставления. Это
именно потолок, а не достижимый результат.

Запуск: .venv/Scripts/python.exe scripts/assoc_oracle.py --videos 50
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from evaluation.hota_eval import evaluate_hota  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT, gt_frames, iou_matrix  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402

RUNS = PROJECT_ROOT / 'runs' / 'assoc_oracle'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'assoc_oracle.csv'
KEYS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSw', 'MOTP')


def run_video(vid: str, conf: float, iou_thr: float, out_dir: Path) -> tuple:
    gt = gt_frames(vid)
    cache = load_video_cache(CACHE, 'val', vid)
    id_map: dict[str, int] = {}
    frames, n_hit, n_miss = [], 0, 0
    fake = 1_000_000                      # номера для детекций без эталона
    for fidx, fname, xyxy, conf_arr, cls in iter_frames(cache):
        keep = conf_arr >= conf
        det, dconf, dcls = xyxy[keep], conf_arr[keep], cls[keep]
        ids, boxes = gt.get(fidx, ([], np.zeros((0, 4))))
        labels = []
        if len(det):
            M = iou_matrix(det.astype(np.float64), boxes)
            best = M.argmax(axis=1) if len(boxes) else np.zeros(len(det), int)
            taken: set[int] = set()
            for i in range(len(det)):
                j = int(best[i]) if len(boxes) else -1
                if len(boxes) and M[i, j] >= iou_thr and j not in taken:
                    taken.add(j)          # один эталон — одной детекции
                    oid = ids[j]
                    tid = id_map.setdefault(oid, len(id_map) + 1)
                    n_hit += 1
                else:
                    tid = fake
                    fake += 1
                    n_miss += 1
                x1, y1, x2, y2 = det[i]
                labels.append({'id': str(tid),
                               'category': CLASS_NAMES.get(int(dcls[i]),
                                                           f'class_{int(dcls[i])}'),
                               'score': float(dconf[i]),
                               'box2d': {'x1': float(x1), 'y1': float(y1),
                                         'x2': float(x2), 'y2': float(y2)}})
        frames.append({'videoName': vid, 'name': fname, 'frameIndex': fidx,
                       'labels': labels})
    (out_dir / f'{vid}.json').write_text(json.dumps(frames), encoding='utf-8')
    return n_hit, n_miss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='порог уверенности — рабочая точка трекера')
    ap.add_argument('--iou', type=float, default=0.5)
    args = ap.parse_args()

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    out_dir = RUNS / f'conf{args.conf}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] видео {len(vids)}, порог уверенности {args.conf}')

    hit = miss = 0
    for vid in vids:
        a, b = run_video(vid, args.conf, args.iou, out_dir)
        hit += a
        miss += b
    print(f'[INFO] детекций с эталоном {hit}, без эталона {miss} '
          f'({100 * miss / max(hit + miss, 1):.1f}% ложных)')

    met = evaluate_tracking_bdd(out_dir, vids, RUNS / 'oracle.json')
    hota = evaluate_hota(out_dir, vids, tracker_name='assoc_oracle')
    print('\n=== потолок ассоциации при неизменных детекциях ===')
    print('  штатный конвейер: ' + '  '.join(
        f'{k}={met[k]:.3f}' if isinstance(met.get(k), float) else f'{k}={met.get(k)}'
        for k in ('IDF1', 'MOTA', 'IDSw', 'MOTP') if k in met))
    print('  TrackEval:        ' + '  '.join(
        f'{k}={hota[k]:.3f}' for k in KEYS if k in hota))

    rows = [{'источник': 'bdd100k', 'метрика': k, 'значение': round(float(v), 4)}
            for k, v in met.items() if isinstance(v, (int, float))]
    rows += [{'источник': 'trackeval', 'метрика': k, 'значение': round(v, 4)}
             for k, v in hota.items()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
