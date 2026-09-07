"""Гейт А: какая часть потерь трекера вообще связана с предсказанием.

Зачем. Вейвлетное очищение улучшает ПРЕДСКАЗАНИЕ положения объекта на кадрах,
где детекции нет. Прежде чем улучшать предсказание, надо знать цену вопроса:
сколько метрик трекинга можно выиграть, если предсказание сделать ИДЕАЛЬНЫМ.
Это верхняя граница для всей линии. Если она мала, то спор о базисах вейвлета
не имеет практических последствий независимо от того, кто в нём прав.

Как измеряется. Один и тот же кэш детекций YOLO26, один и тот же ByteTrack,
меняется ровно одно: чем заполняется состояние трека перед сопоставлением.

    baseline     — штатное предсказание фильтра Калмана;
    oracle_gaps  — трекам, пропустившим предыдущий кадр, состояние подменяется
                   ИСТИННЫМ положением объекта из разметки. Это ровно та
                   мишень, на которую нацелено вейвлетное очищение: предсказание
                   в разрыве становится безошибочным;
    oracle_all   — истинным положением подменяется состояние ВСЕХ треков на
                   каждом кадре. Верхняя граница вклада предсказания вообще;
    center[:A]   — подменяется ТОЛЬКО центр рамки, размер остаётся за Калманом.
                   Это честный потолок для предсказателя координат центра:
                   полный оракул подменяет и размер, которым предсказатель
                   траектории не занимается;
    gaps:A       — предсказание в разрыве сдвигается к истине на долю A, то
                   есть его ошибка уменьшается ровно на A*100 процентов. Это
                   шкала чувствительности: она переводит «улучшили предсказание
                   на столько-то процентов» в пункты метрик трекинга, а без неё
                   спор о процентах ошибки в пикселях не имеет цены.

Соответствие трека объекту разметки берётся не из воздуха: после каждого кадра
рамка обновлённого трека сопоставляется с эталонными рамками по IoU, и трек
получает эталонный идентификатор. Оракул пользуется только этим соответствием и
разметкой текущего кадра — то есть подменяет предсказание, но не ассоциацию:
сопоставление по IoU и пороги остаются штатными.

Разница baseline и oracle_gaps по AssA/IDF1/IDSw и есть ответ гейта А.

Запуск: .venv/Scripts/python.exe scripts/tracker_gate_a.py
        ... --videos 50 --tau 0.25
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.trackers.byte_tracker import BYTETracker, STrack  # noqa: E402

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl'
GT = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'box_track_20' / 'val'
RUNS = PROJECT_ROOT / 'runs' / 'gate_a'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'gate_a_oracle.csv'
MODES = ('baseline', 'gaps:0.02', 'gaps:0.05', 'gaps:0.10',
         'gaps:0.25', 'gaps:0.50', 'oracle_gaps', 'oracle_all',
         'center', 'center:0.25', 'center:0.50')
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw', 'MOTP', 'DetA', 'AssA')


def gt_frames(vid: str) -> dict[int, tuple[list[str], np.ndarray]]:
    data = json.loads((GT / f'{vid}.json').read_text(encoding='utf-8'))
    out = {}
    for fr in data:
        ids, boxes = [], []
        for l in fr.get('labels') or []:
            b = l.get('box2d')
            if not b:
                continue
            ids.append(l['id'])
            boxes.append([b['x1'], b['y1'], b['x2'], b['y2']])
        out[fr['frameIndex']] = (ids, np.array(boxes, np.float64).reshape(-1, 4))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-9)


class OracleByteTrack(BYTETracker):
    """ByteTrack, которому предсказание положения подменяется разметкой."""

    def __init__(self, args, gt=None, mode='oracle_gaps'):
        super().__init__(args)
        self.gt = gt or {}
        self.mode = mode
        self.track2gt: dict[int, str] = {}
        self.n_override = 0
        self.n_pred = 0

    @property
    def alpha(self) -> float:
        """Доля пути к истине: 1.0 у оракула, A у режима gaps:A."""
        return float(self.mode.split(':')[1]) if ':' in self.mode else 1.0

    @property
    def only_gaps(self) -> bool:
        return self.mode != 'oracle_all'

    @property
    def center_only(self) -> bool:
        """Подменять только центр рамки, оставляя её размер за Калманом.

        Нужно потому, что полный оракул подменяет и размер тоже, а вейвлетный
        предсказатель работает только с координатами центра. Полный оракул
        поэтому даёт ЗАВЫШЕННУЮ верхнюю границу, а этот режим — честную для
        предсказателя центра.
        """
        return self.mode.startswith('center')

    def multi_predict(self, tracks):
        super().multi_predict(tracks)
        if self.mode == 'baseline' or not tracks:
            return
        ids, boxes = self.gt.get(self.frame_id - 1, ([], np.zeros((0, 4))))
        if not ids:
            self.n_pred += len(tracks)
            return
        pos = {oid: i for i, oid in enumerate(ids)}
        a = self.alpha
        for st in tracks:
            self.n_pred += 1
            # только треки, не обновлённые на предыдущем кадре (кроме oracle_all)
            if self.only_gaps and st.frame_id >= self.frame_id - 1:
                continue
            oid = self.track2gt.get(st.track_id)
            if oid is None or oid not in pos:
                continue
            b = boxes[pos[oid]]
            tlwh = np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], np.float64)
            true_state = STrack.tlwh_to_xyah(tlwh)
            k = 2 if self.center_only else 4
            st.mean[:k] = st.mean[:k] + a * (true_state[:k] - st.mean[:k])
            self.n_override += 1

    def bind_gt(self, frame: int, iou_thr: float = 0.5) -> None:
        """Привязать обновлённые на этом кадре треки к объектам разметки."""
        ids, boxes = self.gt.get(frame, ([], np.zeros((0, 4))))
        if not ids:
            return
        live = [t for t in self.tracked_stracks if t.frame_id == self.frame_id]
        if not live:
            return
        M = iou_matrix(np.array([t.xyxy for t in live], np.float64), boxes)
        for i, t in enumerate(live):
            j = int(M[i].argmax())
            if M[i, j] >= iou_thr:
                self.track2gt[t.track_id] = ids[j]


def run_mode(vids: list[str], mode: str, args_ns, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tot_ov = tot_pr = 0
    for vid in vids:
        out_json = out_dir / f'{vid}.json'
        cache = load_video_cache(CACHE, 'val', vid)
        gt = gt_frames(vid)
        tracker = OracleByteTrack(args_ns, gt=gt, mode=mode)
        STrack.reset_id()
        frames = []
        for fidx, fname, xyxy, conf, cls in iter_frames(cache):
            out = tracker.update(_DetSlice(xyxy, conf, cls))
            tracker.bind_gt(fidx)
            labels = []
            for row in out:
                x1, y1, x2, y2, tid, score, c = row[:7]
                labels.append({'id': str(int(tid)),
                               'category': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                               'score': float(score),
                               'box2d': {'x1': float(x1), 'y1': float(y1),
                                         'x2': float(x2), 'y2': float(y2)}})
            frames.append({'videoName': vid, 'name': fname,
                           'frameIndex': fidx, 'labels': labels})
        out_json.write_text(json.dumps(frames), encoding='utf-8')
        tot_ov += tracker.n_override
        tot_pr += tracker.n_pred
    return tot_ov, tot_pr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--tau', type=float, default=0.25, help='track_high_thresh')
    ap.add_argument('--buffer', type=int, default=30)
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    args = ap.parse_args()

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    if not vids:
        raise SystemExit(f'нет кэша детекций: {CACHE}')
    print(f'[INFO] видео {len(vids)}, τ={args.tau}, буфер={args.buffer}')

    tr_args = make_tracker_args(track_high_thresh=args.tau,
                                track_buffer=args.buffer)
    res = {}
    for mode in args.modes:
        out_dir = RUNS / mode.replace(':', '_')
        ov, pr = run_mode(vids, mode, tr_args, out_dir)
        met = evaluate_tracking_bdd(out_dir, vids, RUNS / f"{mode.replace(':', '_')}.json")
        res[mode] = met
        print('[%-11s] подмен предсказания %d из %d состояний трека'
              % (mode, ov, pr))
        print('             ' + '  '.join(
            f'{k}={met.get(k)}' if not isinstance(met.get(k), float)
            else f'{k}={met[k]:.3f}' for k in METRIC_KEYS if k in met))

    print('\n=== ответ гейта А: цена идеального предсказания ===')
    base = res.get('baseline')
    rows = []
    for mode in args.modes:
        m = res[mode]
        for k in METRIC_KEYS:
            if k not in m:
                continue
            d = (m[k] - base[k]) if base and k in base else None
            rows.append({'режим': mode, 'метрика': k,
                         'значение': round(float(m[k]), 4),
                         'дельта_к_baseline': None if d is None else round(float(d), 4)})
    if base:
        for mode in args.modes:
            if mode == 'baseline':
                continue
            m = res[mode]
            print('  %-12s ' % mode + '  '.join(
                '%s %+.2f' % (k, m[k] - base[k])
                for k in ('HOTA', 'IDF1', 'MOTA', 'AssA', 'DetA', 'IDSw')
                if k in m and k in base))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
