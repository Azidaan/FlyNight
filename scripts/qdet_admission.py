"""Метод: калиброванная достоверность детекции управляет допуском в трекер.

Замысел, следующий прямо из измерений проекта. Ночью 21.0% объектов имеют
верную рамку, но уверенность ниже рабочего порога, против 15.5% днём; при равном
балле ночная детекция вернее дневной на +0.098 в рабочей полосе, а класс
«автобус» переоценен на 0.20-0.40. Значит один и тот же порог по сырому баллу
означает РАЗНУЮ вероятность быть настоящим объектом в разных условиях и у разных
классов. Метод: заменить сырой балл на калиброванную вероятность q_det в тех
решениях трекера, где он используется как порог, оставив всё остальное нетронутым.

Три точки, где ByteTrack принимает решение по баллу:

    1. разделение на сильные и слабые детекции (track_high_thresh);
    2. допуск во вторую волну (track_low_thresh);
    3. создание нового трека (new_track_thresh).

Здесь подменяется величина ровно в этих трёх местах. Стоимость сопоставления
(fuse_score) по умолчанию остаётся на сыром балле: так меняется один механизм,
и прирост можно отнести к нему. Вариант с подменой и в стоимости включается
отдельным флагом — это уже второй механизм, и он меряется отдельно.

Честность сравнения. q_det распределена иначе, чем сырой балл, поэтому один и
тот же порог 0.25 означал бы другую рабочую точку, и прирост оказался бы
приростом от сдвига порога, а не от калибровки. Поэтому порог по q подбирается
так, чтобы ДОЛЯ допущенных детекций совпадала с долей при сыром пороге. Тогда
трекер принимает столько же наблюдений, сколько принимал, — но других.

Источник q_det: scripts/reliability_gate.py, предсказания вне обучения
(калибратор не видел того видео, к которому применяется).

Запуск: .venv/Scripts/python.exe scripts/qdet_admission.py --step Q3
        ... --steps Q1 Q2 Q3 --fuse-q
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
from evaluation.hota_eval import evaluate_hota  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT, gt_frames, iou_matrix  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

QPATH = PROJECT_ROOT / 'runs' / 'reliability' / 'q_det.npz'
RUNS = PROJECT_ROOT / 'runs' / 'qdet'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'qdet_admission.csv'
KEYS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSw', 'MOTP')


class _DetSliceQ(_DetSlice):
    """Детекции кадра, несущие и сырой балл, и калиброванную достоверность."""

    def __init__(self, xyxy, conf, cls, q):
        super().__init__(xyxy, conf, cls)
        self.q = q

    def __getitem__(self, mask):
        return _DetSliceQ(self._xyxy[mask], self.conf[mask], self.cls[mask],
                          self.q[mask])


class QAdmissionByteTrack(BYTETracker):
    """ByteTrack, у которого допуск решается по q_det, а не по сырому баллу."""

    def __init__(self, args, fuse_q: bool = False):
        super().__init__(args)
        self.fuse_q = fuse_q
        self._q_of_track: dict[int, float] = {}

    def _split_detections(self, results):
        """Разделение на сильные и слабые — по калиброванной достоверности."""
        q = getattr(results, 'q', results.conf)
        high = q >= self.args.track_high_thresh
        low = (q > self.args.track_low_thresh) & (q < self.args.track_high_thresh)
        return results[high], results[low], high, low

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        q = getattr(results, 'q', None)
        if q is not None:
            for t, qi in zip(tracks, q):
                t.q_det = float(qi)
                if self.fuse_q:          # второй механизм, по умолчанию выключен
                    t.score = float(qi)
        return tracks

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        """Создание трека — по q_det: именно здесь сидит измеренный разрыв."""
        for inew in u_detection:
            track = detections[inew]
            if getattr(track, 'q_det', track.score) < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)


def load_q(step: str) -> dict:
    """q_det по (видео, строка кэша) плюс сырой балл для подбора порога."""
    d = np.load(QPATH, allow_pickle=True)
    if step not in d.files:
        raise SystemExit(f'в {QPATH.name} нет ступени {step}; есть: {d.files}')
    videos = [str(v) for v in d['videos']]
    out = {}
    for vi, name in enumerate(videos):
        m = d['video'] == vi
        out[name] = {'row': d['row'][m], 'q': d[step][m], 'conf': d['conf'][m]}
    return out, np.asarray(d[step]), np.asarray(d['conf'])


def match_threshold(q_all: np.ndarray, conf_all: np.ndarray, tau: float) -> float:
    """Порог по q, допускающий ТУ ЖЕ ДОЛЮ детекций, что и сырой порог tau."""
    frac = float((conf_all >= tau).mean())
    return float(np.quantile(q_all, 1 - frac)) if 0 < frac < 1 else tau


def run(vids, qmap, tau_hi, tau_lo, args, out_dir: Path, fuse_q: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_args = make_tracker_args(track_high_thresh=tau_hi, track_low_thresh=tau_lo,
                                track_buffer=args.buffer,
                                new_track_thresh=tau_hi)
    births = {'всего': 0}
    for vid in vids:
        cache = load_video_cache(CACHE, 'val', vid)
        info = qmap.get(vid)
        q_full = np.full(len(cache['conf']), np.nan, np.float64)
        if info is not None:
            q_full[info['row']] = info['q']
        # детекции ниже нижней границы гейта калибратора остаются со своим баллом
        q_full = np.where(np.isnan(q_full), cache['conf'].astype(np.float64), q_full)
        fidx_all = cache['frame_idx']
        STrack.reset_id()
        tracker = QAdmissionByteTrack(tr_args, fuse_q=fuse_q)
        frames = []
        for fidx, fname, xyxy, conf, cls in iter_frames(cache):
            q = q_full[fidx_all == fidx]
            out = tracker.update(_DetSliceQ(xyxy, conf, cls, q))
            labels = []
            for row in np.atleast_2d(out) if len(out) else []:
                x1, y1, x2, y2, tid, score, c = row[:7]
                labels.append({'id': str(int(tid)),
                               'category': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                               'score': float(score),
                               'box2d': {'x1': float(x1), 'y1': float(y1),
                                         'x2': float(x2), 'y2': float(y2)}})
            frames.append({'videoName': vid, 'name': fname,
                           'frameIndex': fidx, 'labels': labels})
        (out_dir / f'{vid}.json').write_text(json.dumps(frames), encoding='utf-8')
    return births


def track_births(result_dir: Path, vids: list[str], iou_thr: float = 0.5) -> dict:
    """Механистический счёт: сколько треков рождено верно, сколько впустую.

    HOTA на вопрос «улучшилось ли решение о создании траектории» не отвечает,
    поэтому рождения считаются отдельно: трек считается истинным, если кадр его
    появления содержит эталонный объект того же класса с перекрытием >= 0.5.
    """
    true_b = false_b = 0
    for vid in vids:
        gt = gt_frames(vid)
        data = json.loads((result_dir / f'{vid}.json').read_text(encoding='utf-8'))
        seen: set[str] = set()
        for fr in sorted(data, key=lambda x: x['frameIndex']):
            ids, boxes = gt.get(fr['frameIndex'], ([], np.zeros((0, 4))))
            for l in fr['labels']:
                if l['id'] in seen:
                    continue
                seen.add(l['id'])
                b = l['box2d']
                box = np.array([[b['x1'], b['y1'], b['x2'], b['y2']]], np.float64)
                ok = len(boxes) and iou_matrix(box, boxes).max() >= iou_thr
                true_b += int(bool(ok))
                false_b += int(not ok)
    prec = true_b / max(true_b + false_b, 1)
    return {'рождений_истинных': true_b, 'рождений_ложных': false_b,
            'точность_рождений': round(100 * prec, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--tau', type=float, default=0.25, help='сырой порог базы')
    ap.add_argument('--low', type=float, default=0.1)
    ap.add_argument('--buffer', type=int, default=30)
    ap.add_argument('--steps', nargs='+', default=['Q1', 'Q2', 'Q3'])
    ap.add_argument('--fuse-q', action='store_true',
                    help='подменять балл и в стоимости сопоставления (2-й механизм)')
    args = ap.parse_args()

    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    print(f'[INFO] видео {len(vids)}, сырой порог {args.tau}/{args.low}, '
          f'ступени {args.steps}, подмена в стоимости: {args.fuse_q}')

    rows, res = [], {}

    # база: тот же трекер, тот же код, но допуск по сырому баллу
    base_dir = RUNS / 'baseline'
    qmap_empty = {}
    run(vids, qmap_empty, args.tau, args.low, args, base_dir, False)
    met = evaluate_tracking_bdd(base_dir, vids, RUNS / 'baseline.json')
    hota = evaluate_hota(base_dir, vids, tracker_name='base')
    b = {**{k: v for k, v in met.items() if isinstance(v, (int, float))}, **hota,
         **track_births(base_dir, vids)}
    res['сырой балл'] = b
    print('[%-14s] ' % 'сырой балл' + '  '.join(
        f'{k}={b[k]:.3f}' for k in KEYS if k in b))
    print('               рождений истинных %d, ложных %d, точность %.2f%%'
          % (b['рождений_истинных'], b['рождений_ложных'], b['точность_рождений']))

    for step in args.steps:
        qmap, q_all, conf_all = load_q(step)
        thi = match_threshold(q_all, conf_all, args.tau)
        tlo = match_threshold(q_all, conf_all, args.low)
        print(f'[INFO] {step}: порог по q подобран по равной доле допуска — '
              f'{thi:.4f} вместо {args.tau} и {tlo:.4f} вместо {args.low}')
        d = RUNS / step
        run(vids, qmap, thi, tlo, args, d, args.fuse_q)
        met = evaluate_tracking_bdd(d, vids, RUNS / f'{step}.json')
        hota = evaluate_hota(d, vids, tracker_name=step)
        m = {**{k: v for k, v in met.items() if isinstance(v, (int, float))}, **hota,
             **track_births(d, vids)}
        res[step] = m
        print('[%-14s] ' % step + '  '.join(f'{k}={m[k]:.3f}' for k in KEYS if k in m))
        print('               рождений истинных %d, ложных %d, точность %.2f%%'
              % (m['рождений_истинных'], m['рождений_ложных'], m['точность_рождений']))

    print('\n=== прирост к допуску по сырому баллу (та же доля допущенных) ===')
    for name, m in res.items():
        if name == 'сырой балл':
            continue
        print('  %-4s ' % name + '  '.join(
            '%s %+.2f (%+.1f%%)' % (k, m[k] - b[k], 100 * (m[k] - b[k]) / b[k])
            for k in ('HOTA', 'AssA', 'DetA', 'IDF1') if k in m and b.get(k)))
        print('       рождения: истинных %+d, ложных %+d, точность %+.2f п.п.'
              % (m['рождений_истинных'] - b['рождений_истинных'],
                 m['рождений_ложных'] - b['рождений_ложных'],
                 m['точность_рождений'] - b['точность_рождений']))

    for name, m in res.items():
        for k, v in m.items():
            rows.append({'вариант': name, 'метрика': k, 'значение': round(float(v), 4)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
