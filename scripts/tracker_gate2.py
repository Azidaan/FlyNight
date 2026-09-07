"""Гейт 2: что даёт обучаемый предсказатель движения в настоящем трекере.

Критерий и постановка записаны заранее в dissertation/Wavelet/Wavelet.md, п. 7.3.

Модель встаёт ровно в ту точку, где в гейте А стоял оракул, — в предсказание
состояния трека перед сопоставлением. Всё остальное неизменно: тот же кэш
детекций YOLO26, те же пороги, тот же ByteTrack, те же 50 видео.

Отличие от оракула принципиальное и его надо помнить при чтении чисел. Оракул
подставлял ИСТИНУ и потому ошибиться не мог. Модель подставляет свою догадку и
ошибается систематически: уверенный промах может увести трек на чужую
детекцию, чего оракул произвести не способен. Поэтому шкала гейта А — верхняя
оценка поведения, а не предсказание результата.

Подменяется ТОЛЬКО центр рамки: размер остаётся за фильтром Калмана, как и в
режиме `center` гейта А, потолок которого +6.47 IDF1. Это делает сравнение с
потолком корректным.

Наблюдения для модели берутся такими же, как при обучении, — центры ДЕТЕКЦИЙ,
сопоставленных с треком, а не сглаженное состояние фильтра. Окно требуется
непрерывное: если в истории трека есть пропуск, подмена не делается и работает
штатный Калман. Так вход модели остаётся тем же, на котором она училась.

Режимы:
    baseline — штатный ByteTrack (точка отсчёта);
    wtconv   — предсказатель на WTConv1d;
    plain    — тот же предсказатель с обычной свёрткой (контроль: отделяет
               вклад вейвлетного слоя от вклада самого факта обучения).

Запуск: .venv/Scripts/python.exe scripts/tracker_gate2.py
        ... --gaps 1 2 3 4 5 6 8 10 --wt-type db3
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.trackers.byte_tracker import BYTETracker, STrack  # noqa: E402

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from models.wtconv1d import build_model  # noqa: E402
from scripts.tracker_gate_a import CACHE, GT, iou_matrix  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

RUNS = PROJECT_ROOT / 'runs' / 'gate2'
CKPT = PROJECT_ROOT / 'runs' / 'wtconv1d'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'gate2_predictor.csv'
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw', 'MOTP', 'DetA', 'AssA')
SCALE = 10.0          # тот же масштаб, что при обучении


class PredictorByteTrack(BYTETracker):
    """ByteTrack, у которого предсказание центра даёт обученная модель."""

    def __init__(self, args, model=None, gaps=(1, 2, 3, 4, 5, 6, 8, 10),
                 window: int = 12, device: str = 'cpu'):
        super().__init__(args)
        self.model = model
        self.gaps = list(gaps)
        self.window = window
        self.device = device
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self.n_override = 0
        self.n_fallback = 0

    def _window_for(self, tid: int):
        """Непрерывное окно последних наблюдений трека или None."""
        h = self.hist.get(tid)
        if h is None or len(h) < self.window:
            return None
        frames = [f for f, _ in h]
        if frames[-1] - frames[0] != self.window - 1:   # был пропуск внутри окна
            return None
        return np.array([c for _, c in h], np.float32), frames[-1]

    def multi_predict(self, tracks):
        super().multi_predict(tracks)
        if self.model is None or not tracks:
            return
        batch, meta = [], []
        for st in tracks:
            if st.frame_id >= self.frame_id - 1:        # не в разрыве
                continue
            got = self._window_for(st.track_id)
            if got is None:
                self.n_fallback += 1
                continue
            win, last_frame = got
            gap = self.frame_id - last_frame
            if gap not in self.gaps:                    # вне обученного диапазона
                self.n_fallback += 1
                continue
            batch.append(win)
            meta.append((st, win[-1], self.gaps.index(gap)))
        if not batch:
            return
        arr = np.stack(batch)
        d = np.diff(arr, axis=1, prepend=arr[:, :1])
        x = torch.tensor(np.transpose(d, (0, 2, 1)) / SCALE).to(self.device)
        with torch.no_grad():
            pred = self.model(x).cpu().numpy() * SCALE
        for (st, last_xy, gi), p in zip(meta, pred):
            cx, cy = last_xy + p[gi]
            st.mean[0] = cx                              # центр — от модели
            st.mean[1] = cy                              # размер остаётся Калману
            self.n_override += 1

    def observe(self, frame: int, xyxy: np.ndarray) -> None:
        """Записать наблюдения: центры ДЕТЕКЦИЙ, сопоставленных с треками.

        Берётся не состояние фильтра, а рамка детекции — на таких рядах модель
        и обучалась. Треки, обновлённые на этом кадре, сопоставляются с
        детекциями по IoU.
        """
        live = [t for t in self.tracked_stracks if t.frame_id == self.frame_id]
        if not live or len(xyxy) == 0:
            return
        M = iou_matrix(np.array([t.xyxy for t in live], np.float64),
                       xyxy.astype(np.float64))
        for i, t in enumerate(live):
            j = int(M[i].argmax())
            if M[i, j] >= 0.7:
                b = xyxy[j]
                c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            else:
                w = t.xywh
                c = (float(w[0]), float(w[1]))
            self.hist[t.track_id].append((frame, np.array(c, np.float32)))


def run_mode(vids, model, args, tr_args, out_dir: Path, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = fb = 0
    for vid in vids:
        cache = load_video_cache(CACHE, 'val', vid)
        tracker = PredictorByteTrack(tr_args, model=model, gaps=args.gaps,
                                     window=args.window, device=device)
        STrack.reset_id()
        frames = []
        for fidx, fname, xyxy, conf, cls in iter_frames(cache):
            out = tracker.update(_DetSlice(xyxy, conf, cls))
            keep = conf >= tr_args.track_high_thresh
            tracker.observe(fidx, xyxy[keep])
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
        (out_dir / f'{vid}.json').write_text(json.dumps(frames), encoding='utf-8')
        ov += tracker.n_override
        fb += tracker.n_fallback
    return ov, fb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=50)
    ap.add_argument('--tau', type=float, default=0.25)
    ap.add_argument('--buffer', type=int, default=30)
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--gaps', nargs='+', type=int,
                    default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument('--wt-type', default='db3')
    ap.add_argument('--channels', type=int, default=48)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--wt-levels', type=int, default=2)
    ap.add_argument('--modes', nargs='+', default=['baseline', 'wtconv', 'plain'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vids = sorted(p.stem for p in (CACHE / 'val').glob('*.npz')
                  if (GT / f'{p.stem}.json').exists())[:args.videos]
    print(f'[INFO] видео {len(vids)}, τ={args.tau}, буфер={args.buffer}, '
          f'разрывы {args.gaps}, устройство {device}')

    tr_args = make_tracker_args(track_high_thresh=args.tau,
                                track_buffer=args.buffer)
    res, rows = {}, []
    for mode in args.modes:
        model = None
        if mode != 'baseline':
            ck = CKPT / f'{mode}_{args.wt_type}.pt'
            if not ck.exists():
                print(f'[ПРОПУСК] нет чекпойнта {ck}')
                continue
            model = build_model(mixer=mode, wt_type=args.wt_type, gaps=args.gaps,
                                channels=args.channels, blocks=args.blocks,
                                wt_levels=args.wt_levels).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
        ov, fb = run_mode(vids, model, args, tr_args, RUNS / mode, device)
        met = evaluate_tracking_bdd(RUNS / mode, vids, RUNS / f'{mode}.json')
        res[mode] = met
        print('[%-8s] подмен предсказания %d, отказов (короткая или рваная '
              'история) %d' % (mode, ov, fb))
        print('          ' + '  '.join(
            f'{k}={met[k]:.3f}' if isinstance(met.get(k), float) else f'{k}={met.get(k)}'
            for k in METRIC_KEYS if k in met))

    base = res.get('baseline')
    print('\n=== гейт 2: прирост к штатному ByteTrack ===')
    for mode in res:
        m = res[mode]
        if base and mode != 'baseline':
            print('  %-8s ' % mode + '  '.join(
                '%s %+.2f (%+.1f%%)' % (k, m[k] - base[k],
                                        100 * (m[k] - base[k]) / base[k])
                for k in ('IDF1', 'MOTA', 'IDSw') if k in m and base.get(k)))
        for k in METRIC_KEYS:
            if k not in m:
                continue
            d = (m[k] - base[k]) if base and k in base else None
            rows.append({'режим': mode, 'метрика': k,
                         'значение': round(float(m[k]), 4),
                         'дельта_к_baseline': None if d is None else round(float(d), 4),
                         'процент_к_baseline':
                             None if not d else round(100 * d / base[k], 3)})

    if base and 'wtconv' in res:
        gain = res['wtconv']['IDF1'] - base['IDF1']
        print('\n  доля измеренного потолка (+6.47 IDF1, оракул по центру): '
              '%.0f%%' % (100 * gain / 6.47))
        if 'plain' in res:
            diff = res['wtconv']['IDF1'] - res['plain']['IDF1']
            print('  вклад собственно вейвлетного слоя: %+.2f IDF1' % diff)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
