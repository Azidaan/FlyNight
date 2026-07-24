"""Сквозная демонстрация: видео → детекция+Q → трекинг → рендер (Этап 10, п. 10.2).

Главный артефакт практической части (раздел 3.3): принимает директорию
кадров BDD-видео, прогоняет полный конвейер и пишет mp4 с наложенными
траекториями. Режим трекера переключается — для демонстрации «стандартный
теряет объект, LQA-guided не теряет» на одном и том же видео.

Использование:
    python prototype/pipeline.py --model weights/drb_aug_full_seed42/best.pt \
        --video data/bdd100k/images/track/val/b1c9c847-3bda4659 \
        --mode lqa --out figs/final/demo_lqa.mp4
    ... --mode standard --out figs/final/demo_standard.mp4
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.inference_with_q import QInference  # noqa: E402
from tracking.lqa_guided_tracker import LQAGuidedByteTrack, _DetSliceQ  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402
from visualization.draw_tracks import draw_frame, track_color  # noqa: E402


def run_pipeline(model_path: str, video_dir: str | Path, out_mp4: str | Path,
                 mode: str = 'lqa', tau_base: float = 0.5, k: float = 0.2,
                 fps: int = 5) -> Path:
    """Полный конвейер на одном видео. Возвращает путь к mp4."""
    video_dir = Path(video_dir)
    frames = sorted(video_dir.glob('*.jpg'))
    if not frames:
        raise FileNotFoundError(f'Нет кадров в {video_dir}')

    qi = QInference(model_path)
    args = make_tracker_args(track_high_thresh=tau_base)
    if mode == 'lqa':
        if not qi.with_drb:
            raise RuntimeError('Режим lqa требует модель с DRB')
        args.lqa_k = k
        tracker = LQAGuidedByteTrack(args)
    elif mode == 'standard':
        from ultralytics.trackers.byte_tracker import BYTETracker
        tracker = BYTETracker(args)
    else:
        raise ValueError(f'mode: standard|lqa, получено {mode!r}')

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))
    trails: dict[str, list] = defaultdict(list)
    t0 = time.time()
    n_tracks: set[int] = set()
    for fp in frames:
        det = qi.predict_frame(fp, aggregations=('mean',))
        if mode == 'lqa':
            dets = _DetSliceQ(det['xyxy'], det['conf'], det['cls'], det['q_mean'])
        else:
            dets = _DetSlice(det['xyxy'], det['conf'], det['cls'])
        out = tracker.update(dets)
        labels = [{'id': str(int(r[4])), 'box2d': {
            'x1': float(r[0]), 'y1': float(r[1]),
            'x2': float(r[2]), 'y2': float(r[3])}} for r in out]
        n_tracks.update(int(r[4]) for r in out)
        frame = cv2.imread(str(fp))
        writer.write(draw_frame(frame, labels, trails))
    writer.release()
    dt = time.time() - t0
    print(f'[OK ] {len(frames)} кадров за {dt:.1f}s '
          f'({len(frames)/dt:.1f} FPS конвейера, {len(n_tracks)} треков)')
    return out_mp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--video', required=True, help='директория кадров видео')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='lqa', choices=['standard', 'lqa'])
    ap.add_argument('--tau-base', type=float, default=0.5)
    ap.add_argument('--k', type=float, default=0.2)
    args = ap.parse_args()
    out = run_pipeline(args.model, args.video, args.out, args.mode,
                       args.tau_base, args.k)
    print(f'[OK ] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
