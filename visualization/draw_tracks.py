"""Наложение треков на кадры и сборка видео (Этап 10, подэтапы 10.2-10.3).

Вход — scalabel-JSON трекинга (формат evaluation/tracking_metrics и
tracking/offline_association). Каждому track_id — стабильный цвет;
рисуются рамка, id и хвост траектории (центры последних кадров).

Использование как модуль:
    draw_frame(frame, labels, trails)  — один кадр
    render_video(frames_dir, scalabel_json, out_mp4)  — видео целиком
Как скрипт:
    python visualization/draw_tracks.py --frames data/bdd100k/images/track/val/<vid> \
        --tracks runs/track_modes/lqa_night/<vid>.json --out figs/final/demo.mp4
"""
import argparse
import colorsys
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIL_LEN = 30


def track_color(track_id: str) -> tuple[int, int, int]:
    """Стабильный насыщенный цвет BGR по id трека."""
    h = (hash(track_id) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_frame(frame: np.ndarray, labels: list[dict],
               trails: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    """Рисует рамки, id и хвосты. trails обновляется на месте."""
    out = frame.copy()
    for lab in labels:
        b = lab['box2d']
        tid = lab['id']
        color = track_color(tid)
        p1 = (int(b['x1']), int(b['y1']))
        p2 = (int(b['x2']), int(b['y2']))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, tid, (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        trails[tid].append((cx, cy))
    for tid, pts in trails.items():
        pts = pts[-TRAIL_LEN:]
        trails[tid] = pts
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, track_color(tid), 2)
    return out


def render_video(frames_dir: str | Path, scalabel_json: str | Path,
                 out_mp4: str | Path, fps: int = 5) -> Path:
    """Собирает mp4 с наложенными треками (5 fps — родная частота BDD MOT)."""
    frames_dir = Path(frames_dir)
    data = json.loads(Path(scalabel_json).read_text(encoding='utf-8'))
    by_name = {f['name']: f['labels'] for f in data}

    files = sorted(frames_dir.glob('*.jpg'))
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))
    trails: dict[str, list] = defaultdict(list)
    for f in files:
        frame = cv2.imread(str(f))
        writer.write(draw_frame(frame, by_name.get(f.name, []), trails))
    writer.release()
    return out_mp4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', required=True, help='директория кадров видео')
    ap.add_argument('--tracks', required=True, help='scalabel JSON трекинга')
    ap.add_argument('--out', required=True, help='выходной mp4')
    ap.add_argument('--fps', type=int, default=5)
    args = ap.parse_args()
    out = render_video(args.frames, args.tracks, args.out, args.fps)
    print(f'[OK ] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
