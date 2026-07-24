"""Визуальная проверка конвертации: кадр из каждого поднабора с рамками (Этап 2).

Рамки рисуются ИЗ YOLO txt (а не из исходного JSON) — так проверяется именно
результат конвертации. Выход: experiments/results/subset_samples/<subset>.jpg.
Критерий готовности Этапа 2: рамки на месте, классы корректные (проверка глазами).
"""
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from datasets.subset_filter import (  # noqa: E402
    SUBSET_DEFINITIONS, filter_videos_by_subset, load_video_attributes,
)

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'subset_samples'
COLORS = [
    (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
    (0, 0, 255), (255, 255, 0), (128, 0, 255), (255, 0, 0),
]


def draw_yolo_labels(img_path: Path, txt_path: Path) -> 'cv2.Mat':
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if txt_path.exists():
        for line in txt_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()
            cls = int(cls)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASS_NAMES[cls], (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    det_json = DATA / 'labels' / 'bdd100k_labels_images_val.json'
    print('[INFO] Загрузка атрибутов val...', flush=True)
    video_attrs = load_video_attributes(str(det_json))
    img_root = DATA / 'images' / 'track' / 'val'
    mot_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())

    ok = True
    for subset in SUBSET_DEFINITIONS:
        vids = filter_videos_by_subset(video_attrs, mot_ids, subset)
        if not vids:
            print(f'[FAIL] {subset}: нет видео в val')
            ok = False
            continue
        vid = vids[0]
        frames = sorted((img_root / vid).glob('*.jpg'))
        frame = frames[len(frames) // 2]  # середина видео — обычно самая насыщенная
        txt = DATA / 'labels' / 'track' / 'val' / vid / f'{frame.stem}.txt'
        if not txt.exists():
            print(f'[FAIL] {subset}: нет YOLO txt {txt} — конвертация не выполнена?')
            ok = False
            continue
        img = draw_yolo_labels(frame, txt)
        n_boxes = len([ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()])
        out_path = OUT / f'{subset}_{vid}_{frame.stem}.jpg'
        cv2.imwrite(str(out_path), img)
        print(f'[OK ] {subset}: {out_path.name} ({n_boxes} рамок)')

    print(f'\n{"--- OK: сэмплы сохранены, проверьте глазами ---" if ok else "--- ПРОВЕРКА НЕ ПРОЙДЕНА ---"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
