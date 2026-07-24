"""Метрики детекции по поднаборам (Этап 3, Часть 2.6 плана).

Три разбивки одним прогоном:
- общие mAP@0.50, mAP@0.50:0.95, Precision, Recall — из Ultralytics model.val();
- per-class AP — оттуда же (results.box.ap_class_index / ap50 / ap);
- AP_small / AP_medium / AP_large — через pycocotools COCOeval:
  GT собирается из YOLO txt-разметки, предсказания — из predictions.json
  Ultralytics (save_json=True) с приведением image_id к общим числовым id.

Границы размеров — стандарт COCO: small < 32², medium 32²–96², large > 96².
"""
import contextlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES, build_subset_yaml, list_subset_frames  # noqa: E402
from datasets.format_converter import IMG_H, IMG_W  # noqa: E402
from datasets.subset_filter import filter_videos_by_subset, load_video_attributes  # noqa: E402

DATA = PROJECT_ROOT / 'data' / 'bdd100k'


# ——————————————————— COCO-инфраструктура для AP по размерам ———————————————————

def build_coco_gt(frames: list[Path], out_json: Path) -> dict[str, int]:
    """Собирает COCO-GT из YOLO txt. Возвращает отображение stem -> image_id."""
    stem2id: dict[str, int] = {}
    images, annotations = [], []
    ann_id = 1
    for img_id, frame in enumerate(frames, 1):
        stem2id[frame.stem] = img_id
        images.append({'id': img_id, 'file_name': frame.name,
                       'width': IMG_W, 'height': IMG_H})
        txt = _label_path(frame)
        if not txt.exists():
            continue
        for line in txt.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            bw, bh = float(w) * IMG_W, float(h) * IMG_H
            x = float(cx) * IMG_W - bw / 2
            y = float(cy) * IMG_H - bh / 2
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': int(c),
                'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0,
            })
            ann_id += 1
    gt = {
        'images': images, 'annotations': annotations,
        'categories': [{'id': i, 'name': n} for i, n in CLASS_NAMES.items()],
    }
    out_json.write_text(json.dumps(gt), encoding='utf-8')
    return stem2id


def _label_path(frame: Path) -> Path:
    """images/track/<split>/<vid>/f.jpg -> labels/track/<split>/<vid>/f.txt."""
    parts = list(frame.parts)
    parts[parts.index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def remap_prediction_ids(pred_json: Path, stem2id: dict[str, int],
                         out_json: Path) -> int:
    """predictions.json Ultralytics использует stem как image_id — приводим к числам."""
    preds = json.loads(pred_json.read_text(encoding='utf-8'))
    fixed = []
    for p in preds:
        img_id = stem2id.get(str(p['image_id']))
        if img_id is not None:
            p['image_id'] = img_id
            fixed.append(p)
    out_json.write_text(json.dumps(fixed), encoding='utf-8')
    return len(fixed)


def coco_size_metrics(gt_json: Path, pred_json: Path) -> dict[str, float]:
    """AP_small / AP_medium / AP_large (индексы 3,4,5 stats COCOeval)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):  # COCO болтлив
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {
        'AP_small': float(ev.stats[3]),
        'AP_medium': float(ev.stats[4]),
        'AP_large': float(ev.stats[5]),
    }


# ——————————————————— основной вход ———————————————————

def evaluate_on_subsets(
    model_path: str,
    subsets: list[str] = ('normal', 'night', 'rain', 'fog'),
    split: str = 'val',
    per_class: bool = True,
    per_size: bool = True,
    workdir: Path | None = None,
    frame_step: int = 1,
) -> dict:
    """Оценка модели на каждом поднаборе.

    frame_step = 1 — боевой режим (полный val, без прореживания);
    frame_step > 1 — только для smoke-проверки самого конвейера оценки.
    Возвращает {subset: {metric: value, ...}} — общие, per-class и per-size.
    """
    from ultralytics import YOLO

    workdir = workdir or (PROJECT_ROOT / 'runs' / 'eval' / Path(model_path).parent.name)
    workdir.mkdir(parents=True, exist_ok=True)

    det_json = DATA / 'labels' / f'bdd100k_labels_images_{split}.json'
    video_attrs = load_video_attributes(str(det_json))
    all_ids = sorted(p.name for p in (DATA / 'images' / 'track' / split).iterdir()
                     if p.is_dir())

    out: dict[str, dict] = {}
    for subset in subsets:
        vids = filter_videos_by_subset(video_attrs, all_ids, subset)
        frames = list_subset_frames(split, vids, frame_step=frame_step)
        print(f'[INFO] {subset}: {len(vids)} видео, {len(frames)} кадров', flush=True)

        yaml_path = build_subset_yaml(f'eval_{subset}_{split}', frames, frames,
                                      out_dir=workdir / 'yaml')
        model = YOLO(model_path)
        run_name = f'{subset}'
        results = model.val(data=str(yaml_path), split='val', batch=16,
                            save_json=per_size, project=str(workdir),
                            name=run_name, exist_ok=True, verbose=False,
                            plots=False)

        m: dict = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'videos': len(vids), 'frames': len(frames),
        }
        if per_class:
            for idx, cls_id in enumerate(results.box.ap_class_index):
                name = CLASS_NAMES[int(cls_id)]
                m[f'AP50_{name}'] = float(results.box.ap50[idx])
                m[f'AP_{name}'] = float(results.box.ap[idx])
        if per_size:
            pred_json = workdir / run_name / 'predictions.json'
            if pred_json.exists():
                gt_json = workdir / run_name / 'gt_coco.json'
                stem2id = build_coco_gt(frames, gt_json)
                fixed = workdir / run_name / 'predictions_cocoids.json'
                n = remap_prediction_ids(pred_json, stem2id, fixed)
                if n:
                    m.update(coco_size_metrics(gt_json, fixed))
        out[subset] = m
    return out
