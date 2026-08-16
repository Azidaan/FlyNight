"""Smoke-тест метрик по размерам объектов (AP_small/medium/large).

Ловит ошибку, из-за которой эти метрики молча обнулялись: Ultralytics для
нестандартных датасетов нумерует классы в predictions.json С ЕДИНИЦЫ, а
эталон COCO собирался С НУЛЯ. Предсказание «пешеход» сверялось с эталонным
«водитель», совпадений не было, и AP по размерам выходил около нуля —
при том что основные метрики оставались правдоподобными.

Проверка простая и жёсткая: если предсказания в точности совпадают с
эталоном, AP обязан быть около единицы. Любое расхождение соглашений о
нумерации классов роняет его до нуля.

Запуск: python scripts/smoke/test_coco_size_metrics.py
"""
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.format_converter import IMG_H, IMG_W  # noqa: E402
from evaluation.detection_metrics import build_coco_gt, coco_size_metrics  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


# кадры трёх размеров: мелкий (<32px), средний, крупный
BOXES = [
    # (класс, cx, cy, w, h) в долях кадра
    (0, 0.50, 0.50, 20 / IMG_W, 20 / IMG_H),      # small  ~400 px^2
    (2, 0.25, 0.30, 70 / IMG_W, 70 / IMG_H),      # medium ~4900 px^2
    (2, 0.75, 0.60, 200 / IMG_W, 200 / IMG_H),    # large  ~40000 px^2
]


def build_case(root: Path) -> tuple[list[Path], Path]:
    """Кадр с разметкой на диске + путь, куда положить эталон."""
    img_dir = root / 'images' / 'track' / 'val' / 'vid1'
    lbl_dir = root / 'labels' / 'track' / 'val' / 'vid1'
    img_dir.mkdir(parents=True); lbl_dir.mkdir(parents=True)
    frame = img_dir / 'vid1-0000001.jpg'
    frame.write_bytes(b'')                       # содержимое не читается
    (lbl_dir / 'vid1-0000001.txt').write_text(
        '\n'.join(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' for c, cx, cy, w, h in BOXES),
        encoding='utf-8')
    return [frame], root / 'gt.json'


def predictions_from_gt(gt_json: Path, out: Path, shift: int) -> None:
    """Идеальные предсказания из эталона; shift — сдвиг нумерации классов."""
    gt = json.loads(gt_json.read_text(encoding='utf-8'))
    preds = [{
        'image_id': a['image_id'],
        'category_id': a['category_id'] + shift,
        'bbox': a['bbox'],
        'score': 0.9,
    } for a in gt['annotations']]
    out.write_text(json.dumps(preds), encoding='utf-8')


def main() -> int:
    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        tmp = Path(tmp)
        frames, gt_json = build_case(tmp)
        build_coco_gt(frames, gt_json)

        gt = json.loads(gt_json.read_text(encoding='utf-8'))
        cat_ids = sorted(c['id'] for c in gt['categories'])
        check('нумерация классов в эталоне начинается с 1',
              min(cat_ids) == 1, f'ids: {cat_ids[:3]}…')
        check('у рамок проставлена площадь',
              all(a['area'] > 0 for a in gt['annotations']))

        # идеальные предсказания в том же соглашении, что пишет Ultralytics
        pred = tmp / 'pred.json'
        predictions_from_gt(gt_json, pred, shift=0)
        m = coco_size_metrics(gt_json, pred)
        print('    идеальные предсказания:', {k: round(v, 3) for k, v in m.items()})
        check('AP_small ~ 1 при точном совпадении', m['AP_small'] > 0.9, str(m['AP_small']))
        check('AP_medium ~ 1 при точном совпадении', m['AP_medium'] > 0.9, str(m['AP_medium']))
        check('AP_large ~ 1 при точном совпадении', m['AP_large'] > 0.9, str(m['AP_large']))

        # та самая ошибка: предсказания сдвинуты на единицу
        predictions_from_gt(gt_json, pred, shift=1)
        bad = coco_size_metrics(gt_json, pred)
        print('    со сдвигом нумерации:', {k: round(v, 3) for k, v in bad.items()})
        check('сдвиг нумерации действительно обнуляет AP',
              max(bad.values()) < 0.1, str(bad))

    print(f'\n--- {"OK: метрики по размерам считаются верно" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
