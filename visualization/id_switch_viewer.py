"""Поиск кандидатов «ID switch до/после» для защиты (Этап 10, подэтап 10.3).

Сравнивает два scalabel-результата одного видео (standard vs lqa-guided)
и находит кадры, где в standard-режиме рамка «перескочила» на новый id
(эвристика: рамка кадра t+1 с IoU >= порога к рамке кадра t, но с другим id,
при этом старый id в t+1 исчез), а в lqa-режиме в той же области id стабилен.

Это отбор КАНДИДАТОВ: финальный выбор 2-3 ярких примеров для слайдов —
глазами, как и предписывает план (риск «нет чётких примеров»).

Выход: CSV со столбцами video, frame_idx, box, sw_old_id, sw_new_id
и (опц.) вырезанные пары кадров с отрисовкой.

Использование:
    python visualization/id_switch_viewer.py \
        --standard runs/track_modes/standard_night --lqa runs/track_modes/lqa_night \
        --out experiments/results/id_switch_candidates.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IOU_MATCH = 0.6


def iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    sa = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    sb = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / (sa + sb - inter + 1e-9)


def find_switches(frames: list[dict]) -> list[dict]:
    """Эвристические ID switch в одном scalabel-результате."""
    out = []
    for prev, cur in zip(frames, frames[1:]):
        cur_ids = {l['id'] for l in cur['labels']}
        for lp in prev['labels']:
            if lp['id'] in cur_ids:
                continue  # трек продолжился — не switch
            # старый id исчез; ищем новую рамку на его месте
            best = max(cur['labels'],
                       key=lambda lc: iou(lp['box2d'], lc['box2d']),
                       default=None)
            if best is None:
                continue
            if iou(lp['box2d'], best['box2d']) >= IOU_MATCH:
                out.append({'frame_idx': cur['frameIndex'],
                            'old_id': lp['id'], 'new_id': best['id'],
                            'box': best['box2d']})
    return out


def region_stable(frames: list[dict], frame_idx: int, box: dict) -> bool:
    """Стабилен ли id в той же области у другого режима (кадры t-1, t)."""
    prev = next((f for f in frames if f['frameIndex'] == frame_idx - 1), None)
    cur = next((f for f in frames if f['frameIndex'] == frame_idx), None)
    if prev is None or cur is None:
        return False
    lp = max(prev['labels'], key=lambda l: iou(box, l['box2d']), default=None)
    lc = max(cur['labels'], key=lambda l: iou(box, l['box2d']), default=None)
    if lp is None or lc is None:
        return False
    return (iou(box, lc['box2d']) >= IOU_MATCH and lp['id'] == lc['id'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--standard', required=True,
                    help='директория scalabel-JSON standard-режима')
    ap.add_argument('--lqa', required=True,
                    help='директория scalabel-JSON lqa-режима')
    ap.add_argument('--out', default=str(
        PROJECT_ROOT / 'experiments' / 'results' / 'id_switch_candidates.csv'))
    args = ap.parse_args()

    std_dir, lqa_dir = Path(args.standard), Path(args.lqa)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_found = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video', 'frame_idx', 'old_id', 'new_id',
                    'x1', 'y1', 'x2', 'y2'])
        for std_json in sorted(std_dir.glob('*.json')):
            if std_json.name.startswith('_'):
                continue
            lqa_json = lqa_dir / std_json.name
            if not lqa_json.exists():
                continue
            std = json.loads(std_json.read_text(encoding='utf-8'))
            lqa = json.loads(lqa_json.read_text(encoding='utf-8'))
            for sw in find_switches(std):
                if region_stable(lqa, sw['frame_idx'], sw['box']):
                    b = sw['box']
                    w.writerow([std_json.stem, sw['frame_idx'], sw['old_id'],
                                sw['new_id'], f'{b["x1"]:.0f}', f'{b["y1"]:.0f}',
                                f'{b["x2"]:.0f}', f'{b["y2"]:.0f}'])
                    n_found += 1
    print(f'[OK ] {out_path}: {n_found} кандидатов '
          f'(switch в standard, стабильно в lqa)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
