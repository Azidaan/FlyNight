"""Разовая конвертация всех MOT-аннотаций BDD100K в YOLO txt (Этап 2).

labels/box_track_20/{train,val}/<video>.json ->
labels/track/{train,val}/<video>/<frame>.txt

Путь labels/track/... зеркалит images/track/... — Ultralytics находит разметку
автоматической заменой /images/ -> /labels/ в пути кадра.

Статистика пишется в notes/conversion_stats.json.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.format_converter import convert_split  # noqa: E402

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
OUT_JSON = PROJECT_ROOT / 'notes' / 'conversion_stats.json'


def main() -> int:
    report = {}
    for split in ('train', 'val'):
        src = DATA / 'labels' / 'box_track_20' / split
        dst = DATA / 'labels' / 'track' / split
        print(f'[INFO] Конвертация {split}: {src} -> {dst}', flush=True)
        t0 = time.perf_counter()
        stats = convert_split(str(src), str(dst))
        stats['seconds'] = round(time.perf_counter() - t0, 1)
        report[split] = stats
        print(f'[OK ] {split}: {stats["videos"]} видео, {stats["frames"]} кадров, '
              f'{stats["boxes"]} рамок, crowd пропущено {stats["crowd_skipped"]}, '
              f'неизвестные категории {stats["unknown_skipped"]}, '
              f'{stats["seconds"]} c')

    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[INFO] Статистика: {OUT_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
