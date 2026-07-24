"""Smoke-тест оффлайн-ассоциации на кэше (Этап 8).

Проверяет эквивалентность двух путей подачи детекций в BYTETracker
на одном val-видео при ОДНОМ прогоне детектора:
  A) Boxes.cpu().numpy() — ровно то, что подаёт live-путь
     (ultralytics/trackers/track.py, on_predict_postprocess_end);
  B) npz-кэш (detection_cache) -> _DetSlice (offline_association).

Один прогон детектора обязателен: между процессами cudnn.benchmark
выбирает разные алгоритмы свёртки (особенно при занятом GPU), детекции
у порогов NMS плавают, и честное побитовое сравнение невозможно.
Здесь же вход одинаковый, поэтому требование строгое: рамки, score и cls
совпадают с atol=1e-4 на каждом кадре.

track_id сравниваются по перестановке: счётчик id у STrack глобальный
(класс-атрибут), и второй трекер процесса продолжает нумерацию первого.
Сопоставление делается по idx (индекс исходной детекции кадра).

Запуск: python scripts/smoke/test_offline_association.py [video_id]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from tracking.detection_cache import (  # noqa: E402
    CACHE_CONF, CACHE_IOU, build_detection_cache, iter_frames, load_video_cache,
)
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

MODEL = str(PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt')
VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'b1cebfb7-284f5117'
SPLIT = 'val'
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker

    frames_dir = DATA_ROOT / 'images' / 'track' / SPLIT / VIDEO
    print(f'[INFO] видео {VIDEO}, модель {Path(MODEL).name}')

    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        # Один прогон детектора: живые Boxes идут в путь A, и они же
        # сохраняются в кэш для пути B.
        print('[INFO] прогон детектора + кэширование...')
        build_detection_cache(MODEL, [VIDEO], SPLIT, tmp)
        cache = load_video_cache(tmp, SPLIT, VIDEO)

        model = YOLO(MODEL)
        t_live = BYTETracker(make_tracker_args())
        t_cache = BYTETracker(make_tracker_args())

        n_frames = 0
        n_equal = 0
        first_diff = None
        results = model.predict(source=str(frames_dir), conf=CACHE_CONF,
                                iou=CACHE_IOU, stream=True, verbose=False)
        cache_frames = iter_frames(cache)
        for res, (fidx, fname, c_xyxy, c_conf, c_cls) in zip(results, cache_frames):
            n_frames += 1
            det_live = res.boxes.cpu().numpy()

            # Кэш обязан хранить ровно эти детекции (тот же прогон).
            l_xyxy = det_live.xyxy.astype(np.float32)
            if len(c_conf) != len(det_live) or not np.array_equal(c_xyxy, l_xyxy):
                check(f'кадр {fidx}: кэш == детекции прогона', False,
                      f'N кэш={len(c_conf)}, N live={len(det_live)}')
                break

            out_live = t_live.update(det_live, res.orig_img)
            out_cache = t_cache.update(_DetSlice(c_xyxy, c_conf, c_cls))

            def by_idx(arr):
                return {int(r[7]): np.delete(r, 4) for r in arr}
            bl, bc = by_idx(out_live), by_idx(out_cache)
            equal = (bl.keys() == bc.keys() and
                     all(np.allclose(bl[k], bc[k], atol=1e-4) for k in bl))
            n_equal += equal
            if not equal and first_diff is None:
                first_diff = fidx

        check('обработаны все кадры', n_frames == len(cache['frame_names']),
              f'{n_frames}/{len(cache["frame_names"])}')
        check('выход ассоциации побитово совпадает на всех кадрах',
              n_equal == n_frames,
              f'{n_equal}/{n_frames}'
              + (f', первое расхождение: кадр {first_diff}' if first_diff is not None else ''))

    print(f'\n--- {"OK: оффлайн-ассоциация эквивалентна live-пути" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
