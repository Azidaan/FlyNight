"""Smoke-проверка evaluation-модулей ДО запуска настоящего baseline (Этап 3).

Использует мини-модель из runs/smoke_stage3 (2 эпохи на mini-BDD) — метрики
будут низкие, это нормально: проверяется работоспособность конвейера, не качество.

1. Детекция: evaluate_on_subsets на normal+night с прореживанием x50
   (~150-300 кадров на поднабор) — общие метрики, per-class, AP по размерам.
2. Трекинг: ByteTrack на 2 val-видео + официальная оценка bdd100k (HOTA/IDF1/MOTA).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.subset_filter import filter_videos_by_subset, load_video_attributes  # noqa: E402
from evaluation.detection_metrics import evaluate_on_subsets  # noqa: E402
from evaluation.stress_delta import compute_stress_delta  # noqa: E402
from evaluation.tracking_metrics import evaluate_tracking_bdd, run_tracker_on_videos  # noqa: E402

MODEL = PROJECT_ROOT / 'runs' / 'smoke_stage3' / 'seed42' / 'weights' / 'best.pt'


def main() -> int:
    if not MODEL.exists():
        print(f'[FAIL] Нет мини-модели {MODEL}')
        return 1
    failures = []

    print('=== 1. Детекция по поднаборам (smoke, step 50) ===', flush=True)
    det = evaluate_on_subsets(str(MODEL), subsets=['normal', 'night'],
                              frame_step=50,
                              workdir=PROJECT_ROOT / 'runs' / 'eval_smoke')
    for subset, m in det.items():
        keys = ('mAP50', 'mAP50_95', 'precision', 'recall', 'AP_small',
                'AP_medium', 'AP_large', 'AP50_car')
        present = [k for k in keys if k in m]
        missing = [k for k in keys if k not in m]
        print(f'[{"OK " if not missing else "FAIL"}] {subset}: '
              + ', '.join(f'{k}={m[k]:.3f}' for k in present))
        if missing:
            failures.append(f'{subset}: нет метрик {missing}')
    delta = compute_stress_delta(det.get('normal', {}), det.get('night', {}))
    print(f'[OK ] Δ(night) посчитана: mAP50 Δ={delta.get("mAP50", float("nan")):.3f}')

    print('\n=== 2. Трекинг + официальная оценка BDD (2 видео) ===', flush=True)
    det_json = PROJECT_ROOT / 'data' / 'bdd100k' / 'labels' / 'bdd100k_labels_images_val.json'
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(p.name for p in
                     (PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val').iterdir()
                     if p.is_dir())
    vids = filter_videos_by_subset(attrs, all_ids, 'normal')[:2]
    track_dir = PROJECT_ROOT / 'runs' / 'track_smoke'
    run_tracker_on_videos(str(MODEL), vids, track_dir)
    try:
        summary = evaluate_tracking_bdd(
            track_dir, vids, PROJECT_ROOT / 'runs' / 'track_smoke' / 'eval.json')
        print(f'[OK ] метрики трекинга: {summary}')
        if not any(k in summary for k in ('HOTA', 'IDF1', 'MOTA', 'mHOTA', 'mIDF1', 'mMOTA')):
            failures.append(f'в сводке трекинга нет ключевых метрик: {list(summary)[:10]}')
    except Exception as exc:  # noqa: BLE001 — smoke должен дожить до вердикта
        print(f'[FAIL] оценка трекинга: {type(exc).__name__}: {exc}')
        failures.append(f'tracking eval: {exc}')

    if failures:
        print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
        for f in failures:
            print(f'  [FAIL] {f}')
        return 1
    print('\n--- OK: конвейер оценки работоспособен, baseline можно запускать ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
