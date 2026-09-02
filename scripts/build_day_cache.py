"""Кэш детекций контрольной модели на дневных и ночных видео — для сравнения.

Зачем. Замер устойчивости назначения (check_assign_stability.py) упёрся в две
дыры: в готовом q-кэше нет дневных видео, и построен он моделью С БЛОКОМ, а не
контролем. Сравнивать не с чем.

Здесь строится кэш КОНТРОЛЬНОЙ модели на равном числе дневных и ночных видео.
После него сравнение честное: одна модель, два условия.

Про конфаундер, который замером не снимается. Кадры кэша идут с шагом 5, между
соседними записями примерно 1/6 секунды, и объекты за это время реально
смещаются. Значит доля несопоставленных — это движение ПЛЮС нестабильность
назначения. Разделять их приходится сравнением условий, а не абсолютным
значением. В тексте это оговаривать обязательно.

Запуск: python scripts/build_day_cache.py --videos 25
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS = PROJECT_ROOT / 'weights' / 'baseline_seed42' / 'best.pt'
CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=int, default=25,
                    help='сколько видео на каждое условие')
    ap.add_argument('--conds', nargs='+', default=['normal', 'night'])
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from tracking.detection_cache import build_detection_cache

    if not WEIGHTS.exists():
        raise SystemExit(f'нет весов контроля: {WEIGHTS}')

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    img_root = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
    all_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())

    picked: list[str] = []
    for cond in args.conds:
        vids = filter_videos_by_subset(attrs, all_ids, cond)[:args.videos]
        print(f'[INFO] {cond}: {len(vids)} видео', flush=True)
        picked += vids

    print(f'[INFO] всего {len(picked)} видео, модель {WEIGHTS.name}', flush=True)
    out = build_detection_cache(str(WEIGHTS), picked, 'val', CACHE)
    print(f'[OK ] кэш: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
