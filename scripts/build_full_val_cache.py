"""Кэш детекций контрольной модели на ПОЛНОМ val BDD100K MOT (200 видео).

Зачем. Все замеры проекта до 10.09 сделаны на 50 видео из 200. Для классов
с малым числом объектов классоусреднённые метрики (mMOTA, mIDF1, mHOTA) на
четверти выборки смещены неизвестным образом, и сравнение с литературой
некорректно: опубликованные числа BDD100K считаются на всех 200 видео.

Стоимость измерена заранее: 50 видео на разрешении 960 собрались за 5 мин 32 с
(`runs/daycache.log` и метки времени кэша), то есть около 6.6 с на видео.
Оставшиеся 150 видео — примерно 17 минут.

Кэш достраивается: `build_detection_cache` пропускает видео, для которых npz
уже есть, поэтому готовые 50 не пересчитываются. Путь к модели передаётся
ровно той же строкой, что записана в `meta.json` — иначе проверка меты сочтёт
модель другой и откажется дописывать.

Запуск:
    python scripts/build_full_val_cache.py
    python scripts/build_full_val_cache.py --imgsz 640 --cache runs/det_cache_ctl
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# строка модели должна совпадать с записанной в meta.json существующего кэша
MODEL = 'weights/baseline_seed42/best.pt'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='runs/det_cache_ctl_960')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0, help='0 — все видео сплита')
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    from tracking.detection_cache import build_detection_cache

    if not (PROJECT_ROOT / MODEL).exists():
        raise SystemExit(f'нет весов контроля: {MODEL}')

    img_root = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / args.split
    vids = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if args.limit:
        vids = vids[:args.limit]

    cache_dir = PROJECT_ROOT / args.cache
    done = {p.stem for p in (cache_dir / args.split).glob('*.npz')}
    todo = [v for v in vids if v not in done]

    print(f'модель: {MODEL}')
    print(f'разрешение: {args.imgsz}')
    print(f'кэш: {cache_dir}')
    print(f'видео в сплите {args.split}: {len(vids)}')
    print(f'уже собрано: {len(done)}')
    print(f'осталось собрать: {len(todo)}')
    print(f'ожидаемое время при 6.6 с на видео: {len(todo) * 6.6 / 60:.0f} мин\n',
          flush=True)

    if not todo:
        print('[OK ] кэш уже полон, работы нет')
        return 0

    out = build_detection_cache(MODEL, vids, args.split, cache_dir,
                                imgsz=args.imgsz)
    n = len(list(Path(out).glob('*.npz')))
    print(f'\n[OK ] кэш: {out}, видео в нём: {n}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
