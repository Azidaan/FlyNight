"""Метки YOLO с закодированным возрастом объекта в его траектории.

Замысел. Обучению нужен признак «насколько объект нов в сцене», но формат меток
YOLO — ровно пять чисел в строке, и номера трека там нет. Добавить шестую
колонку нельзя: Ultralytics читает строку из шести и более чисел как полигон
сегментации. Поднять число классов тоже нельзя — это изменит размер головы
детектора и смешает метод с архитектурой.

Решение: признак кодируется ДРОБНОЙ частью класса.

    класс_в_файле = класс + 0.1 * (корзина + 1)

Корзины возраста: 0 → возраст 0 (первое появление), 1 → возраст 1,
2 → возраст 2, 3 → возраст 3-4, 4 → возраст 5 и старше.

Почему это работает. Проверка `verify_image_label` требует `max_cls < nc`;
максимум здесь 7 + 0.5 = 7.5 < 8, проверка проходит. Массив меток и так
читается как float, аугментации переносят его как есть, мозаика тащит признак
вместе с объектом. Целая часть снимается в функции потерь одной операцией до
того, как метку увидит назначатель целей.

Почему корзина, а не флаг. Доля найденных растёт по возрасту монотонно
(38.00 → 45.36 → 49.86 → 55.31 → 63.02 → 70.98 ночью на стенде 960), порога
нет. Корзины позволяют задать вес убывающей функцией возраста, а не ступенькой,
и это вдвое увеличивает мишень метода (5289 объектов против 2295).

Почему смещение на 0.1, а не на 0. Метки без признака (обычные целые классы,
например валидационные) дают дробную часть 0 и читаются как «сведений нет,
вес 1». Если бы корзина 0 кодировалась нулём, обычная метка молча считалась бы
первым появлением.

ВАЖНО про прореживание. Возраст считается по ПОЛНОМУ видео, а не по прореженной
обучающей выборке. Если брать каждый пятый кадр, «первая встреча» в выборке не
совпадает с первой встречей в сцене, и метод учился бы на неверном признаке.

Запуск:
    python scripts/make_birth_labels.py --ann data/bdd100k_mot_mini/annotations_json \
        --out data/birth_smoke --images data/bdd100k_mot_mini/images
    python scripts/make_birth_labels.py --ann data/bdd100k/labels/box_track_20/train \
        --out data/bdd100k_birth/train --images data/bdd100k/images/track/train
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

W, H = 1280, 720                  # BDD100K: размер кадра постоянный
CATS = {'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7}
# (нижняя граница, верхняя граница, номер корзины)
AGE_BINS = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 4, 3), (5, 10 ** 9, 4)]
STEP = 0.1                        # шаг кодирования корзины в дробной части


def bucket_of(age: int) -> int:
    for lo, hi, b in AGE_BINS:
        if lo <= age <= hi:
            return b
    return AGE_BINS[-1][2]


def encode(cls: int, bucket: int) -> float:
    """Класс с признаком: целая часть — класс, дробная — корзина возраста."""
    return cls + STEP * (bucket + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True,
                    help='каталог с JSON аннотаций (формат box_track_20)')
    ap.add_argument('--out', required=True,
                    help='корень выходного набора; метки лягут в <out>/labels')
    ap.add_argument('--images', default=None,
                    help='каталог с изображениями; будет создана связь '
                         '<out>/images на него (junction на Windows)')
    ap.add_argument('--limit', type=int, default=0, help='0 — все видео')
    args = ap.parse_args()

    ann_dir = Path(args.ann)
    out_root = Path(args.out)
    lab_root = out_root / 'labels'
    files = sorted(ann_dir.glob('*.json'))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'нет аннотаций в {ann_dir}')

    print(f'аннотации: {ann_dir}')
    print(f'выход: {lab_root}')
    print(f'корзины возраста: {[(lo, hi, b) for lo, hi, b in AGE_BINS]}')
    print(f'кодирование: класс + {STEP} * (корзина + 1)')
    print(f'видео: {len(files)}\n', flush=True)

    per_bucket = Counter()
    n_obj = n_frames = n_tracks = 0
    skipped_cat = Counter()

    for vi, f in enumerate(files, 1):
        data = json.loads(f.read_text(encoding='utf-8'))
        frames = sorted(data, key=lambda x: x['frameIndex'])

        # возраст считается по ПОЛНОМУ видео
        seen = defaultdict(list)
        for fr in frames:
            for l in fr.get('labels') or []:
                if l.get('box2d') and l.get('category') in CATS:
                    seen[l['id']].append(fr['frameIndex'])
        age_of = {(t, fi): k
                  for t, fis in seen.items()
                  for k, fi in enumerate(sorted(fis))}
        n_tracks += len(seen)

        vid = f.stem
        out_dir = lab_root / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        for fr in frames:
            lines = []
            for l in fr.get('labels') or []:
                b, c = l.get('box2d'), l.get('category')
                if not b:
                    continue
                if c not in CATS:
                    skipped_cat[c] += 1
                    continue
                bkt = bucket_of(age_of[(l['id'], fr['frameIndex'])])
                per_bucket[bkt] += 1
                cx = (b['x1'] + b['x2']) / 2 / W
                cy = (b['y1'] + b['y2']) / 2 / H
                bw = (b['x2'] - b['x1']) / W
                bh = (b['y2'] - b['y1']) / H
                if bw <= 0 or bh <= 0:
                    continue
                lines.append('%.4f %.6f %.6f %.6f %.6f'
                             % (encode(CATS[c], bkt), cx, cy, bw, bh))
                n_obj += 1
            name = Path(fr.get('name') or f'{vid}-{fr["frameIndex"]:07d}.jpg').stem
            (out_dir / f'{name}.txt').write_text('\n'.join(lines) + '\n'
                                                if lines else '', encoding='utf-8')
            n_frames += 1

        if vi % 100 == 0:
            print(f'[INFO] {vi}/{len(files)} видео', flush=True)

    print(f'\nвидео обработано: {len(files)}')
    print(f'кадров: {n_frames}')
    print(f'треков: {n_tracks}')
    print(f'объектов: {n_obj}')
    if skipped_cat:
        print(f'пропущено по категории: {dict(skipped_cat)}')
    print('\nраспределение по корзинам возраста:')
    names = {0: 'возраст 0', 1: 'возраст 1', 2: 'возраст 2',
             3: 'возраст 3-4', 4: 'возраст 5+'}
    for b in sorted(per_bucket):
        share = 100.0 * per_bucket[b] / n_obj if n_obj else 0.0
        print('  корзина %d (%-11s) объектов %8d  доля %6.2f%%  '
              'дробная часть %.1f' % (b, names[b], per_bucket[b], share,
                                      STEP * (b + 1)))
    young = per_bucket[0] + per_bucket[1]
    print(f'\nдоля возраста 0-1 (мишень метода): '
          f'{100.0 * young / n_obj if n_obj else 0:.2f}%')

    if args.images:
        link = out_root / 'images'
        src = Path(args.images).resolve()
        if link.exists():
            print(f'\n[INFO] {link} уже существует, связь не создаю')
        else:
            out_root.mkdir(parents=True, exist_ok=True)
            import subprocess
            r = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(src)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(f'\n[OK ] связь {link} -> {src}')
            else:
                print(f'\n[ОШИБКА] mklink: {r.stdout}{r.stderr}')
                print('       создайте связь вручную или скопируйте изображения')

    print(f'\n[OK ] метки: {lab_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
