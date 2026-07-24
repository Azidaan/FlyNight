"""Проверка целостности сконвертированной YOLO-разметки (перепроверка Этапа 2).

1. Число txt-файлов == числу jpg-кадров в каждом сплите (train, val).
2. Число видео-папок совпадает в images и labels.
3. Выборочная валидация формата: 1000 случайных txt на сплит —
   5 полей в строке, класс в [0..7], координаты в [0, 1], w/h > 0.
4. mini-BDD: 20 видео, для каждого счётчики jpg == txt, манифест согласован.
"""
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / 'data' / 'bdd100k'
MINI = PROJECT_ROOT / 'data' / 'bdd100k_mot_mini'
N_SAMPLE = 1000


def count_files(root: Path, suffix: str) -> tuple[int, int]:
    """(число видео-папок, число файлов с суффиксом)."""
    n_dirs, n_files = 0, 0
    for video_dir in root.iterdir():
        if not video_dir.is_dir():
            continue
        n_dirs += 1
        n_files += sum(1 for f in video_dir.iterdir() if f.suffix == suffix)
    return n_dirs, n_files


def validate_txt(path: Path) -> str | None:
    """None если файл корректен, иначе описание проблемы."""
    for ln, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            return f'{path}: строка {ln}: {len(parts)} полей вместо 5'
        cls, cx, cy, w, h = parts
        if not cls.isdigit() or not 0 <= int(cls) <= 7:
            return f'{path}: строка {ln}: класс {cls} вне [0..7]'
        vals = [float(v) for v in (cx, cy, w, h)]
        if not all(0.0 <= v <= 1.0 for v in vals):
            return f'{path}: строка {ln}: координаты вне [0,1]: {vals}'
        if vals[2] <= 0 or vals[3] <= 0:
            return f'{path}: строка {ln}: неположительные w/h'
    return None


def main() -> int:
    rng = random.Random(42)
    failures: list[str] = []

    for split in ('train', 'val'):
        img_root = DATA / 'images' / 'track' / split
        lbl_root = DATA / 'labels' / 'track' / split
        img_dirs, n_jpg = count_files(img_root, '.jpg')
        lbl_dirs, n_txt = count_files(lbl_root, '.txt')
        ok = img_dirs == lbl_dirs and n_jpg == n_txt
        print(f'[{"OK " if ok else "FAIL"}] {split}: видео {img_dirs}/{lbl_dirs} '
              f'(img/lbl), кадров jpg={n_jpg}, txt={n_txt}')
        if not ok:
            failures.append(f'{split}: расхождение счётчиков')

        all_txt = [f for d in lbl_root.iterdir() if d.is_dir()
                   for f in d.glob('*.txt')]
        sample = rng.sample(all_txt, min(N_SAMPLE, len(all_txt)))
        bad = [err for f in sample if (err := validate_txt(f))]
        print(f'[{"OK " if not bad else "FAIL"}] {split}: выборка {len(sample)} txt — '
              f'ошибок {len(bad)}')
        failures.extend(bad[:5])

    manifest = json.loads((MINI / 'manifest.json').read_text(encoding='utf-8'))
    mini_ok = len(manifest) == 20
    for m in manifest:
        vid = m['video_id']
        n_jpg = sum(1 for _ in (MINI / 'images' / vid).glob('*.jpg'))
        n_txt = sum(1 for _ in (MINI / 'labels' / vid).glob('*.txt'))
        if n_jpg != m['frames'] or n_txt != n_jpg:
            mini_ok = False
            failures.append(f'mini-BDD {vid}: jpg={n_jpg}, txt={n_txt}, '
                            f'манифест={m["frames"]}')
    subsets = {m['subset'] for m in manifest}
    print(f'[{"OK " if mini_ok else "FAIL"}] mini-BDD: {len(manifest)} видео, '
          f'условия {sorted(subsets)}, jpg==txt по всем видео: {mini_ok}')

    if failures:
        print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
        for f in failures[:10]:
            print(f'  [FAIL] {f}')
        return 1
    print('\n--- OK: разметка целостна ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
