"""Снимок результатов в отдельную папку — кнопка «Сохранить результаты».

Копируется только незаменимое: обученные веса, чекпойнты, кривые обучения и
посчитанные таблицы. Всё, что пересчитывается из них за минуты — картинки,
предсказания, кэши, датасет — не копируется, иначе снимок разрастается на
десятки гигабайт и его перестают делать.

Что именно и почему: код и конфиги лежат в git, а `weights/`, `runs/` и
`notes/` в .gitignore — их не восстановит никакой `git clone`. Снимок
закрывает ровно эту дыру.

Рядом со снимком кладётся ВОЗВРАТ.md — куда что положить, чтобы проект ожил
на новой машине, — и ОПИСЬ.csv со списком файлов и их контрольными суммами.

Запуск: python scripts/backup_results.py --dest "D:/Копии"
"""
import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Целиком: маленькие и незаменимые. Каталог dissertation и сводный план
# лежат в .gitignore — по требованию в репозиторий заливается только код, —
# поэтому никакой git их не восстановит, и снимок для них единственная
# страховка. Весят они около сотни килобайт.
WHOLE_DIRS = ['weights', 'experiments/results', 'notes', 'dissertation']
WHOLE_FILES = ['СВОДНЫЙ_ПЛАН1.md']
# Из runs/ берём выборочно. Модели (.pt) — всегда: их не восстановить
# ничем, кроме повторного обучения на сутки. Остальное — только если файл
# не разросся: дампы предсказаний из оценки занимают там 3.6 ГБ и заново
# считаются из весов за час, а хвосты журналов не стоят гигабайтов.
RUNS_KEEP_SUFFIX = ('.pt', '.csv', '.yaml', '.json', '.log')
MAX_OTHER_BYTES = 20 * 1024 * 1024
SKIP_RUN_DIRS = ('smoke', 'eval_smoke', 'detect')


def _iter_files() -> list[tuple[Path, Path]]:
    """[(откуда, относительный путь)] — что попадёт в снимок."""
    out: list[tuple[Path, Path]] = []
    for d in WHOLE_DIRS:
        root = PROJECT_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file():
                out.append((p, p.relative_to(PROJECT_ROOT)))
    for f in WHOLE_FILES:
        p = PROJECT_ROOT / f
        if p.is_file():
            out.append((p, p.relative_to(PROJECT_ROOT)))
    runs = PROJECT_ROOT / 'runs'
    if runs.exists():
        for p in runs.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in RUNS_KEEP_SUFFIX:
                continue
            rel = p.relative_to(PROJECT_ROOT)
            if any(part in SKIP_RUN_DIRS for part in rel.parts):
                continue                      # прогоны smoke-проверок
            if p.suffix.lower() != '.pt' and p.stat().st_size > MAX_OTHER_BYTES:
                continue                      # дампы предсказаний, толстые логи
            out.append((p, rel))
    return sorted(out, key=lambda t: str(t[1]).lower())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


README = """# Как вернуть эти результаты в проект

Снимок сделан {stamp}. Внутри — то, чего нет в git: обученные веса,
чекпойнты прогонов, кривые обучения и посчитанные таблицы.

## Порядок

1. Разверни сам проект из git:
   `git clone https://gitlab.com/aziteam-group/drbdiss.git E:/Dissertation`
2. Скопируй **содержимое папки `проект`** из этого снимка в корень проекта,
   с заменой. Папки `weights`, `runs`, `experiments`, `notes` встанут на свои
   места — структура внутри снимка уже такая же, как в проекте.
3. Датасет BDD100K снимок не содержит: он весит около 45 ГБ и качается
   заново. Положи его в `data/` как раньше.
4. Проверь, что всё на месте: `powershell -File scripts/smoke/run_all_smoke.ps1 -CpuOnly`
   и открой панель «Запусти меня.py» — пройденные этапы должны подсветиться
   зелёным сами, они определяются по файлам на диске.

## Что где лежит

- `проект/weights/<имя прогона>_seed<N>/best.pt` — итоговые модели. Именно их
  берут оценка по поднаборам, трекинг и замер скорости.
- `проект/runs/<имя прогона>/seed<N>/` — ход обучения: `results.csv` (метрики
  по эпохам), `args.yaml` (с какими параметрами шло), `weights/last.pt`
  (чекпойнт, с него можно продолжить прерванное обучение).
- `проект/experiments/results/` — посчитанные таблицы для текста диссертации.
- `проект/dissertation/` — черновики глав и хроника работы над блоком DRB.
  Этого нет в git: в репозиторий заливается только код. Восстанавливается
  единственным способом — отсюда.
- `проект/СВОДНЫЙ_ПЛАН1.md` — план работы по этапам, тоже только отсюда.

## Проверка целостности

`ОПИСЬ.csv` содержит размер и контрольную сумму каждого файла. Если есть
сомнения, что копия дошла целой, сверься по ней.

Всего файлов: {count}, объём: {size}.
"""


def human(n: int) -> str:
    for unit in ('Б', 'КБ', 'МБ', 'ГБ'):
        if n < 1024 or unit == 'ГБ':
            return f'{n:.1f} {unit}' if unit != 'Б' else f'{n} {unit}'
        n /= 1024
    return f'{n:.1f} ГБ'


def backup(dest_root: Path, log=print) -> Path:
    """Делает снимок в dest_root/DRB_снимок_<дата>. Возвращает путь снимка."""
    files = _iter_files()
    if not files:
        raise SystemExit('нечего сохранять: ни весов, ни прогонов не найдено')
    total = sum(src.stat().st_size for src, _ in files)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    snap, k = dest_root / f'DRB_снимок_{stamp}', 2
    while snap.exists():      # два снимка в одну минуту не должны сливаться
        snap, k = dest_root / f'DRB_снимок_{stamp}_{k}', k + 1
    log(f'Снимок: {len(files)} файлов, {human(total)} → {snap}')

    rows, done, step = [], 0, max(1, len(files) // 10)
    for i, (src, rel) in enumerate(files, 1):
        dst = snap / 'проект' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        done += size
        rows.append([rel.as_posix(), size, _sha256(dst)])
        if i % step == 0 or i == len(files):
            log(f'  скопировано {i}/{len(files)} ({human(done)})')

    with open(snap / 'ОПИСЬ.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['файл', 'байт', 'sha256'])
        w.writerows(rows)
    (snap / 'ВОЗВРАТ.md').write_text(
        README.format(stamp=datetime.now().strftime('%d.%m.%Y в %H:%M'),
                      count=len(files), size=human(total)),
        encoding='utf-8')
    log(f'Готово: {snap}')
    return snap


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dest', help='куда положить снимок')
    ap.add_argument('--list', action='store_true',
                    help='только показать состав и объём, ничего не копировать')
    args = ap.parse_args()
    if args.list:
        files = _iter_files()
        total = sum(s.stat().st_size for s, _ in files)
        print(f'{len(files)} файлов, {human(total)}')
        return 0
    if not args.dest:
        ap.error('нужен --dest (куда сохранять) или --list')
    backup(Path(args.dest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
