"""Smoke-тест снимка результатов (кнопка «Сохранить результаты»).

Проверяет отбор файлов и содержимое снимка на игрушечном проекте: настоящие
веса и прогоны при этом не читаются и никуда не копируются.

Запуск: python scripts/smoke/test_backup.py
"""
import csv
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def make_fake_project(root: Path) -> None:
    """Игрушечный проект: по файлу каждого вида, который встречается в жизни."""
    def put(rel: str, data: bytes) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    put('weights/baseline_seed42/best.pt', b'M' * 2048)
    put('weights/pretrained/yolo26s.pt', b'P' * 1024)
    put('experiments/results/baseline_metrics.csv', 'сид,значение\n42,0.47\n'.encode())
    put('notes/tracker_choice.md', 'выбран ByteTrack\n'.encode())
    put('runs/baseline/seed42/results.csv', 'epoch,mAP50\n1,0.31\n'.encode())
    put('runs/baseline/seed42/args.yaml', 'epochs: 100\n'.encode())
    put('runs/baseline/seed42/weights/last.pt', b'L' * 4096)
    put('runs/baseline/seed42/train_batch0.jpg', b'J' * 512)      # картинка
    put('runs/eval/normal/predictions.json', b'{' + b'x' * (21 << 20))  # дамп
    put('runs/baseline/seed42/heartbeat.json', b'{"epoch": 7}')   # мелкий json
    put('runs/smoke/seed42/weights/best.pt', b'S' * 1024)         # прогон теста


def main() -> int:
    from scripts import backup_results as mod

    src = Path(tempfile.mkdtemp(prefix='backup_src_'))
    dest = Path(tempfile.mkdtemp(prefix='backup_dst_'))
    real_root = mod.PROJECT_ROOT
    mod.PROJECT_ROOT = src
    check('тест работает на игрушечном проекте, не на настоящем',
          mod.PROJECT_ROOT != real_root)
    try:
        make_fake_project(src)
        picked = {rel.as_posix() for _, rel in mod._iter_files()}

        check('веса модели попадают в снимок',
              'weights/baseline_seed42/best.pt' in picked)
        check('чекпойнт прогона попадает в снимок',
              'runs/baseline/seed42/weights/last.pt' in picked)
        check('кривая обучения попадает в снимок',
              'runs/baseline/seed42/results.csv' in picked)
        check('параметры прогона попадают в снимок',
              'runs/baseline/seed42/args.yaml' in picked)
        check('посчитанные таблицы попадают в снимок',
              'experiments/results/baseline_metrics.csv' in picked)
        check('заметки попадают в снимок', 'notes/tracker_choice.md' in picked)
        check('мелкий служебный json остаётся',
              'runs/baseline/seed42/heartbeat.json' in picked)

        check('картинки прогона не копируются (перерисуются из весов)',
              'runs/baseline/seed42/train_batch0.jpg' not in picked)
        check('дамп предсказаний не копируется (пересчитается из весов)',
              'runs/eval/normal/predictions.json' not in picked,
              'иначе снимок разрастается на гигабайты')
        check('прогоны smoke-проверок не копируются',
              'runs/smoke/seed42/weights/best.pt' not in picked)

        snap = mod.backup(dest, log=lambda s: None)
        check('снимок создан', snap.exists(), snap.name)
        check('файлы лежат под «проект», как в самом проекте',
              (snap / 'проект' / 'weights' / 'baseline_seed42' / 'best.pt').exists())
        check('рядом инструкция по возврату', (snap / 'ВОЗВРАТ.md').exists())
        readme = (snap / 'ВОЗВРАТ.md').read_text(encoding='utf-8')
        check('в инструкции сказано, что датасет качается заново',
              'BDD100K' in readme and '45' in readme)
        check('в инструкции есть команда клонирования проекта',
              'git clone' in readme)

        opis = list(csv.DictReader(
            open(snap / 'ОПИСЬ.csv', encoding='utf-8')))
        check('опись перечисляет все файлы', len(opis) == len(picked),
              f'{len(opis)} и {len(picked)}')
        row = next(r for r in opis if r['файл'].endswith('best.pt'))
        real = hashlib.sha256(
            (snap / 'проект' / row['файл']).read_bytes()).hexdigest()
        check('контрольная сумма в описи сходится', row['sha256'] == real)

        second = mod.backup(dest, log=lambda s: None)
        check('повторный снимок не затирает прошлый', second != snap
              and snap.exists(), f'{snap.name} и {second.name}')
    finally:
        mod.PROJECT_ROOT = real_root
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)

    print(f'\n--- {"OK: снимок собирается верно" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
