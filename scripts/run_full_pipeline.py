"""Автономный прогон всего конвейера обучений с проверками (Этапы 3-10).

Один процесс проходит все незавершённые шаги подряд: обучения, оценки,
ablation, трекинг, финальные таблицы. После каждого шага — проверки
адекватности из scripts/sanity_checks.py. Находка уровня fail
останавливает прогон: дальше считать нечего, все последующие шаги
опирались бы на негодный результат.

Определение шагов не дублируется — PIPELINE берётся из панели
«Запусти меня.py», так что кнопки в интерфейсе и автономный прогон
всегда делают одно и то же.

Всё пишется одновременно в stdout (панель показывает поток в журнале)
и в runs/full_pipeline.log с отметками времени.

Использование:
    python scripts/run_full_pipeline.py                 # всё, что не готово
    python scripts/run_full_pipeline.py --dry-run       # план без запуска
    python scripts/run_full_pipeline.py --check-only    # только проверки
    python scripts/run_full_pipeline.py --from drb      # начиная с шага
    python scripts/run_full_pipeline.py --only track_grid track_modes
    python scripts/run_full_pipeline.py --skip-checks   # без проверок
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sanity_checks import (  # noqa: E402
    check_step, format_findings, worst_level,
)

VENV_PY = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
LOG_PATH = PROJECT_ROOT / 'runs' / 'full_pipeline.log'


def load_pipeline() -> list[dict]:
    """PIPELINE из панели «Запусти меня.py» (единственный источник правды)."""
    spec = importlib.util.spec_from_file_location(
        'launcher_pipeline', PROJECT_ROOT / 'Запусти меня.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PIPELINE


class Journal:
    """Печать в stdout и дозапись в лог-файл с отметками времени."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write(f'\n{"=" * 70}\n'
                   f'НОВЫЙ ПРОГОН {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                   f'{"=" * 70}', stamp=False)

    def write(self, text: str, stamp: bool = True) -> None:
        line = f'[{datetime.now():%d.%m %H:%M:%S}] {text}' if stamp else text
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def raw(self, text: str) -> None:
        """Строка подпроцесса: в stdout как есть, в файл — тоже."""
        print(text, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


def run_command(argv: list[str], jr: Journal) -> int:
    """Запуск с построчной трансляцией вывода. Возвращает код возврата."""
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
    proc = subprocess.Popen(
        argv, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8',
        errors='replace', bufsize=1, env=env)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip('\n').split('\r')[-1].rstrip()
        if line:
            jr.raw(line)
    proc.wait()
    return proc.returncode or 0


def run_checks(step_id: str, jr: Journal) -> str:
    """Проверки шага. Возвращает 'ok' | 'warn' | 'fail'."""
    findings = check_step(step_id)
    if not findings:
        jr.write(f'проверок для шага «{step_id}» не предусмотрено')
        return 'ok'
    jr.write(f'проверки шага «{step_id}»:')
    for line in format_findings(findings):
        jr.raw('    ' + line)
    level = worst_level(findings)
    jr.write({'ok': 'проверки пройдены',
              'warn': 'проверки пройдены с замечаниями — продолжаю',
              'fail': 'ПРОВЕРКИ ПРОВАЛЕНЫ — прогон останавливается'}[level])
    return level


def human_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f'{h} ч {m} мин' if h else (f'{m} мин {s} с' if m else f'{s} с')


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')            # консоль Windows — cp866

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true', help='показать план и выйти')
    ap.add_argument('--check-only', action='store_true',
                    help='только проверки готовых шагов, без запусков')
    ap.add_argument('--skip-checks', action='store_true', help='без проверок')
    ap.add_argument('--from', dest='from_step', help='начать с этого шага')
    ap.add_argument('--only', nargs='+', help='только указанные шаги')
    args = ap.parse_args()

    pipeline = load_pipeline()
    ids = [s['id'] for s in pipeline]
    for name in (args.only or []) + ([args.from_step] if args.from_step else []):
        if name not in ids:
            print(f'Неизвестный шаг: {name}. Доступны: {", ".join(ids)}')
            return 2

    selected = pipeline
    if args.from_step:
        selected = pipeline[ids.index(args.from_step):]
    if args.only:
        selected = [s for s in pipeline if s['id'] in args.only]

    jr = Journal(LOG_PATH)

    if args.dry_run or args.check_only:
        jr.write('режим: ' + ('только проверки' if args.check_only else 'план без запуска'))
        worst = 'ok'
        for step in selected:
            status, detail = step['status']()
            jr.write(f'{step["id"]:14s} {status:8s} {step["name"]} [{detail}]')
            if args.check_only and status == 'done':
                level = run_checks(step['id'], jr)
                if level == 'fail':
                    worst = 'fail'
                elif level == 'warn' and worst == 'ok':
                    worst = 'warn'
        jr.write(f'итог: {worst.upper()}')
        return 1 if worst == 'fail' else 0

    started = time.time()
    todo = [s for s in selected if s['status']()[0] != 'done']
    jr.write(f'к выполнению шагов: {len(todo)} из {len(selected)}')
    for step in todo:
        jr.write(f'  · {step["id"]} — {step["name"]} ({step["time"]})')

    for step in todo:
        step_started = time.time()
        status, detail = step['status']()
        jr.write(f'{"─" * 60}')
        jr.write(f'ШАГ «{step["name"]}» (статус {status}, {detail})')
        cmds = step['cmds']()
        if not cmds:
            jr.write('команд нет — шаг считается выполненным')
        for i, cmd in enumerate(cmds, 1):
            argv = [str(VENV_PY), str(PROJECT_ROOT / cmd[0]), *cmd[1:]]
            jr.write(f'команда {i}/{len(cmds)}: {" ".join(cmd)}')
            code = run_command(argv, jr)
            if code != 0:
                jr.write(f'команда завершилась с кодом {code} — прогон остановлен')
                jr.write(f'проработано {human_time(time.time() - started)}')
                return 1

        new_status, new_detail = step['status']()
        if new_status != 'done':
            jr.write(f'шаг отработал, но статус остался «{new_status}» ({new_detail}) '
                     f'— прогон остановлен, нужен разбор')
            return 1
        jr.write(f'шаг завершён за {human_time(time.time() - step_started)}')

        if not args.skip_checks and run_checks(step['id'], jr) == 'fail':
            jr.write(f'проработано {human_time(time.time() - started)}')
            return 2

    jr.write(f'{"─" * 60}')
    jr.write(f'ВЕСЬ КОНВЕЙЕР ПРОЙДЕН за {human_time(time.time() - started)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
