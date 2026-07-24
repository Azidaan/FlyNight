"""Супервизор обучения: автодетект зависания и перезапуск с --resume.

Мотивация: 04-05.07.2026 baseline завис в дедлоке DataLoader (Windows,
риск Этапа 1 плана) на 14 часов — процесс жив, heartbeat тикает,
но итерации не движутся. Супервизор закрывает этот класс отказов для
всех оставшихся обучений (~13 прогонов Этапов 5-7).

Логика:
- запускает переданную команду обучения как дочерний процесс;
- прогресс = mtime самого свежего last.pt в наблюдаемой директории runs
  (Ultralytics пишет чекпойнт после каждой эпохи, ~15-30 мин);
- грация на старт (скан данных + прогрев + первая эпоха) — отдельный,
  более длинный таймаут;
- при стагнации > stall_min при живом процессе: убить дерево процессов,
  перезапустить команду с добавленным --resume (до max_restarts раз);
- если процесс завершился сам: код 0 — успех, иначе — перезапуск
  с --resume (падение могло быть OOM/сбоем драйвера).

Использование:
    python scripts/train_supervisor.py --watch runs/baseline ^
        -- python training/train_baseline.py --seed 42
    python scripts/train_supervisor.py --watch runs/drb_full ^
        -- python training/train_drb.py --seed 42

    (всё после `--` — команда обучения; --resume добавляется автоматически
     при перезапусках, если его нет в команде)

Лог супервизора: <watch>/supervisor.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(watch_dir: Path, msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} [SUP] {msg}'
    print(line, flush=True)
    try:
        with open(watch_dir / 'supervisor.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass


def newest_checkpoint_mtime(watch_dir: Path) -> float:
    """mtime самого свежего last.pt под наблюдаемой директорией (0 — нет)."""
    newest = 0.0
    for p in watch_dir.rglob('last.pt'):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def kill_tree(pid: int) -> None:
    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                   capture_output=True)


def run_once(cmd: list[str], watch_dir: Path, stall_min: int,
             startup_grace_min: int, poll_sec: int) -> int:
    """Один запуск под надзором. Возвращает: 0 успех, 1 падение, 2 зависание."""
    log(watch_dir, f'запуск: {" ".join(cmd)}')
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    started = time.time()
    baseline_mtime = newest_checkpoint_mtime(watch_dir)
    saw_progress = False

    while True:
        try:
            # wait с таймаутом: мгновенная реакция на завершение процесса,
            # по истечении таймаута — очередная проверка стагнации.
            rc = proc.wait(timeout=poll_sec)
            log(watch_dir, f'процесс завершился с кодом {rc}')
            return 0 if rc == 0 else 1
        except subprocess.TimeoutExpired:
            pass

        mt = newest_checkpoint_mtime(watch_dir)
        if mt > baseline_mtime:
            baseline_mtime = mt
            saw_progress = True

        ref = mt if saw_progress else max(mt, started)
        stall = (time.time() - ref) / 60
        limit = stall_min if saw_progress else startup_grace_min
        if stall > limit:
            log(watch_dir, f'СТАГНАЦИЯ {stall:.0f} мин (лимит {limit}) — '
                           f'убиваю дерево PID {proc.pid}')
            kill_tree(proc.pid)
            time.sleep(5)
            return 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--watch', required=True,
                    help='директория runs конфигурации (ищется last.pt)')
    ap.add_argument('--stall-min', type=float, default=60,
                    help='минут без нового чекпойнта = зависание (default 60)')
    ap.add_argument('--startup-grace-min', type=float, default=120,
                    help='грация до первого чекпойнта (скан+прогрев+эпоха 1)')
    ap.add_argument('--max-restarts', type=int, default=5)
    ap.add_argument('--poll-sec', type=float, default=120)
    ap.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='-- команда обучения')
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        ap.error('укажите команду обучения после --')

    watch_dir = PROJECT_ROOT / args.watch
    watch_dir.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        attempt += 1
        run_cmd = list(cmd)
        # при перезапусках добавляем --resume (если чекпойнт уже есть)
        if attempt > 1 and '--resume' not in run_cmd \
                and newest_checkpoint_mtime(watch_dir) > 0:
            run_cmd.append('--resume')
        result = run_once(run_cmd, watch_dir, args.stall_min,
                          args.startup_grace_min, args.poll_sec)
        if result == 0:
            log(watch_dir, f'успех с попытки {attempt}')
            return 0
        if attempt > args.max_restarts:
            log(watch_dir, f'исчерпаны {args.max_restarts} перезапусков — сдаюсь')
            return 1
        reason = 'зависание' if result == 2 else 'падение'
        log(watch_dir, f'{reason}; перезапуск {attempt}/{args.max_restarts} '
                       f'через 30 с...')
        time.sleep(30)


if __name__ == '__main__':
    sys.exit(main())
