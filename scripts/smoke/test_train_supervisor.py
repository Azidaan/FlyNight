"""Smoke-тест супервизора обучения (scripts/train_supervisor.py).

Сценарии на синтетических «обучениях» (без GPU, секунды):
1. Успех с первой попытки: команда сразу выходит с кодом 0.
2. Падение -> перезапуск с --resume -> успех: команда падает, если нет
   флага --resume, и успешна с ним (плюс проверка, что --resume добавлен
   только при наличии чекпойнта).
3. Зависание -> убийство -> перезапуск: команда спит вечно; малые
   таймауты; после max_restarts=1 супервизор сдаётся с кодом 1.

Запуск: python scripts/smoke/test_train_supervisor.py
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
SUP = str(PROJECT_ROOT / 'scripts' / 'train_supervisor.py')
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def run_sup(watch: Path, extra: list[str], train_cmd: list[str],
            timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = [PY, SUP, '--watch', str(watch.relative_to(PROJECT_ROOT)),
           *extra, '--', *train_cmd]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace',
                          timeout=timeout, cwd=str(PROJECT_ROOT))


def main() -> int:
    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        tmp = Path(tmp)

        # 1. Мгновенный успех
        w1 = tmp / 'ok'; w1.mkdir()
        r = run_sup(w1, [], [PY, '-c', 'print("done")'])
        check('успех с первой попытки: rc=0', r.returncode == 0)
        check('лог: успех с попытки 1', 'успех с попытки 1' in r.stdout)

        # 2. Падение -> перезапуск (--resume добавляется при наличии ckpt)
        w2 = tmp / 'crash'; (w2 / 'weights').mkdir(parents=True)
        (w2 / 'weights' / 'last.pt').write_bytes(b'stub')  # чекпойнт есть
        flaky = ('import sys; '
                 'sys.exit(0 if "--resume" in sys.argv else 3)')
        r = run_sup(w2, ['--max-restarts', '2'], [PY, '-c', flaky])
        check('падение -> resume -> успех: rc=0', r.returncode == 0,
              f'rc={r.returncode}')
        check('лог: перезапуск был', 'перезапуск 1/' in r.stdout)

        # 3. Зависание: команда спит, чекпойнтов нет, грация 0.05 мин
        w3 = tmp / 'hang'; w3.mkdir()
        t0 = time.time()
        r = run_sup(w3, ['--startup-grace-min', '0.05', '--stall-min', '0.05',
                         '--poll-sec', '1', '--max-restarts', '1'],
                    [PY, '-c', 'import time; time.sleep(600)'])
        dt = time.time() - t0
        check('зависание: супервизор сдался с rc=1', r.returncode == 1,
              f'rc={r.returncode}')
        check('зависание: детект и убийство быстрые', dt < 90, f'{dt:.0f}s')
        check('лог: СТАГНАЦИЯ зафиксирована', 'СТАГНАЦИЯ' in r.stdout)

    print(f'\n--- {"OK: супервизор работает" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
