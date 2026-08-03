"""Smoke-тест логики конвейера панели «Запусти меня» (без GUI).

Проверяет: расчёт статусов шагов по фактам на диске (done/partial/todo,
роль active у первого незавершённого), построение команд с пропуском
готовых сидов и --resume для чекпойнтов.

Запуск: python scripts/smoke/test_launcher_pipeline.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        'launcher', PROJECT_ROOT / 'Запусти меня.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # мы в .venv — reexec пропустится

    api = mod.Api()
    steps = api.pipeline_state()
    print('=== состояние конвейера ===')
    for i, s in enumerate(steps, 1):
        print(f'  {i:02d} {s["role"]:7s} {s["status"]:8s} {s["name"]}'
              f'  [{s["detail"]}]')

    check('11 шагов', len(steps) == 11, str(len(steps)))
    active = [s for s in steps if s['role'] == 'active']
    check('ровно один активный шаг', len(active) == 1,
          active[0]['id'] if active else 'нет')
    check('активный — baseline (эпоха 27 в чекпойнте)',
          active and active[0]['id'] == 'baseline')
    check('baseline: partial (есть чекпойнт resume)',
          active and active[0]['status'] == 'partial', active[0]['detail'])
    idx_active = next(i for i, s in enumerate(steps) if s['role'] == 'active')
    check('до активного — только done', all(
        s['role'] == 'done' for s in steps[:idx_active]))
    check('после активного — только locked', all(
        s['role'] == 'locked' for s in steps[idx_active + 1:]))

    print('=== команды активного шага ===')
    cmds = mod.PIPELINE[0]['cmds']()
    for c in cmds:
        print('  ', ' '.join(c))
    check('3 команды (готовых сидов нет)', len(cmds) == 3, str(len(cmds)))
    check('сид 42 с --resume', '--resume' in cmds[0], ' '.join(cmds[0][-4:]))
    check('сиды 123/2024 без --resume',
          all('--resume' not in c for c in cmds[1:]))
    check('все команды через супервизор',
          all(c[0].endswith('train_supervisor.py') for c in cmds))

    test_detached_run(mod)

    print(f'\n--- {"OK: конвейер панели корректен" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


def test_detached_run(mod) -> None:
    """Автономный прогон живёт отдельным процессом и переживает панель.

    Настоящий run_full_pipeline.py подменяется безобидной заглушкой: она
    пишет строки в журнал и завершается — проверяется сам механизм
    (отдельный процесс, метка PID, чтение журнала), а не обучение.
    """
    import tempfile
    import time

    print('=== автономный прогон (заглушка вместо обучения) ===')
    tmp = Path(tempfile.mkdtemp(prefix='detached_', dir=str(PROJECT_ROOT / 'runs')))
    stub = tmp / 'stub_runner.py'
    stub.write_text(
        'import sys, time\n'
        'from pathlib import Path\n'
        'log = Path(sys.argv[0]).parent / "stub.log"\n'
        'for i in range(3):\n'
        '    with open(log, "a", encoding="utf-8") as f:\n'
        '        f.write(f"строка {i} автономного прогона\\n")\n'
        '    time.sleep(0.6)\n',
        encoding='utf-8')

    orig = (mod.FULL_RUNNER, mod.FULL_LOG, mod.FULL_PID)
    mod.FULL_RUNNER, mod.FULL_LOG = stub, tmp / 'stub.log'
    mod.FULL_PID = tmp / 'stub.pid'
    try:
        api = mod.Api()                       # без окна: _js/_log — пустышки
        check('до запуска прогона нет', api._full_pid() is None)

        api.run_full()
        time.sleep(1.0)
        pid = api._full_pid()
        check('прогон запущен отдельным процессом', pid is not None, str(pid))
        check('метка PID записана', mod.FULL_PID.exists())

        deadline = time.time() + 20
        while api._full_pid() is not None and time.time() < deadline:
            time.sleep(0.5)
        check('прогон завершился сам', api._full_pid() is None)
        lines = mod.FULL_LOG.read_text(encoding='utf-8').splitlines()
        check('журнал прогона наполнялся', len(lines) == 3, f'{len(lines)} строк')

        time.sleep(2.5)                        # даём хвостовому потоку прибраться
        check('метка PID убрана после завершения', not mod.FULL_PID.exists())

        mod.FULL_PID.write_text('999999', encoding='utf-8')
        check('устаревшая метка PID не считается живой', api._full_pid() is None)
    finally:
        mod.FULL_RUNNER, mod.FULL_LOG, mod.FULL_PID = orig
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
