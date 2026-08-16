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
    # Проверяем ИНВАРИАНТЫ, а не конкретное состояние: конвейер идёт неделями,
    # и тест, приколоченный к «сейчас активен baseline», устаревает сам собой.
    check('активный шаг не пройден',
          active and active[0]['status'] in ('todo', 'partial'),
          active[0]['status'] if active else 'нет')
    idx_active = next(i for i, s in enumerate(steps) if s['role'] == 'active')
    check('до активного — только done', all(
        s['role'] == 'done' for s in steps[:idx_active]))
    check('после активного — только locked', all(
        s['role'] == 'locked' for s in steps[idx_active + 1:]))

    print('=== команды активного шага ===')
    step = next(s for s in mod.PIPELINE if s['id'] == active[0]['id'])
    cmds = step['cmds']()
    for c in cmds:
        print('  ', ' '.join(c))
    check('у активного шага есть команды', len(cmds) > 0, str(len(cmds)))
    check('все пути команд существуют',
          all((PROJECT_ROOT / c[0]).exists() for c in cmds),
          str([c[0] for c in cmds if not (PROJECT_ROOT / c[0]).exists()]))
    train_cmds = [c for c in cmds if c[0].endswith('train_supervisor.py')]
    if train_cmds:
        check('обучения идут через супервизор', True, f'{len(train_cmds)} шт.')
        check('вложенная команда — python со скриптом обучения',
              all('python' in c and any('train_' in a for a in c) for c in train_cmds))
    # --resume ставится ровно тем сидам, у которых есть чекпойнт
    for c in train_cmds:
        seed = c[c.index('--seed') + 1] if '--seed' in c else '?'
        run = c[c.index('--watch') + 1].split('/')[-1]
        ckpt = (PROJECT_ROOT / 'runs' / run / f'seed{seed}' / 'weights' / 'last.pt')
        check(f'сид {seed}: продолжение только при наличии чекпойнта',
              ('--resume' in c) == ckpt.exists(),
              f'чекпойнт {"есть" if ckpt.exists() else "нет"}')

    test_probes(mod)
    test_detached_run(mod)

    print(f'\n--- {"OK: конвейер панели корректен" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


def test_probes(mod) -> None:
    """Проба инициализации DRB: состояние по диску, команды, сброс."""
    print('=== проба инициализации DRB (узел Е) ===')
    api = mod.Api()
    st = api.probe_state()
    check('три варианта пробы', len(st) == 3, str([p['id'] for p in st]))
    check('стартовые веса 0 / -2 / -4',
          [p['w'] for p in mod.PROBES] == [0.0, -2.0, -4.0],
          str([p['w'] for p in mod.PROBES]))

    cmds = mod._probe_cmds('w4', -4.0)
    line = ' '.join(cmds[0])
    check('команда идёт через супервизор', cmds[0][0].endswith('train_supervisor.py'))
    check('передан стартовый вес', '--alpha-init -4.0' in line, line[-60:])
    check('12 эпох — порог включения нужного оптимизатора',
          f'--epochs {mod.PROBE_EPOCHS}' in line and mod.PROBE_EPOCHS >= 12)
    check('свой каталог, основной прогон не затрагивается',
          'runs/init_probe/w4' in line and 'runs/drb_full' not in line)

    # состояние читается по диску: подсовываем журнал на 12 эпох
    run = mod.PROBE_ROOT / 'w2' / f'seed{mod.PROBE_SEED}'
    try:
        run.mkdir(parents=True, exist_ok=True)
        (run / 'results.csv').write_text(
            'epoch\n' + '\n'.join(str(i) for i in range(1, 13)), encoding='utf-8')
        st = {p['id']: p for p in api.probe_state()}
        check('полная проба помечается готовой', st['w2']['status'] == 'done',
              st['w2']['detail'])
        (run / 'results.csv').write_text('epoch\n1\n2\n3', encoding='utf-8')
        st = {p['id']: p for p in api.probe_state()}
        check('неполная проба помечается частичной', st['w2']['status'] == 'partial',
              st['w2']['detail'])
        check('для частичной добавляется продолжение',
              '--resume' in ' '.join(mod._probe_cmds('w2', -2.0)[0]))
    finally:
        shutil.rmtree(mod.PROBE_ROOT, ignore_errors=True)
    st = {p['id']: p for p in api.probe_state()}
    check('после сброса снова «не начата»', st['w2']['status'] == 'todo')


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
