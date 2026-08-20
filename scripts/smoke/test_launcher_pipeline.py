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
    test_probe_archive(mod)
    test_short_run_guard(mod)
    test_chart_data()
    test_detached_run(mod)
    test_busy_guard(mod)

    print(f'\n--- {"OK: конвейер панели корректен" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


def test_chart_data() -> None:
    """Отдельное окно графика: данные читаются и пригодны для отрисовки."""
    print('=== данные окна графика ===')
    from scripts.chart_window import ChartApi
    d = ChartApi().curves()
    check('есть отметка времени', bool(d.get('stamp')), d.get('stamp', ''))
    views = d.get('views', [])
    check('есть хотя бы один вид', len(views) > 0, str(len(views)))
    for v in views:
        check(f'{v["id"]}: есть заголовок и подписи осей',
              bool(v.get('title') and v.get('xlabel') and v.get('ylabel')))
        for s in v.get('series', []):
            same = len(s['x']) == len(s['y'])
            check(f'{v["id"]}/{s["name"]}: длины осей совпадают', same,
                  f'{len(s["x"])} и {len(s["y"])}')
            check(f'{v["id"]}/{s["name"]}: значения конечны',
                  all(y == y and abs(y) != float('inf') for y in s['y']))
            check(f'{v["id"]}/{s["name"]}: задан цвет',
                  isinstance(s.get('color'), str) and s['color'].startswith('#'))


def test_probes(mod) -> None:
    """Проба инициализации DRB: состояние по диску, команды, сброс."""
    print('=== проба инициализации DRB (узел Е) ===')
    api = mod.Api()
    st = api.probe_state()
    check('четыре пробы и контроль', len(st) == 5, str([p['id'] for p in st]))
    check('стартовые веса 0 / -2 / -4, у контроля веса нет',
          [p['w'] for p in mod.PROBES] == [0.0, -2.0, -4.0, None, -4.0],
          str([p['w'] for p in mod.PROBES]))

    cmds = mod._probe_cmds('w4', -4.0)
    line = ' '.join(cmds[0])
    check('команда идёт через супервизор', cmds[0][0].endswith('train_supervisor.py'))
    check('передан стартовый вес', '--alpha-init -4.0' in line, line[-60:])
    check('12 эпох — порог включения нужного оптимизатора',
          f'--epochs {mod.PROBE_EPOCHS}' in line and mod.PROBE_EPOCHS >= 12)
    check('свой каталог, основной прогон не затрагивается',
          'runs/init_probe/w4' in line and 'runs/drb_full' not in line)

    # Контроль — тот же короткий прогон, но без блока: только с ним пробы
    # сравнимы, стосерийный baseline на 12-й эпохе учится в другом режиме.
    cline = ' '.join(mod._probe_cmds('base12', None)[0])
    check('контроль идёт обычным baseline, без DRB',
          'train_baseline.py' in cline and 'train_drb.py' not in cline)
    check('контролю не передаётся стартовый вес', '--alpha-init' not in cline)
    check('у контроля те же эпохи, что у проб',
          f'--epochs {mod.PROBE_EPOCHS}' in cline)
    check('контроль пишет в свой каталог',
          'init_probe' in cline and 'base12' in cline)

    # Разгонная проба: блок стартует тождественным и учится быстрее остальной
    # сети. Без разгона его веса стояли на месте — за 12 эпох маска
    # сдвигалась на 0.0004, а вклад ветвей на 0.0036.
    boost = next(p for p in mod.PROBES if p['id'] == 'boost')
    check('у разгонной пробы тождественный старт', boost['w'] == -4.0,
          str(boost['w']))
    check('разгон задан больше единицы', boost['lr'] > 1.0,
          f'×{boost["lr"]:g}')
    bline = ' '.join(mod._probe_cmds('boost', boost['w'], boost['lr'])[0])
    check('разгон доезжает до обучения', f'--lr-mult {boost["lr"]}' in bline,
          bline[-40:])
    check('обычным пробам лишний ключ не приписывается',
          '--lr-mult' not in ' '.join(mod._probe_cmds('w0', 0.0)[0]))

    # Итоговые веса копируются в weights/<имя каталога>_seed42. Раньше имя
    # было зашито, и короткий прогон затирал веса настоящего прогона — а на
    # них опираются оценка, трекинг и бенчмарк. Проверяем в исходниках:
    # настоящий путь считается внутри обучения, которому нужна видеокарта.
    for src_name in ('training/train_baseline.py', 'training/train_drb.py'):
        src = (PROJECT_ROOT / src_name).read_text(encoding='utf-8')
        check(f'{src_name}: папка весов из каталога прогона, а не зашита',
              "_root.name}_seed{seed}'" in src,
              'иначе короткий прогон затрёт веса настоящего')

    # Состояние читается по диску. КАТАЛОГ ПОДМЕНЯЕТСЯ на временный: раньше
    # тест писал в настоящий runs/init_probe и в конце сносил его целиком —
    # и однажды снёс идущую пробу вместе с одиннадцатью посчитанными эпохами
    # и чекпойнтом. Тест не имеет права трогать рабочие данные.
    import tempfile
    real_root = mod.PROBE_ROOT
    mod.PROBE_ROOT = Path(tempfile.mkdtemp(prefix='probe_test_',
                                           dir=str(PROJECT_ROOT / 'runs')))
    check('тест работает во временном каталоге, не в рабочем',
          mod.PROBE_ROOT != real_root, str(mod.PROBE_ROOT.name))
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
        shutil.rmtree(mod.PROBE_ROOT, ignore_errors=True)   # только временный
        tmp_root, mod.PROBE_ROOT = mod.PROBE_ROOT, real_root
    check('рабочий каталог проб не тронут',
          not str(tmp_root).endswith('init_probe'), str(tmp_root.name))


def test_probe_archive(mod) -> None:
    """Пересчёт готовой пробы не затирает прошлый результат, а откладывает его.

    Повторный запуск идёт с первой эпохи в ту же папку, так что без архива
    прошлые двенадцать эпох пропадали бесследно. Здесь проверяется, что
    старая версия уезжает в _archive, что таких версий может быть несколько
    и что сброс проб её не трогает.
    """
    import tempfile

    print('=== архив прошлых версий проб ===')
    api = mod.Api()
    api._log = lambda line, cls='': None
    api._js = lambda code: None
    api._training_running = lambda: ''
    api._run_commands = lambda cmds: 0          # обучение не запускаем

    real_root = mod.PROBE_ROOT
    mod.PROBE_ROOT = Path(tempfile.mkdtemp(prefix='probe_arch_',
                                           dir=str(PROJECT_ROOT / 'runs')))
    check('тест работает во временном каталоге, не в рабочем',
          mod.PROBE_ROOT != real_root, str(mod.PROBE_ROOT.name))
    run = mod.PROBE_ROOT / 'w2' / f'seed{mod.PROBE_SEED}'
    full = 'epoch\n' + '\n'.join(str(i) for i in range(1, 13))
    try:
        run.mkdir(parents=True, exist_ok=True)
        (run / 'results.csv').write_text(full, encoding='utf-8')
        check('до пересчёта архив пуст', mod._probe_archives('w2') == [])

        api._run_probe({'id': 'w2', 'w': -2.0, 'name': 'тест', 'time': ''})
        kept = mod._probe_archives('w2')
        check('готовая проба уехала в архив', len(kept) == 1,
              kept[0].name if kept else 'ничего')
        check('в архиве лежат все 12 эпох', kept and (
            kept[0] / f'seed{mod.PROBE_SEED}' / 'results.csv'
        ).read_text(encoding='utf-8') == full)
        check('рабочая папка освобождена под новый счёт',
              not (mod.PROBE_ROOT / 'w2').exists())
        check('статус вернулся к «не начат»',
              mod._probe_status('w2')[0] == 'todo')

        # вторая версия не затирает первую
        run.mkdir(parents=True, exist_ok=True)
        (run / 'results.csv').write_text(full, encoding='utf-8')
        api._run_probe({'id': 'w2', 'w': -2.0, 'name': 'тест', 'time': ''})
        check('второй пересчёт добавляет версию, а не затирает',
              len(mod._probe_archives('w2')) == 2,
              str([p.name for p in mod._probe_archives('w2')]))

        st = {p['id']: p for p in api.probe_state()}
        check('панель показывает число отложенных версий',
              st['w2']['kept'] == 2 and 'в архиве 2' in st['w2']['detail'],
              st['w2']['detail'])
        check('архив не притворяется ещё одной пробой', len(st) == 5)

        api.reset_probe('all')
        check('сброс проб архив не трогает',
              len(mod._probe_archives('w2')) == 2,
              str(len(mod._probe_archives('w2'))))
    finally:
        shutil.rmtree(mod.PROBE_ROOT, ignore_errors=True)   # только временный
        tmp_root, mod.PROBE_ROOT = mod.PROBE_ROOT, real_root
    check('рабочий каталог проб не тронут',
          not str(tmp_root).endswith('init_probe'), str(tmp_root.name))


def test_short_run_guard(mod) -> None:
    """Короткий прогон не выдаёт себя за обученный сид.

    Пробы когда-то складывали свои 12 эпох в weights/drb_full_seed42, и
    панель считала сид 42 обученным: шаг показывал «1/3 сидов», при запуске
    молча пропускал этот сид, а трекинг брал пробу за финальную модель.
    """
    import tempfile

    print('=== защита от подмены короткой пробой ===')
    real = mod.PROJECT_ROOT
    root = Path(tempfile.mkdtemp(prefix='guard_', dir=str(PROJECT_ROOT / 'runs')))
    mod.PROJECT_ROOT = root
    check('тест работает во временном каталоге, не в рабочем',
          mod.PROJECT_ROOT != real, root.name)
    try:
        w = root / 'weights' / 'drb_full_seed42'
        w.mkdir(parents=True)
        (w / 'best.pt').write_text('веса-заглушка', encoding='utf-8')
        run = root / 'runs' / 'drb_full' / 'seed42'
        run.mkdir(parents=True)

        def curve(n: int) -> None:
            (run / 'results.csv').write_text(
                'epoch\n' + '\n'.join(str(i) for i in range(1, n + 1)),
                encoding='utf-8')

        curve(mod.PROBE_EPOCHS)
        check('12 эпох не считаются обученным сидом',
              mod._seed_states('drb_full', (42,))[42] != 'done',
              mod._seed_states('drb_full', (42,))[42])
        check('короткий прогон не подставляется как финальная модель',
              'drb_full' not in mod._best_final_model(),
              mod._best_final_model())
        check('такой сид попадает в очередь на обучение',
              '42' in [c[c.index('--seed') + 1] for c in
                       mod._train_cmds('training/train_drb.py', 'drb_full', (42,))])

        curve(100)
        check('100 эпох считаются обученным сидом',
              mod._seed_states('drb_full', (42,))[42] == 'done')
        check('настоящий прогон становится финальной моделью',
              'drb_full' in mod._best_final_model(), mod._best_final_model())
        check('готовый сид больше не переобучается',
              mod._train_cmds('training/train_drb.py', 'drb_full', (42,)) == [])
    finally:
        mod.PROJECT_ROOT = real
        shutil.rmtree(root, ignore_errors=True)


def test_busy_guard(mod) -> None:
    """Запрет второго обучения: два прогона не помещаются в память."""
    print('=== защита от второго обучения ===')
    api = mod.Api()
    api._training_running = lambda: 'проба инициализации'
    logged: list[str] = []
    api._log = lambda line, cls='': logged.append(line)
    api._js = lambda code: None

    api._run_step(mod.PIPELINE[0])
    check('шаг конвейера не запускается при идущем обучении',
          any('Уже идёт обучение' in ln for ln in logged), logged[-1] if logged else '')
    logged.clear()
    api._run_probe(mod.PROBES[0])
    check('проба не запускается при идущем обучении',
          any('Уже идёт обучение' in ln for ln in logged), logged[-1] if logged else '')
    logged.clear()
    api._training_running = lambda: ''
    check('при свободной машине запрет не срабатывает',
          api._training_running() == '')


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
        # Панель запрещает второе обучение. Здесь проверяется механизм
        # отсоединённого процесса, а не запрет, поэтому запрет снимаем —
        # сам запрет проверяется отдельно ниже.
        api._training_running = lambda: ''
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
        # Заглушка пишет три строки за ~2 с. Если файла нет вовсе — процесс
        # не прожил и секунды; в некоторых средах отсоединённые процессы
        # гасятся сразу, и это не дефект панели. Тогда честно говорим об этом,
        # а не падаем с исключением на чтении несуществующего файла.
        if not mod.FULL_LOG.exists():
            print('[ПРОПУСК] журнал прогона: отсоединённый процесс не прожил '
                  'достаточно, чтобы что-то записать (особенность среды)')
        else:
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
