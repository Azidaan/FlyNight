"""«Запусти меня» — панель управления проектом диссертации.

Нативное окно (pywebview + WebView2), внутри — интерфейс в стиле инженерного
чертежа (ui/launcher.html). Через панель запускаются все операции проекта:
smoke-тесты, проверки данных, мини-обучение; логи стримятся в журнал,
лампы-индикаторы показывают состояние узлов, смотровое окно листает кадры
BDD100K с разметкой — «как сеть видит данные».

Запуск: двойной клик или `python "Запусти меня.py"` — скрипт сам
перезапустится из локального .venv проекта, если запущен другим интерпретатором.

Служебный режим: `--selftest` — открыть окно, снять скриншот в
ui/selftest_screenshot.png и закрыться (проверка, что интерфейс собирается).
"""
import base64
import ctypes
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PY = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
HTML = PROJECT_ROOT / 'ui' / 'launcher.html'
WINDOW_TITLE = 'Запусти меня — панель управления обучением'
DATA = PROJECT_ROOT / 'data' / 'bdd100k'
# автономный прогон: журнал и метка процесса (переживают закрытие панели)
FULL_RUNNER = PROJECT_ROOT / 'scripts' / 'run_full_pipeline.py'
FULL_LOG = PROJECT_ROOT / 'runs' / 'full_pipeline.log'
FULL_PID = PROJECT_ROOT / 'runs' / 'full_pipeline.pid'


REEXEC_FLAG = '--in-venv'      # признак «этот процесс уже перезапущен»


def reexec_into_venv() -> None:
    """Перезапуск из .venv проекта, если запущены системным интерпретатором.

    Новый процесс ОТСОЕДИНЯЕТСЯ от родителя: иначе среда, из которой
    запустили (VS Code, IDE, терминал), убивает всю группу процессов сразу
    после выхода родителя — окно панели успевало только мигнуть.
    Под --selftest отсоединять нельзя: вызывающему нужен код возврата.

    Перезапуск строго однократный — по метке в аргументах, а не по сравнению
    путей интерпретатора. Сравнение путей подводило: в некоторых средах
    (запуск по F5 из VS Code) процесс из .venv считал себя «чужим» и
    запускал ещё один, тот — третий, и два окна дрались за один профиль
    WebView2. Итог выглядел как намертво зависшая программа.
    """
    if REEXEC_FLAG in sys.argv:
        sys.argv.remove(REEXEC_FLAG)
        return
    try:
        same = Path(sys.executable).resolve() == VENV_PY.resolve()
    except OSError:
        same = False
    if not same and VENV_PY.exists():
        flags = 0 if '--selftest' in sys.argv else (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        proc = subprocess.Popen(
            [str(VENV_PY), str(Path(__file__).resolve()), REEXEC_FLAG,
             *sys.argv[1:]],
            cwd=str(PROJECT_ROOT), creationflags=flags)
        if flags == 0:
            sys.exit(proc.wait())
        print(f'Панель запущена из .venv проекта (PID {proc.pid}).')
        sys.stdout.flush()
        # os._exit, а не sys.exit: родитель обязан исчезнуть немедленно.
        # Иначе в списке процессов висят два «Запусти меня», и непонятно,
        # какое окно живое — этим уже путались.
        os._exit(0)


reexec_into_venv()

import webview  # noqa: E402  (после reexec — гарантированно из .venv)

# Задача -> список скриптов (последовательно) + лампа, которую зажигает успех.
TASKS: dict[str, dict] = {
    'smoke': {
        'scripts': ['scripts/smoke/test_cuda.py', 'scripts/smoke/test_fft.py',
                    'scripts/smoke/test_bdd_annotations_parse.py',
                    'scripts/smoke/test_dataloader_workers.py',
                    'scripts/smoke/test_bdd_eval_tools.py',
                    'scripts/smoke/test_yolo_versions.py'],
        'lamp': 'gpu',
    },
    'subsets': {'scripts': ['scripts/test_subsets.py'], 'lamp': 'dataset'},
    'verify': {'scripts': ['scripts/verify_labels.py'], 'lamp': 'labels'},
    'visualize': {'scripts': ['scripts/visualize_subset_samples.py'], 'lamp': None},
    'minitrain': {'scripts': ['scripts/test_yolo_on_mini_bdd.py'], 'lamp': 'mini'},
}

RE_EPOCH = re.compile(r'^\s*(\d+)/(\d+)\s')          # строка эпохи Ultralytics
# Ultralytics красит вывод, а WebView2 показывает управляющие последовательности
# как мусор вида «[34m[1mtrain:» — вычищаем перед показом.
RE_ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
# строка прогресса обучения: эпоха, память, три ошибки, объектов, размер, процент
RE_TRAIN = re.compile(
    r'^\s*(\d+)/(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
    r'(\d+)\s+(\d+):\s*(\d+)%')
# хвост той же строки: итерации, скорость, прошло<осталось
# время бывает и «29.2s», и «12:34», и «?» — принимаем все формы
RE_TRAIN_TAIL = re.compile(
    r'(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)\s+([\d.:]+s?)<([\d.:?]+s?)')
RE_STEP = re.compile(r'^==========\s*(\S+)')          # шаг run_all_smoke

# ————— конвейер обучений (Этапы 3-10): статусы и команды —————
RESULTS = PROJECT_ROOT / 'experiments' / 'results'
SEEDS3 = (42, 123, 2024)


def _seed_states(run_name: str, seeds: tuple[int, ...]) -> dict[int, str]:
    """Состояние каждого сида: 'done' | 'resumable' | 'todo'."""
    out = {}
    for s in seeds:
        if (PROJECT_ROOT / 'weights' / f'{run_name}_seed{s}' / 'best.pt').exists():
            out[s] = 'done'
        elif (PROJECT_ROOT / 'runs' / run_name / f'seed{s}' / 'weights' / 'last.pt').exists():
            out[s] = 'resumable'
        else:
            out[s] = 'todo'
    return out


def _train_step_status(run_name: str, seeds: tuple[int, ...]) -> tuple[str, str]:
    """(status, detail) обучающего шага по фактам на диске."""
    st = _seed_states(run_name, seeds)
    n_done = sum(v == 'done' for v in st.values())
    if n_done == len(seeds):
        return 'done', f'{n_done}/{len(seeds)} сидов'
    if any(v == 'resumable' for v in st.values()):
        return 'partial', f'{n_done}/{len(seeds)} сидов · есть чекпойнт — продолжить'
    if n_done:
        return 'partial', f'{n_done}/{len(seeds)} сидов'
    return 'todo', f'0/{len(seeds)} сидов'


def _train_cmds(script: str, run_name: str, seeds: tuple[int, ...],
                extra: list[str] | None = None) -> list[list[str]]:
    """Команды супервизора для недоделанных сидов (готовые пропускаются)."""
    cmds = []
    for s, state in _seed_states(run_name, seeds).items():
        if state == 'done':
            continue
        cmd = ['scripts/train_supervisor.py', '--watch', f'runs/{run_name}',
               '--', 'python', script, '--seed', str(s), *(extra or [])]
        if state == 'resumable':
            cmd.append('--resume')
        cmds.append(cmd)
    return cmds


def _files_status(paths: list[str]) -> tuple[str, str]:
    have = sum((RESULTS / p).exists() if not p.startswith(('runs/', 'figs/'))
               else (PROJECT_ROOT / p).exists() for p in paths)
    if have == len(paths):
        return 'done', 'таблицы готовы'
    return ('partial', f'{have}/{len(paths)} таблиц') if have else ('todo', '')


def _ablation_status() -> tuple[str, str]:
    parts = [_train_step_status('ablation_lqa_only', (42, 123)),
             _train_step_status('ablation_fga_only', (42, 123)),
             _train_step_status('ablation_drb_input', (42,))]
    states = [p[0] for p in parts]
    if all(s == 'done' for s in states):
        status = 'done'
    elif any(s in ('done', 'partial') for s in states):
        status = 'partial'
    else:
        status = 'todo'
    detail = ' · '.join(f'{n} {p[1].split(" ")[0]}'
                        for n, p in zip(('lqa', 'fga', 'input'), parts))
    return status, detail


def _best_final_model() -> str:
    """Лучшая финальная модель для трекинг-шагов (drb_aug_full seed42)."""
    for run in ('drb_aug_full', 'drb_full', 'baseline'):
        p = PROJECT_ROOT / 'weights' / f'{run}_seed42' / 'best.pt'
        if p.exists():
            return str(p.relative_to(PROJECT_ROOT)).replace('\\', '/')
    return 'weights/drb_aug_full_seed42/best.pt'


# Каждый шаг: id, name, время, status() -> (status, detail), cmds() -> [argv]
PIPELINE: list[dict] = [
    {
        'id': 'baseline', 'name': 'Baseline · 3 сида × 100 эпох', 'time': '~2 сут',
        'status': lambda: _train_step_status('baseline', SEEDS3),
        'cmds': lambda: _train_cmds('training/train_baseline.py', 'baseline', SEEDS3),
    },
    {
        'id': 'eval_baseline', 'name': 'Оценка baseline по поднаборам', 'time': '~1 ч',
        'status': lambda: _files_status(['baseline_summary.csv']),
        'cmds': lambda: [['scripts/evaluate_baseline.py']],
    },
    {
        'id': 'drb', 'name': 'DRB-full · 3 сида × 100 эпох', 'time': '~2.5 сут',
        'status': lambda: _train_step_status('drb_full', SEEDS3),
        'cmds': lambda: _train_cmds('training/train_drb.py', 'drb_full', SEEDS3),
    },
    {
        'id': 'eval_drb', 'name': 'Оценка DRB + сравнение с baseline', 'time': '~1 ч',
        'status': lambda: _files_status(['drb_full_summary.csv', 'drb_vs_baseline.csv']),
        'cmds': lambda: [['scripts/evaluate_run.py', '--run', 'drb_full'],
                         ['scripts/compare_runs.py', '--runs', 'baseline', 'drb_full',
                          '--out', 'drb_vs_baseline'],
                         ['scripts/extract_drb_weights.py']],
    },
    {
        'id': 'drb_aug', 'name': 'DRB + аугментации · ФИНАЛ · 3 сида', 'time': '~2.5 сут',
        'status': lambda: _train_step_status('drb_aug_full', SEEDS3),
        'cmds': lambda: _train_cmds('training/train_drb.py', 'drb_aug_full', SEEDS3,
                                    ['--config', 'training/configs/drb_aug_full.yaml']),
    },
    {
        'id': 'eval_drb_aug', 'name': 'Оценка финала + тройное сравнение', 'time': '~1 ч',
        'status': lambda: _files_status(['drb_aug_full_summary.csv',
                                         'drb_aug_vs_drb_vs_baseline.csv']),
        'cmds': lambda: [['scripts/evaluate_run.py', '--run', 'drb_aug_full'],
                         ['scripts/compare_runs.py', '--runs', 'baseline', 'drb_full',
                          'drb_aug_full', '--out', 'drb_aug_vs_drb_vs_baseline']],
    },
    {
        'id': 'ablation', 'name': 'Ablation · 5 обучений × 50 эпох', 'time': '~2 сут',
        'status': lambda: _ablation_status(),
        'cmds': lambda: (
            _train_cmds('training/train_drb.py', 'ablation_lqa_only', (42, 123),
                        ['--variant', 'lqa', '--config', 'training/configs/lqa_only.yaml'])
            + _train_cmds('training/train_drb.py', 'ablation_fga_only', (42, 123),
                          ['--variant', 'fga', '--config', 'training/configs/fga_only.yaml'])
            + _train_cmds('training/train_drb.py', 'ablation_drb_input', (42,),
                          ['--variant', 'input', '--config', 'training/configs/drb_at_input.yaml'])),
    },
    {
        'id': 'eval_ablation', 'name': 'Оценка ablation + таблицы синергии', 'time': '~2 ч',
        'status': lambda: _files_status(['ablation_synergy.csv', 'ablation_paradigm.csv']),
        'cmds': lambda: [['scripts/evaluate_run.py', '--run', 'ablation_lqa_only', '--seeds', '42', '123'],
                         ['scripts/evaluate_run.py', '--run', 'ablation_fga_only', '--seeds', '42', '123'],
                         ['scripts/evaluate_run.py', '--run', 'ablation_drb_input', '--seeds', '42'],
                         ['scripts/build_ablation_tables.py']],
    },
    {
        'id': 'track_grid', 'name': 'Трекинг · сетка порогов + яркость', 'time': '~6 ч',
        'status': lambda: _files_status(['tracking_param_search.csv',
                                         'brightness_optimal_tau.csv']),
        'cmds': lambda: [['scripts/run_tracker_grid_search.py', '--model', _best_final_model()],
                         ['scripts/fit_brightness_regression.py', '--cache', 'runs/det_cache']],
    },
    {
        'id': 'track_modes', 'name': 'Трекинг · 4 режима + корреляция q/conf', 'time': '~5 ч',
        'status': lambda: _files_status(['lqa_k_sweep.csv',
                                         'tracking_three_modes.csv',
                                         'confidence_qi_correlation.csv',
                                         'q_aggregation_ablation.csv']),
        'cmds': lambda: [['scripts/sweep_lqa_k.py', '--model', _best_final_model()],
                         ['scripts/run_tracking_three_modes.py', '--model', _best_final_model()],
                         ['scripts/q_aggregation_ablation.py', '--model', _best_final_model()],
                         ['scripts/correlation_experiment.py', '--model', _best_final_model()]],
    },
    {
        'id': 'final', 'name': 'Финал · benchmark + сводная таблица', 'time': '~1 ч',
        'status': lambda: _files_status(['performance_table.csv', 'final/main_results.csv']),
        'cmds': lambda: [['utils/benchmark.py', '--model', 'weights/baseline_seed42/best.pt',
                          '--tag', 'baseline'],
                         ['utils/benchmark.py', '--model', _best_final_model(),
                          '--tag', 'drb_aug_full'],
                         ['scripts/build_final_summary.py']],
    },
]


class Api:
    def __init__(self) -> None:
        self.window: 'webview.Window | None' = None
        self._proc: subprocess.Popen | None = None
        self._abort = False
        self._lock = threading.Lock()
        self._last_epoch: int | None = None    # для обновления картинок по эпохам
        self._batch_cache: int | None = None   # размер батча идущего прогона
        self.boot_event = threading.Event()   # JS сигналит о конце boot()

    def boot_done(self) -> None:
        """Вызывается из JS после успешной инициализации интерфейса."""
        self.boot_event.set()
        if self._full_pid() is not None:
            self._log('Автономный прогон ИДЁТ прямо сейчас — подключаюсь к его '
                      'журналу (показываю последние строки).', 'ok')
            self._js('ui.setBusy("full", true)')
            tail_from = max(0, FULL_LOG.stat().st_size - 4000) if FULL_LOG.exists() else 0
            self._watch_full(tail_from)
        else:
            self._log('Обучение НЕ идёт. Панель ничего не делает сама — она '
                      'только показывает состояние. Чтобы начать, нажмите '
                      '«Запустить всё подряд» в строке состояния наверху.', 'info')

    # ——— взаимодействие с интерфейсом ———
    def _js(self, code: str) -> None:
        if self.window is not None:
            try:
                self.window.evaluate_js(code)
            except Exception:
                pass

    def _log(self, line: str, cls: str = '') -> None:
        self._js(f'ui.log({json.dumps(line)}, {json.dumps(cls)})')

    def _emit_line(self, raw: str) -> None:
        """Одна строка вывода подпроцесса -> журнал или живые показатели.

        Строки прогресса обучения в журнал НЕ попадают: их тысячи, и без
        подписей колонки всё равно нечитаемы. Они разбираются и уходят в
        узел Г с человеческими названиями.
        """
        line = RE_ANSI.sub('', raw.rstrip('\n').split('\r')[-1]).rstrip()
        if not line:
            return

        m = RE_TRAIN.match(line)
        if m:
            data = {
                'epoch': int(m.group(1)), 'epochs': int(m.group(2)),
                'mem': m.group(3), 'box': m.group(4), 'cls': m.group(5),
                'dfl': m.group(6), 'inst': m.group(7), 'pct': int(m.group(9)),
            }
            tail = RE_TRAIN_TAIL.search(line[m.end():])
            if tail:
                data['it'] = int(tail.group(1))
                data['iters'] = int(tail.group(2))
                # ВАЖНО: it/s — это батчи в секунду, а не кадры. В батче
                # обычно 16 кадров, и подпись «кадр/с» вводила в заблуждение.
                rate = float(tail.group(3))
                if tail.group(4) == 's/it':
                    rate = 1.0 / rate if rate else 0.0
                batch = self._active_batch()
                data['rate'] = (f'{rate:.1f} батча/с'
                                + (f'  ≈{rate * batch:.0f} кадр/с' if batch else ''))
                data['eta'] = tail.group(6)
            self._js('ui.setTrain(' + json.dumps(data, ensure_ascii=False) + ')')
            phase = f'ЭПОХА {data["epoch"]}/{data["epochs"]}'
            self._js('ui.setPhase(' + json.dumps(phase) + ')')
            if data['epoch'] != self._last_epoch:   # новая эпоха — новые картинки проверки
                self._last_epoch = data['epoch']
                self._js('ui.refreshImages(true)')
                self._js('ui.refreshCurve()')
            return

        if '%|' in line or RE_EPOCH.match(line):
            return                                   # прочие полосы прогресса
        cls = ''
        if '[OK' in line or 'OK:' in line:
            cls = 'ok'
        elif any(t in line for t in ('FAIL', 'Error', 'Traceback', 'ОШИБК',
                                     'СТАГНАЦИЯ', 'ПРОВАЛЕНЫ')):
            cls = 'err'
        elif line.startswith(('[INFO', '[SUP', '»', '===', '[03', '[04')):
            cls = 'info'
        self._log(line, cls)

    # ——— состояние и метрики ———
    def state(self) -> dict:
        lamps = {
            'dataset': 'ok' if (DATA / 'images' / 'track' / 'train').is_dir() else 'fail',
            'labels': 'ok' if (PROJECT_ROOT / 'notes' / 'conversion_stats.json').exists() else '',
            'mini': 'ok' if (PROJECT_ROOT / 'data' / 'bdd100k_mot_mini' / 'manifest.json').exists() else '',
            'eval': 'ok' if (PROJECT_ROOT / '.venv' / 'Lib' / 'site-packages' / 'bdd100k').is_dir() else '',
        }
        return {'lamps': lamps, 'metrics': self._metrics(),
                'date': date.today().strftime('%d.%m.%Y')}

    def _metrics(self) -> dict:
        m: dict = {}
        csv_path = PROJECT_ROOT / 'experiments' / 'results' / 'smoke_test_yolo.csv'
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    if row.get('all_passed') == 'True' and 'version' not in m:
                        m['version'] = row['version']
                        m['fps'] = str(round(float(row['fps'])))
        results = PROJECT_ROOT / 'runs' / 'smoke' / 'mini_bdd_train' / 'results.csv'
        if results.exists():
            import csv as _csv
            with open(results, encoding='utf-8') as f:
                rows = list(_csv.DictReader(f))
            if rows:
                key = next((k for k in rows[-1] if 'mAP50(B)' in k), None)
                if key:
                    m['map50'] = f'{float(rows[-1][key]):.3f}'
        counts = PROJECT_ROOT / 'notes' / 'subset_counts.json'
        if counts.exists():
            data = json.loads(counts.read_text(encoding='utf-8'))
            for name in ('normal', 'night', 'rain', 'fog'):
                m[name] = data['train']['subsets'][name]['videos']
        return m

    def gpu_check(self) -> None:
        def worker() -> None:
            self._js("ui.setLamp('gpu','busy')")
            try:
                rc = subprocess.run(
                    [str(VENV_PY), '-c', 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'],
                    timeout=90, creationflags=subprocess.CREATE_NO_WINDOW,
                ).returncode
            except Exception:
                rc = 1
            self._js(f"ui.setLamp('gpu',{json.dumps('ok' if rc == 0 else 'fail')})")
        threading.Thread(target=worker, daemon=True).start()

    # ——— смотровое окно ———
    def _active_batch(self) -> int | None:
        """Размер батча идущего прогона — из args.yaml, который пишет Ultralytics."""
        if self._batch_cache is not None:
            return self._batch_cache or None
        run = self._active_run_dir()
        args = run / 'args.yaml' if run else None
        if args and args.exists():
            for line in args.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.startswith('batch:'):
                    try:
                        self._batch_cache = int(float(line.split(':', 1)[1]))
                    except ValueError:
                        self._batch_cache = 0
                    return self._batch_cache or None
        self._batch_cache = 0
        return None

    def _active_run_dir(self) -> Path | None:
        """Папка прогона, который обучается прямо сейчас (свежайший results.csv)."""
        runs = PROJECT_ROOT / 'runs'
        newest, newest_mt = None, 0.0
        if runs.is_dir():
            for p in runs.glob('*/seed*/results.csv'):
                mt = p.stat().st_mtime
                if mt > newest_mt:
                    newest, newest_mt = p.parent, mt
        return newest

    def list_images(self) -> list[dict]:
        # Картинки идущего прогона идут первыми, даже если файл не самый свежий:
        # Ultralytics рисует мозаики обучающих батчей только при старте нового
        # обучения и не перерисовывает их при продолжении с чекпойнта.
        active = self._active_run_dir()
        items: list[tuple[float, dict]] = []
        # Сортировка строго по свежести: пока идёт обучение, первыми идут
        # настоящие обучающие батчи (на них сеть и учится прямо сейчас),
        # а не статичные примеры поднаборов.
        samples = PROJECT_ROOT / 'experiments' / 'results' / 'subset_samples'
        if samples.is_dir():
            for p in sorted(samples.glob('*.jpg')):
                subset = p.stem.split('_')[0].upper()
                items.append((p.stat().st_mtime, {
                    'path': str(p.relative_to(PROJECT_ROOT)),
                    'caption': f'ПРИМЕР ПОДНАБОРА · {subset} · '
                               f'{p.stem.split("_", 1)[1][:28]}',
                }))
        runs = PROJECT_ROOT / 'runs'
        if runs.is_dir():
            captions = {
                'train_batch': 'РЕАЛЬНЫЙ ОБУЧАЮЩИЙ БАТЧ (с аугментациями)',
                'val_batch': 'ПРОВЕРКА ПОСЛЕ ЭПОХИ',
            }
            for pattern in ('**/train_batch*.jpg', '**/val_batch*_pred.jpg',
                            '**/val_batch*_labels.jpg'):
                for p in runs.glob(pattern):
                    kind = 'train_batch' if 'train_batch' in p.name else 'val_batch'
                    is_active = active is not None and p.parent == active
                    mark = 'ИДЁТ СЕЙЧАС · ' if is_active else ''
                    run_name = f'{p.parent.parent.name}/{p.parent.name}'
                    items.append((p.stat().st_mtime + (1e12 if is_active else 0), {
                        'path': str(p.relative_to(PROJECT_ROOT)),
                        'caption': f'{mark}{captions[kind]} · {run_name}',
                    }))
        items.sort(key=lambda t: -t[0])
        return [meta for _, meta in items[:32]]

    def start_thermal_poll(self) -> None:
        """Опрос температур раз в 10 с — показывается в узле Г всегда."""
        def worker() -> None:
            from utils.thermal import STOP_TEMP, WARN_TEMP, read_temps
            while True:
                t = read_temps()
                t['warn'], t['stop'] = WARN_TEMP, STOP_TEMP
                self._js('ui.setTemps(' + json.dumps(t) + ')')
                time.sleep(10)
        threading.Thread(target=worker, daemon=True).start()

    def training_curve(self) -> dict:
        """Ход обучения по эпохам для узла Д: ошибка и точность на проверке.

        Берётся самый свежий results.csv под runs/ — это и есть журнал
        Ultralytics: одна строка на завершённую эпоху.
        """
        import csv as _csv
        runs = PROJECT_ROOT / 'runs'
        newest, newest_mtime = None, 0.0
        if runs.is_dir():
            for p in runs.glob('*/seed*/results.csv'):
                mt = p.stat().st_mtime
                if mt > newest_mtime:
                    newest, newest_mtime = p, mt
        if newest is None:
            return {}
        try:
            with open(newest, encoding='utf-8') as f:
                rows = list(_csv.DictReader(f))
        except OSError:
            return {}
        if not rows:
            return {}

        def col(row: dict, needle: str) -> float | None:
            for k, v in row.items():
                if needle in k and v not in (None, ''):
                    try:
                        return float(v)
                    except ValueError:
                        return None
            return None

        epochs, loss, mapv = [], [], []
        for r in rows:
            e = col(r, 'epoch')
            b, m = col(r, 'train/box_loss'), col(r, 'mAP50(B)')
            if e is None:
                continue
            epochs.append(int(e))
            loss.append(b if b is not None else 0.0)
            mapv.append(m if m is not None else 0.0)
        return {
            'run': f'{newest.parent.parent.name}/{newest.parent.name}',
            'epochs': epochs, 'loss': loss, 'map50': mapv,
        }

    def get_image(self, rel_path: str) -> str:
        p = (PROJECT_ROOT / rel_path).resolve()
        if not str(p).startswith(str(PROJECT_ROOT)) or not p.exists():
            return ''
        return 'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode()

    # ——— конвейер обучений ———
    def pipeline_state(self) -> list[dict]:
        """Статусы шагов + вычисление активного (первый не-done)."""
        out = []
        active_found = False
        for step in PIPELINE:
            try:
                status, detail = step['status']()
            except Exception as e:
                status, detail = 'todo', f'статус: {e}'
            role = 'done' if status == 'done' else 'locked'
            if status != 'done' and not active_found:
                role = 'active'
                active_found = True
            out.append({'id': step['id'], 'name': step['name'],
                        'time': step['time'], 'status': status,
                        'role': role, 'detail': detail})
        return out

    def run_step(self, step_id: str) -> None:
        step = next((s for s in PIPELINE if s['id'] == step_id), None)
        if step is None:
            return
        threading.Thread(target=self._run_step, args=(step,), daemon=True).start()

    def _run_step(self, step: dict) -> None:
        cmds = step['cmds']()
        if not cmds:
            self._log('Все части шага уже готовы — нечего запускать.', 'ok')
            self._js(f'ui.taskDone({json.dumps(step["id"])}, 0, null)')
            return
        self._log(f'Шаг «{step["name"]}»: {len(cmds)} команд(ы)', 'info')
        code = self._run_commands([[str(VENV_PY), str(PROJECT_ROOT / c[0]), *c[1:]]
                                   for c in cmds])
        self._js(f'ui.taskDone({json.dumps(step["id"])}, {code}, null)')

    # ——— автономный прогон (переживает закрытие панели) ———
    def _full_pid(self) -> int | None:
        """PID идущего автономного прогона или None."""
        if not FULL_PID.exists():
            return None
        try:
            pid = int(FULL_PID.read_text(encoding='utf-8').strip())
        except (ValueError, OSError):
            return None
        # tasklist без окна: живой процесс печатает строку со своим PID
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                             capture_output=True, text=True, errors='replace',
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return pid if str(pid) in out else None

    def run_full(self) -> None:
        """Автономный прогон всех оставшихся шагов с проверками.

        Запускается ОТДЕЛЬНЫМ процессом (не дочерним): панель можно закрыть,
        обучение продолжится. Панель показывает поток из runs/full_pipeline.log.
        """
        if self._full_pid() is not None:
            self._log('Автономный прогон уже идёт — подключаюсь к журналу.', 'info')
            self._watch_full()
            return
        # лог читаем с конца: старые прогоны в панель не тянем
        start_at = FULL_LOG.stat().st_size if FULL_LOG.exists() else 0
        proc = subprocess.Popen(
            [str(VENV_PY), str(FULL_RUNNER)],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env={**os.environ, 'PYTHONUTF8': '1'},
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        FULL_PID.write_text(str(proc.pid), encoding='utf-8')
        self._log(f'Автономный прогон запущен отдельным процессом (PID {proc.pid}). '
                  f'Панель можно закрыть — обучение продолжится.', 'ok')
        self._watch_full(start_at)

    def _watch_full(self, start_at: int | None = None) -> None:
        threading.Thread(target=self._tail_full, args=(start_at,),
                         daemon=True).start()

    def _tail_full(self, start_at: int | None) -> None:
        """Транслирует новые строки runs/full_pipeline.log в журнал панели."""
        pos = start_at if start_at is not None else 0
        while True:
            alive = self._full_pid() is not None
            if FULL_LOG.exists():
                with open(FULL_LOG, encoding='utf-8', errors='replace') as f:
                    f.seek(pos)
                    for raw in f:
                        self._emit_line(raw)
                    pos = f.tell()
            if not alive:
                break
            time.sleep(2.0)
        FULL_PID.unlink(missing_ok=True)
        self._js('ui.taskDone("full", 0, null)')

    # ——— запуск операций ———
    def run_task(self, task_id: str) -> None:
        if task_id not in TASKS:
            return
        threading.Thread(target=self._run_scripts, args=(task_id,), daemon=True).start()

    def _run_scripts(self, task_id: str) -> None:
        spec = TASKS[task_id]
        code = self._run_commands([[str(VENV_PY), str(PROJECT_ROOT / s)]
                                   for s in spec['scripts']])
        self._js(f'ui.taskDone({json.dumps(task_id)}, {code}, {json.dumps(spec["lamp"])})')

    def _run_commands(self, argv_list: list[list[str]]) -> int:
        """Последовательный запуск команд со стримингом строк в журнал."""
        self._abort = False
        code = 0
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        for argv in argv_list:
            if self._abort:
                code = -15
                break
            shown = ' '.join(Path(a).name if '\\' in a or '/' in a else a
                             for a in argv[1:])
            self._log(f'» {shown}', 'info')
            with self._lock:
                self._proc = subprocess.Popen(
                    argv, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                    errors='replace', bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW, env=env,
                )
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                self._emit_line(raw)
            self._proc.wait()
            code = self._proc.returncode or 0
            if code != 0:
                break
        with self._lock:
            self._proc = None
        return code

    def stop_task(self) -> None:
        """Стоп: убить всё дерево (супервизор + обучение + воркеры)."""
        self._abort = True
        with self._lock:
            if self._proc is not None:
                subprocess.run(['taskkill', '/PID', str(self._proc.pid), '/T', '/F'],
                               capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        full_pid = self._full_pid()
        if full_pid is not None:
            subprocess.run(['taskkill', '/PID', str(full_pid), '/T', '/F'],
                           capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            FULL_PID.unlink(missing_ok=True)
            self._log('Автономный прогон остановлен.', 'err')
        self._log('Остановлено пользователем (дерево процессов убито; '
                  'чекпойнты целы — шаг можно продолжить той же кнопкой).', 'err')

    def open_path(self, rel_path: str) -> None:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            os.startfile(str(p))  # noqa: S606


def _capture_window_by_title(title: str, out_path: Path) -> None:
    """Снимок содержимого окна через PrintWindow — работает и под другими окнами."""
    from ctypes import wintypes
    from PIL import Image

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        raise RuntimeError(f'Окно {title!r} не найдено')
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top

    hdc = user32.GetWindowDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mdc, bmp)
    user32.PrintWindow(hwnd, mdc, 2)  # 2 = PW_RENDERFULLCONTENT (композиция WebView2)

    class BMIH(ctypes.Structure):
        _fields_ = [('biSize', ctypes.c_uint32), ('biWidth', ctypes.c_int32),
                    ('biHeight', ctypes.c_int32), ('biPlanes', ctypes.c_uint16),
                    ('biBitCount', ctypes.c_uint16), ('biCompression', ctypes.c_uint32),
                    ('biSizeImage', ctypes.c_uint32), ('biXPelsPerMeter', ctypes.c_int32),
                    ('biYPelsPerMeter', ctypes.c_int32), ('biClrUsed', ctypes.c_uint32),
                    ('biClrImportant', ctypes.c_uint32)]

    bmi = BMIH(ctypes.sizeof(BMIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    Image.frombuffer('RGB', (w, h), buf, 'raw', 'BGRX', 0, 1).save(out_path)

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, hdc)


def _kill_orphan_webviews() -> None:
    """Гасит процессы WebView2, оставшиеся от прошлого запуска панели.

    Ищем строго по нашему профилю (.cache/webview2) — чужие браузеры и
    другие приложения на WebView2 не трогаем.
    """
    storage = str(PROJECT_ROOT / '.cache' / 'webview2')
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='msedgewebview2.exe'\" "
             f"| Where-Object {{ $_.CommandLine -like '*{storage}*' }} "
             "| Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return
    pids = [p for p in out.split() if p.isdigit()]
    for pid in pids:
        subprocess.run(['taskkill', '/PID', pid, '/T', '/F'], capture_output=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if pids:
        print(f'Убраны остатки прошлого запуска: {len(pids)} процесс(ов) WebView2.')


def main() -> None:
    # Единственный экземпляр: два окна на одном WebView2-профиле дают
    # чёрный рендер второго. Повторный запуск НЕ показывает диалог: модальное
    # окно уходило за другие и выглядело как зависшая программа. Вместо этого
    # поднимаем уже открытое окно и тихо выходим.
    ctypes.windll.kernel32.CreateMutexW(None, False, 'Global\\ZapustiMenyaPanel')
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            print('Панель уже запущена — поднял её окно.')
        else:
            print('Панель уже запущена (окно ещё открывается).')
        sys.exit(0)

    # Мьютекс взят — значит, других панелей нет, и любой процесс WebView2,
    # держащий НАШ профиль, остался от прошлого запуска. Такие «хвосты»
    # блокируют профиль, и новое окно повисает при старте.
    _kill_orphan_webviews()

    try:  # чёткий рендер на HiDPI
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass

    # pywebview при создании окна рекурсивно обходит свойства .NET-объекта
    # (Bounds.Empty.Empty…, ModifierKeys.A.A.A…) и упирается в предел
    # вложенности, после чего печатает простыню ошибок. На работу это не
    # влияет — глушим именно эти записи, настоящие ошибки остаются видны.
    class _DropNativeProbeNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return 'window.native' not in record.getMessage()

    logging.getLogger('pywebview').addFilter(_DropNativeProbeNoise())

    selftest = '--selftest' in sys.argv
    api = Api()
    # Страница отдаётся содержимым, а не путём к файлу: WebView2 кэширует
    # file://-страницы, и правки интерфейса не доезжали до окна — рисовалась
    # старая версия. Страница самодостаточна (стили и скрипт внутри неё,
    # картинки приходят через API как data:), внешних ссылок нет.
    window = webview.create_window(
        WINDOW_TITLE,
        html=HTML.read_text(encoding='utf-8'),
        js_api=api, x=24, y=24, width=1500, height=940,
        min_size=(1180, 760),
    )
    api.window = window

    if selftest:
        def snap() -> None:
            import time
            # ждём сигнала boot_done из JS (данные загружены), потом пост-рендер
            api.boot_event.wait(timeout=25)
            time.sleep(2.5)
            if '--with-train' in sys.argv:
                # снимок в состоянии «идёт обучение»: проверка живых показателей
                api._js('ui.setBusy("full", true)')
                api._emit_line(
                    '\x1b[K     29/100      5.11G      1.425     0.8542   '
                    '0.003134        273        640: 19% ━━──────── '
                    '694/3522 3.9it/s 4:16<12:08')
                time.sleep(1.0)
            try:
                _capture_window_by_title(
                    WINDOW_TITLE,
                    PROJECT_ROOT / 'ui' / 'selftest_screenshot.png',
                )
                print('SELFTEST OK: скриншот сохранён')
            finally:
                window.destroy()
        threading.Thread(target=snap, daemon=True).start()

    storage = PROJECT_ROOT / '.cache' / 'webview2'
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(storage_path=str(storage), private_mode=False)


if __name__ == '__main__':
    main()
