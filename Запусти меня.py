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
from datetime import date, datetime
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


def _epochs_done(results_csv: Path) -> int:
    """Сколько эпох записано в results.csv Ultralytics (0, если файла нет)."""
    if not results_csv.exists():
        return 0
    try:
        with open(results_csv, encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _seed_states(run_name: str, seeds: tuple[int, ...]) -> dict[int, str]:
    """Состояние каждого сида: 'done' | 'resumable' | 'todo'.

    Одного файла весов мало. Короткие прогоны когда-то складывали свою
    двенадцатиэпошную пробу под именем настоящего прогона, и панель считала
    сид обученным: шаг показывал «1/3 сидов», при запуске молча пропускал
    сид 42, а трекинг брал пробу за финальную модель. Поэтому рядом с
    весами проверяется кривая обучения — эпох в ней должно быть больше,
    чем в пробе.
    """
    out = {}
    for s in seeds:
        run = PROJECT_ROOT / 'runs' / run_name / f'seed{s}'
        done = ((PROJECT_ROOT / 'weights' / f'{run_name}_seed{s}' / 'best.pt').exists()
                and _epochs_done(run / 'results.csv') > PROBE_EPOCHS)
        if done:
            out[s] = 'done'
        elif (run / 'weights' / 'last.pt').exists():
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
    """Лучшая финальная модель для трекинг-шагов (drb_aug_full seed42).

    Берётся только по-настоящему обученная: файл весов рядом с короткой
    кривой — это проба, и двенадцать часов трекинга на ней дали бы
    правдоподобные, но неверные числа в сводную таблицу.
    """
    for run in ('drb_aug_full', 'drb_full', 'baseline'):
        p = PROJECT_ROOT / 'weights' / f'{run}_seed42' / 'best.pt'
        if p.exists() and _seed_states(run, (42,))[42] == 'done':
            return str(p.relative_to(PROJECT_ROOT)).replace('\\', '/')
    return 'weights/drb_aug_full_seed42/best.pt'


# ————— проба инициализации DRB (узел Е) —————
# Короткие прогоны по 12 эпох: с какого вклада ветвей стартовать блоку.
# 12 эпох выбраны не наугад — начиная с них Ultralytics переключается на тот
# же оптимизатор (MuSGD), что и в настоящем прогоне.
PROBE_ROOT = PROJECT_ROOT / 'runs' / 'init_probe'
PROBE_EPOCHS = 12
PROBE_SEED = 42
# Во сколько раз быстрее учить сам блок. Он приделан к предобученной сети:
# её веса настроены, его случайны, и на общей скорости он не двигается — за
# 12 эпох маска сдвинулась на 0.0004, вклад ветвей на 0.0036, то есть ни на
# что. Замер относительного шага (|градиент| / |вес|) это подтвердил: у
# отклонения маски он 0.02-0.03, у логитов α 0.7-4.2, а у свежей головы
# детектора 7575 — на четыре-пять порядков больше.
#
# Разгон достаётся только весам ветвей: маска после перехода на радиальные
# кольца собирает градиент со 137 отсчётов спектра вместо одного и учится
# сама. А вес ветви стал вектором по каналам, каждый элемент видит только
# свой канал, и его градиент разбавился — вот ему разгон и нужен. При 200
# вклад ветвей может за 12 эпох пройти около 0.7 в логите, то есть с 1.8%
# почти до 3.5%: меньше — не отличить от шума, больше — поймаем скачок.
PROBE_LR_MULT = 10.0
# Дообучение: стартуем не с весов COCO, а с уже сошедшегося baseline.
# Скорость обучения низкая и оптимизатор задан явно — при 'auto' Ultralytics
# берёт MuSGD со своей скоростью 0.01 и заданную игнорирует, а это не
# дообучение, а перезапуск с нуля.
FT_INIT = 'weights/baseline_seed42/best.pt'
FT_OPTIM = 'SGD'
FT_LR0 = 0.002
FT_WARMUP = 1.0
PROBES: list[dict] = [
    {'id': 'w0', 'w': 0.0, 'name': 'Вклад 50% — как в плане', 'time': '~5 ч'},
    {'id': 'w2', 'w': -2.0, 'name': 'Вклад 12%', 'time': '~5 ч'},
    {'id': 'w4', 'w': -4.0, 'name': 'Вклад 2% — почти тождественный старт',
     'time': '~5 ч'},
    # Контроль: тот же короткий прогон, но БЕЗ блока (w = None). Нужен
    # потому, что стосерийный baseline на 12-й эпохе с пробами несопоставим:
    # там втрое выше скорость обучения и ещё включена мозаика. Сравнивать
    # пробы можно только с прогоном на том же расписании.
    {'id': 'base12', 'w': None, 'name': 'Контроль — без DRB, равные условия',
     'time': '~3.5 ч'},
    # Замер шума: тот же контроль ещё на двух сидах, в ту же папку base12.
    # Зачем: весь опубликованный выигрыш частотного модуля для ночной
    # детекции на BDD100K составляет +0.7 пункта (NF-DETR, абляция: 0.779
    # без частотной ветви против 0.786 с ней), а разброс нашего baseline
    # между тремя сидами на ста эпохах — ровно 0.70 пункта. То есть на
    # одном сиде эффект такого размера неотличим от смены зерна. Пока
    # неизвестно, сколько шума на двенадцати эпохах, любое сравнение проб
    # между собой висит в воздухе. Готовый seed42 не трогается.
    {'id': 'base12_noise', 'w': None, 'out': 'base12', 'seeds': (123, 2024),
     'name': 'Контроль · ещё два сида — сколько тут шума', 'time': '~7 ч'},
    # Главный прогон: те же условия, что у контроля 0.4415 — с весов COCO,
    # двенадцать эпох, то же расписание, — отличается только блок. Настройки
    # собраны из всего, что намерено за 19–20 августа: вклад 12% (стоит 3%
    # точности рамок и ничего от mAP50, тогда как 27% стоят 19%, а 50% рушат
    # вдвое), радиальная маска со стартовым подъёмом высоких частот, разгон
    # весов ветвей вдесятеро (двухсоткратный разводил каналы между собой).
    {'id': 'drb12', 'w': -2.0, 'lr': PROBE_LR_MULT, 'mask': 'highpass',
     'name': 'DRB · те же условия, что у контроля', 'time': '~5 ч'},
    # Решающий прогон. Собран из всего, что намерено 19-21 августа:
    #  * поправка через затвор вместо прибавления копии — при нейтральной
    #    маске блок остаётся тождеством при ЛЮБОМ вкладе, поэтому подбирать
    #    стартовый вклад больше не нужно, и он взят нейтральным (логит 0,
    #    там же не искажает и регуляризация);
    #  * полосовая маска вместо подъёма высоких: диагностика показала, что
    #    ночь не мешает НАХОДИТЬ мелкие объекты (полнота 0.5138 против
    #    0.5186), зато роняет точность на 28% и локализацию с 47% до 38% —
    #    а это работа шума, который живёт в верхних частотах;
    #  * условия те же, что у контроля 0.4415, отличается только блок.
    {'id': 'drb_bp', 'w': 0.0, 'lr': PROBE_LR_MULT, 'mask': 'bandpass',
     'fusion': 'gated', 'name': 'DRB · поправка + полоса (подавление)',
     'time': '~5 ч'},
    # Адаптивный вариант. Статическая маска сходится к нейтрали: усреднённая
    # по дню и ночи, она вынуждена выбрать «ничего не делать». Здесь подъём
    # высоких частот предсказывается по самому кадру — двумя числами, чтобы
    # предсказатель не схлопнулся в константу.
    #
    # Именно подъём, а не подавление: замер 21.08 показал, что у ночных
    # кадров BDD100K вдвое МЕНЬШЕ высокочастотной энергии (0.119 против
    # 0.140) и вдвое слабее градиент, а шума в них меньше, а не больше
    # (0.0030 против 0.0076) — это видео с регистратора, камера уже
    # подавила шум. Значит ночью сигнал ослаблен, а не зашумлён, и нужна
    # компенсирующая добавка.
    {'id': 'drb_ad', 'w': 0.0, 'lr': PROBE_LR_MULT, 'fusion': 'gated',
     'adaptive': True, 'name': 'DRB · подъём по кадру (адаптивный)',
     'time': '~5 ч'},
    # Ступенчатое обучение. Настройки — от лучшей версии блока (адаптивной,
    # 0.4255 против 0.4415 у контроля), отличается ТОЛЬКО порядок: сперва
    # три эпохи с замороженной сетью, где подгоняется один блок, потом
    # обычные двенадцать со всеми размороженными.
    #
    # Почему именно это. Замерено, что вред наносится в начале: на четвёртой
    # эпохе подъём был 0.130 и точность 0.3311 против 0.3527 у контроля, а
    # дальше фильтр сам себя выключил (к финалу ровно ноль на слое 16), но
    # отставание 0.016 так и не отыгралось — двенадцати эпох не хватает на
    # долг. Плюс прямая улика против рецепта: дообучение уронило модель с
    # 0.4396 до 0.4192 БЕЗ всякого блока. Приём известен как LP-FT.
    #
    # Ствол сети получает те же двенадцать эпох, что и контроль: первая
    # фаза его не трогает — ни весов, ни бегущих средних нормировки.
    {'id': 'lp_ft', 'w': 0.0, 'lr': PROBE_LR_MULT, 'fusion': 'gated',
     'adaptive': True, 'warm': 3,
     'name': 'DRB · ступенчато: сперва блок, потом вся сеть',
     'time': '~5 ч'},
    # Фиксированная характеристика. Единственный вариант, где наблюдавшийся
    # режим отказа невозможен по построению: маска не обучается, схлопнуться
    # к нейтрали ей нечем. Обучаемая сходилась к нейтрали ТРИЖДЫ из трёх —
    # по отсчётам спектра, по кольцам и предсказанная по кадру.
    #
    # Форма — срез снизу без верхнего среза. Это полосовая маска минус её
    # вредная половина: замер спектров показал, что у ночных кадров и
    # высокочастотной энергии, и шума МЕНЬШЕ, то есть верхний срез убивал
    # сигнал. Прототип — высокочастотный модуль HS-FPN, который стоит в
    # FPN, то есть там же, где наш блок, и даёт +2.2 AP, у крошечных +2.6.
    #
    # Вес ветви остаётся обучаемым: это сила вмешательства, а не форма
    # фильтра. Предсказатель по кадру убран — снова обучаемое частотное
    # поведение. Обучаемых частотных параметров становится ноль против
    # 23 308 у адаптивного.
    {'id': 'drb_fix', 'w': 0.0, 'lr': PROBE_LR_MULT, 'fusion': 'gated',
     'mask': 'lowcut', 'fix_mask': True,
     'name': 'DRB · фиксированный фильтр, срез снизу', 'time': '~5 ч'},
    # Дообучение поверх сошедшегося baseline. Прежние пробы учили с нуля, а
    # там блок сначала возмущает признаки, и этот долг надо отработать —
    # двенадцать эпох это уже почти сходимость, отрабатывать негде. То есть
    # выиграть в них блок не мог в принципе. Здесь обе модели стартуют с
    # одной и той же точности 0.4396, долга нет ни у кого, и вся разница в
    # конце — заслуга блока. Контроль обязателен: без него не отличить
    # пользу блока от пользы лишних эпох.
    {'id': 'ft_base', 'w': None, 'ft': True,
     'name': 'Дообучение · контроль без блока', 'time': '~3.5 ч'},
    # Вклад 12%: прямой замер показал, что он стоит 3% от точности рамок и
    # ничего от mAP50, тогда как 27% стоят 19%, а 50% рушат вдвое. Маска
    # стартует подъёмом высоких частот — там живут мелкие объекты, худшее
    # место baseline (AP 0.051 ночью против 0.485 на крупных).
    {'id': 'ft_drb', 'w': -2.0, 'lr': PROBE_LR_MULT, 'mask': 'highpass',
     'ft': True, 'name': 'Дообучение · с блоком DRB', 'time': '~5 ч'},
]


def _probe_entry(pid: str) -> dict:
    """Описание пробы по id; для неизвестного — заглушка с одним сидом."""
    return next((x for x in PROBES if x['id'] == pid), {'id': pid})


def _probe_out(p: dict) -> Path:
    """Куда проба пишет прогоны.

    Обычно совпадает с id. Исключение — замер шума: он дописывает сиды в
    папку уже посчитанного контроля, чтобы кривые лежали рядом.
    """
    return PROBE_ROOT / p.get('out', p['id'])


def _probe_seeds(p: dict) -> tuple[int, ...]:
    return tuple(p.get('seeds', (PROBE_SEED,)))


def _probe_warm_root(p: dict) -> Path:
    """Каталог первой фазы ступенчатого обучения (сеть заморожена)."""
    return PROBE_ROOT / f'{_probe_out(p).name}_warm'


def _seed_state(p: dict, seed: int) -> tuple[str, int]:
    """(состояние, посчитано эпох) одного сида пробы."""
    run = _probe_out(p) / f'seed{seed}'
    n = _epochs_done(run / 'results.csv')
    if n >= PROBE_EPOCHS:
        return 'done', n
    if n > 0 or (run / 'weights' / 'last.pt').exists():
        return 'partial', n
    return 'todo', n


def _probe_status(pid: str) -> tuple[str, str]:
    """(status, detail) пробы по фактам на диске.

    У пробы может быть несколько сидов — тогда «готово» только когда
    досчитан каждый, а в подписи видно, сколько уже есть.
    """
    p = _probe_entry(pid)
    seeds = _probe_seeds(p)
    if len(seeds) > 1:
        states = [_seed_state(p, s)[0] for s in seeds]
        ready = states.count('done')
        if ready == len(seeds):
            return 'done', f'{ready} сида из {len(seeds)} — готово'
        if ready or 'partial' in states:
            return 'partial', f'{ready} из {len(seeds)} сидов'
        return 'todo', f'{len(seeds)} сида × {PROBE_EPOCHS} эпох'
    state, n = _seed_state(p, seeds[0])
    if state == 'done':
        return 'done', f'{n} эпох — готово'
    if state == 'partial':
        return 'partial', (f'{n} из {PROBE_EPOCHS} эпох' if n
                           else 'есть чекпойнт — продолжить')
    return 'todo', f'{PROBE_EPOCHS} эпох'


def _probe_phase(p: dict, seed: int, script: str, out: Path, epochs: int,
                 freeze: bool = False, init: Path | None = None,
                 resume: bool = False, watch: Path | None = None) -> list[str]:
    """Одна команда обучения: сид, каталог, число эпох, настройки блока."""
    w = p['w']
    watch = out / f'seed{seed}' if watch is None else watch
    cmd = ['scripts/train_supervisor.py', '--watch',
           watch.relative_to(PROJECT_ROOT).as_posix(),
           '--', 'python', script,
           '--seed', str(seed), '--epochs', str(epochs),
           '--output', str(out)]          # только абсолютный: см. train_drb.py
    if w is not None:
        cmd += ['--alpha-init', str(w)]
        if p.get('lr', 1.0) != 1.0:
            cmd += ['--lr-mult', str(p['lr'])]
        if p.get('mask', 'flat') != 'flat':
            cmd += ['--mask-init', p['mask']]
        if p.get('fusion', 'add') != 'add':
            cmd += ['--fusion', p['fusion']]
        if p.get('adaptive'):
            cmd += ['--adaptive']
        if p.get('fix_mask'):
            cmd += ['--fix-mask']
    if freeze:
        cmd.append('--freeze-base')
    if p.get('ft'):
        cmd += ['--init-weights', FT_INIT, '--optimizer', FT_OPTIM,
                '--lr0', str(FT_LR0), '--warmup', str(FT_WARMUP)]
    elif init is not None:
        cmd += ['--init-weights', str(init)]
    if resume:
        cmd.append('--resume')
    return cmd


def _probe_cmds(p: dict) -> list[list[str]]:
    """Команды прогона пробы. w = None — без DRB, обычный baseline.

    По одной команде на сид, а у ступенчатого обучения — по две: сперва
    короткая фаза с замороженной сетью, где подгоняется один блок, потом
    обычный прогон с её весов.

    У проб с несколькими сидами досчитанные пропускаются: замер шума
    дописывает сиды туда, где уже лежит готовый контроль, и затирать его
    нельзя. У однососедных проб поведение прежнее — пересчёт с первой эпохи
    (прошлый результат перед этим уезжает в архив, см. _archive_probe).
    """
    w = p['w']
    out = _probe_out(p)
    seeds = _probe_seeds(p)
    script = 'training/train_baseline.py' if w is None else 'training/train_drb.py'
    cmds = []
    for seed in seeds:
        state, _ = _seed_state(p, seed)
        if state == 'done' and len(seeds) > 1:
            continue
        init, warm = None, p.get('warm')
        if warm:
            warm_root = _probe_warm_root(p)
            init = warm_root / f'seed{seed}' / 'weights' / 'best.pt'
            if not init.exists():      # готовую первую фазу не пересчитываем
                cmds.append(_probe_phase(p, seed, script, warm_root,
                                         epochs=warm, freeze=True))
        # Наблюдаемый каталог: у многосидовой пробы — свой на каждый сид,
        # иначе супервизор принял бы чужой чекпойнт за собственный прогресс.
        watch = None if len(seeds) > 1 else PROBE_ROOT / p['id']
        cmds.append(_probe_phase(p, seed, script, out, epochs=PROBE_EPOCHS,
                                 init=init, resume=(state == 'partial'),
                                 watch=watch))
    return cmds


def _probe_archive_root() -> Path:
    """Куда уезжают прошлые версии проб.

    Считается на месте, а не константой: тест подменяет PROBE_ROOT на
    временный каталог, и архив обязан переехать вместе с ним.
    """
    return PROBE_ROOT / '_archive'


def _probe_archives(pid: str) -> list[Path]:
    """Отложенные версии пробы, от старых к свежим."""
    root = _probe_archive_root()
    if not root.exists():
        return []
    return sorted(p for p in root.glob(f'{pid}_*') if p.is_dir())


def _archive_probe(pid: str) -> Path | None:
    """Убрать готовую пробу в архив, чтобы пересчёт её не затёр.

    Повторный запуск завершённой пробы начинается с первой эпохи и пишет в
    ту же папку (exist_ok=True в train_drb.py) — то есть затирает прошлый
    результат вместе с весами и графиками. Здесь он переезжает в _archive
    под меткой времени и остаётся доступным для сравнения.
    """
    import shutil
    src = PROBE_ROOT / pid
    if not src.exists():
        return None
    root = _probe_archive_root()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%d-%m_%H%M')
    dst, k = root / f'{pid}_{stamp}', 2
    while dst.exists():                    # два пересчёта в одну минуту
        dst, k = root / f'{pid}_{stamp}_{k}', k + 1
    shutil.move(str(src), str(dst))
    return dst


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
    # Подъём по кадру на полной длине. Двенадцатиэпошные пробы показали
    # ровно одно: на такой длине блок ничего не успевает. Первые четыре-пять
    # эпох он портит признаки, к двенадцатой почти догоняет контроль, и на
    # этом обучение кончается. Сто эпох дают на выправление в восемь раз
    # больше времени — это единственная непроверенная причина, по которой
    # ответ может оказаться другим, а не просто точнее.
    #
    # Один сид, не три. Разброс контроля на ста эпохах 0.0037 против 0.0144
    # на двенадцати; одного прогона хватает, чтобы заметить разницу от
    # 0.0085. Три сида опустили бы порог до 0.0060 ценой втрое большего
    # счёта — сделка плохая, пока неизвестно, есть ли там что замечать.
    #
    # Плечо сравнения готово: стоэпошный baseline на трёх сидах. Конфигурации
    # совпадают целиком, отличается одна строка — включён блок.
    {
        'id': 'drb_ad100',
        'name': 'DRB · подъём по кадру · 100 эпох · 1 сид', 'time': '~1.5 сут',
        'status': lambda: _train_step_status('drb_ad100', (PROBE_SEED,)),
        'cmds': lambda: _train_cmds(
            'training/train_drb.py', 'drb_ad100', (PROBE_SEED,),
            ['--output', str(PROJECT_ROOT / 'runs' / 'drb_ad100'),
             '--alpha-init', '0.0', '--lr-mult', str(PROBE_LR_MULT),
             '--fusion', 'gated', '--adaptive']),
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

    @staticmethod
    def _read_curve(path: Path) -> dict:
        """Кривая одного прогона из results.csv Ultralytics (строка = эпоха)."""
        import csv as _csv
        try:
            with open(path, encoding='utf-8') as f:
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
            if e is None:
                continue
            epochs.append(int(e))
            loss.append(col(r, 'train/box_loss') or 0.0)
            mapv.append(col(r, 'mAP50(B)') or 0.0)
        return {'epochs': epochs, 'loss': loss, 'map50': mapv}

    def training_curve(self) -> dict:
        """Данные для графика в узле Д.

        Два режима. Если начата хоть одна проба инициализации — показываем
        СРАВНЕНИЕ: все пробы и baseline на одних и тех же эпохах, чтобы прямо
        по ходу дела было видно, какой стартовый вклад ветвей лучше. Иначе —
        обычная кривая идущего прогона (точность и ошибка).
        """
        probes = []
        for p in PROBES:
            c = self._read_curve(_probe_out(p) / f'seed{_probe_seeds(p)[0]}'
                                 / 'results.csv')
            if c.get('epochs'):
                probes.append((p, c))

        if probes:
            n = max(len(c['epochs']) for _, c in probes)
            series = []
            base = self._read_curve(
                PROJECT_ROOT / 'runs' / 'baseline' / f'seed{PROBE_SEED}' / 'results.csv')
            if base.get('epochs'):
                series.append({'name': 'baseline (без DRB)', 'style': 'ref',
                               'map50': base['map50'][:n]})
            for p, c in probes:
                series.append({'name': p['name'].split(' — ')[0],
                               'style': p['id'], 'map50': c['map50']})
            return {'mode': 'probe', 'series': series, 'limit': PROBE_EPOCHS,
                    'title': f'проба инициализации · {n} из {PROBE_EPOCHS} эпох'}

        run = self._active_run_dir()
        if run is None:
            return {}
        c = self._read_curve(run / 'results.csv')
        if not c.get('epochs'):
            return {}
        c['mode'] = 'single'
        c['run'] = f'{run.parent.name}/{run.name}'
        return c

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

    def _training_running(self) -> str:
        """Описание уже идущего обучения или '' — защита от второго запуска.

        Два обучения на одной видеокарте не просто делят её: каждое поднимает
        свой набор загрузчиков, память кончается, и оба уползают в своп —
        скорость падает в сорок раз. Дважды наступали, поэтому проверяем.
        """
        try:
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                 "| Where-Object { $_.CommandLine -like '*train_drb*' -or "
                 "$_.CommandLine -like '*train_baseline*' } "
                 "| Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=25,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        except (OSError, subprocess.SubprocessError):
            return ''
        lines = [ln for ln in out.splitlines() if 'train_' in ln]
        if not lines:
            return ''
        for ln in lines:                       # что именно считается
            for mark, name in (('init_probe', 'проба инициализации'),
                               ('drb_aug', 'DRB с аугментациями'),
                               ('ablation', 'ablation'),
                               ('train_drb', 'DRB'),
                               ('train_baseline', 'baseline')):
                if mark in ln:
                    return name
        return 'обучение'

    def backup_results(self) -> None:
        threading.Thread(target=self._backup_results, daemon=True).start()

    def _backup_results(self) -> None:
        """Снимок весов и результатов в папку, выбранную родным диалогом.

        Копирование идёт прямо здесь, а не через _run_commands: там один
        общий слот подпроцесса, и кнопка «остановить» гасила бы копию
        вместо обучения.
        """
        try:
            picked = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        except Exception as exc:
            self._log(f'Не удалось открыть выбор папки: {exc}', 'err')
            self._js('ui.saveDone()')
            return
        if not picked:
            self._log('Сохранение отменено', 'info')
            self._js('ui.saveDone()')
            return

        dest = Path(picked[0])
        busy = self._training_running()
        if busy:
            self._log(f'Идёт обучение ({busy}) — копирование будет делить с ним '
                      f'диск, обе задачи пойдут медленнее. Это не опасно.', 'info')
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.backup_results import backup
            snap = backup(dest, log=lambda line: self._log(line))
            self._log(f'Сохранено: {snap}', 'ok')
            self._log('Рядом со снимком лежит ВОЗВРАТ.md — там написано, '
                      'куда что класть, чтобы проект ожил на другой машине.',
                      'ok')
        except Exception as exc:
            self._log(f'Сохранить не удалось: {exc}', 'err')
        self._js('ui.saveDone()')

    def open_chart(self) -> None:
        """Отдельное окно с подробным графиком — свой процесс и свой профиль."""
        try:
            proc = subprocess.Popen(
                [str(VENV_PY), str(PROJECT_ROOT / 'scripts' / 'chart_window.py')],
                cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
            self._log(f'Окно графика открыто (PID {proc.pid}). '
                      f'Клик по графику — следующий вид, колесо — масштаб.', 'ok')
        except OSError as e:
            self._log(f'Не удалось открыть окно графика: {e}', 'err')

    # ——— проба инициализации DRB (узел Е) ———
    def probe_state(self) -> list[dict]:
        out = []
        for p in PROBES:
            status, detail = _probe_status(p['id'])
            kept = len(_probe_archives(p['id']))
            if kept:
                detail += f' · в архиве {kept}'
            out.append({'id': p['id'], 'name': p['name'], 'time': p['time'],
                        'status': status, 'detail': detail, 'kept': kept})
        return out

    def run_probe(self, probe_id: str) -> None:
        p = next((x for x in PROBES if x['id'] == probe_id), None)
        if p is None:
            return
        threading.Thread(target=self._run_probe, args=(p,), daemon=True).start()

    def _run_probe(self, p: dict) -> None:
        busy = self._training_running()
        if busy:
            self._log(f'Уже идёт обучение ({busy}). Второе запускать нельзя: '
                      f'кончится память и оба уползут в своп. Дождитесь конца '
                      f'или остановите текущее.', 'err')
            self._js(f'ui.taskDone({json.dumps(p["id"])}, 0, null)')
            return
        cmds = _probe_cmds(p)
        if not cmds:
            self._log(f'Считать нечего: у пробы «{p["name"]}» все сиды уже '
                      f'посчитаны.', 'info')
            self._js(f'ui.taskDone({json.dumps(p["id"])}, 0, null)')
            return
        # Готовая проба считается заново с первой эпохи и пишет в ту же
        # папку — прошлый результат сначала откладываем, иначе он пропадёт.
        # Многосидовую не архивируем: она дописывает сиды в чужую папку и
        # готовые не пересчитывает, затирать там нечего.
        if len(_probe_seeds(p)) == 1 and _probe_status(p['id'])[0] == 'done':
            kept = _archive_probe(p['id'])
            if kept is not None:
                self._log(f'Прошлый результат отложен в архив: '
                          f'{kept.relative_to(PROJECT_ROOT)}', 'ok')
        if p.get('ft'):
            what = ('без блока' if p['w'] is None else
                    f'вклад ветви {1/(1+pow(2.718281828, -p["w"])):.1%}')
            self._log(f'Дообучение поверх сошедшегося baseline: {p["name"]} '
                      f'({PROBE_EPOCHS} эпох, {what}, скорость {FT_LR0:g})',
                      'info')
        elif len(_probe_seeds(p)) > 1:
            todo = [c[c.index('--seed') + 1] for c in cmds]
            self._log(f'Замер шума: {p["name"]} — сиды {", ".join(todo)} по '
                      f'{PROBE_EPOCHS} эпох. Готовые сиды не трогаются. Нужно, '
                      f'чтобы знать, какая разница между прогонами значима, а '
                      f'какая объясняется зерном генератора.', 'info')
        elif p['w'] is None:
            self._log(f'Контрольный прогон: {p["name"]} ({PROBE_EPOCHS} эпох, '
                      f'без блока — эталон для проб на том же расписании)',
                      'info')
        else:
            boost = (f', скорость блока ×{p["lr"]:g}'
                     if p.get('lr', 1.0) != 1.0 else '')
            self._log(f'Проба инициализации: {p["name"]} '
                      f'({PROBE_EPOCHS} эпох, вклад ветви '
                      f'{1/(1+pow(2.718281828, -p["w"])):.1%}{boost})', 'info')
        code = self._run_commands(
            [[str(VENV_PY), str(PROJECT_ROOT / c[0]), *c[1:]] for c in cmds])
        self._js(f'ui.taskDone({json.dumps(p["id"])}, {code}, null)')

    def reset_probe(self, probe_id: str) -> None:
        """Сброс пробы: удаляются чекпойнты и журнал, шаг снова «не начат».

        Архив прошлых пересчётов не трогается: он заведён ровно для того,
        чтобы посчитанное не пропадало. Раньше сброс «всего» сносил корень
        целиком — теперь перечисляются именно пробы.
        """
        import shutil
        chosen = PROBES if probe_id == 'all' else [_probe_entry(probe_id)]
        # У ступенчатого обучения две папки; без второй сброс оставил бы
        # готовую первую фазу, и следующий запуск молча стартовал бы с неё.
        targets = [PROBE_ROOT / x['id'] for x in chosen]
        targets += [_probe_warm_root(x) for x in chosen if x.get('warm')]
        gone = [t.name for t in targets if t.exists()]
        for t in targets:
            shutil.rmtree(t, ignore_errors=True)
        if gone:
            self._log(f'Проба сброшена: {", ".join(gone)} · архив не тронут',
                      'err')
        else:
            self._log(f'Сбрасывать нечего: {probe_id}', 'info')
        self._js('pywebview.api.probe_state().then(renderProbes)')

    def eval_probes(self) -> None:
        threading.Thread(target=self._eval_probes, daemon=True).start()

    def _eval_probes(self) -> None:
        """Оценка проб по поднаборам: ночь, дождь, туман и размеры объектов.

        Обучения не запускает — только считает уже обученные модели. Но
        видеокарту занимает, поэтому при идущем обучении не стартует.
        """
        busy = self._training_running()
        if busy:
            self._log(f'Идёт обучение ({busy}) — оценка займёт ту же '
                      f'видеокарту. Дождитесь конца.', 'err')
            self._js('ui.taskDone("probeeval", 0, null)')
            return
        self._log('Оценка по поднаборам: общая точность блока ниже, но тема '
                  'работы про сложные сцены. Самое слабое место базовой '
                  'модели — мелкие объекты ночью, AP 0.051 против 0.485 на '
                  'крупных днём. Здесь смотрим именно туда.', 'info')
        code = self._run_commands(
            [[str(VENV_PY), str(PROJECT_ROOT / 'scripts' / 'evaluate_probes.py')]])
        self._js(f'ui.taskDone("probeeval", {code}, null)')

    def compare_probes(self) -> None:
        threading.Thread(target=self._compare_probes, daemon=True).start()

    def _compare_probes(self) -> None:
        code = self._run_commands(
            [[str(VENV_PY), str(PROJECT_ROOT / 'scripts' / 'compare_init_probe.py')]])
        self._js(f'ui.taskDone("probecmp", {code}, null)')

    def run_step(self, step_id: str) -> None:
        step = next((s for s in PIPELINE if s['id'] == step_id), None)
        if step is None:
            return
        threading.Thread(target=self._run_step, args=(step,), daemon=True).start()

    def _run_step(self, step: dict) -> None:
        busy = self._training_running()
        if busy:
            self._log(f'Уже идёт обучение ({busy}) — второе запускать нельзя: '
                      f'кончится память и оба уползут в своп.', 'err')
            self._js(f'ui.taskDone({json.dumps(step["id"])}, 0, null)')
            return
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
        busy = self._training_running()
        if busy:
            self._log(f'Уже идёт обучение ({busy}) — прогон не запускаю: '
                      f'два обучения не помещаются в память.', 'err')
            self._js('ui.taskDone("full", 0, null)')
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
    # Служебная проверка (--selftest) работает в стороне от рабочей панели:
    # свой профиль WebView2 и без блокировки «единственный экземпляр». Иначе
    # проверить интерфейс можно было бы только закрыв рабочее окно, а во время
    # многосуточного обучения это недопустимо.
    selftest_mode = '--selftest' in sys.argv

    # Единственный экземпляр: два окна на одном WebView2-профиле дают
    # чёрный рендер второго. Повторный запуск НЕ показывает диалог: модальное
    # окно уходило за другие и выглядело как зависшая программа. Вместо этого
    # поднимаем уже открытое окно и тихо выходим.
    if not selftest_mode:
        ctypes.windll.kernel32.CreateMutexW(None, False, 'Global\\ZapustiMenyaPanel')
    if not selftest_mode and ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
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
    # блокируют профиль, и новое окно повисает при старте. Под проверкой не
    # трогаем: там свой профиль, а рабочую панель гасить нельзя.
    if not selftest_mode:
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
            if '--scroll' in sys.argv:
                # колонки прокручиваются — без этого нижние узлы не попадают
                # в снимок, и проверить их вид нельзя
                api._js('document.querySelectorAll(".rail")'
                        '.forEach(r => r.scrollTop = 9999)')
                time.sleep(0.6)
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

    storage = PROJECT_ROOT / '.cache' / (
        'webview2_selftest' if selftest_mode else 'webview2')
    storage.mkdir(parents=True, exist_ok=True)
    webview.start(storage_path=str(storage), private_mode=False)


if __name__ == '__main__':
    main()
