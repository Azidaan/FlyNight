"""Единый сбор результатов всех прогонов и учёт отключённых.

Зачем отдельный модуль. Прогонов набралось два десятка, их числа лежат в
трёх местах (кривые обучения, таблицы по поднаборам, таблица трекинга), и
каждое окно собирало их по-своему. Здесь один сборщик на всех — окно
результатов, окно графика и панель читают одно и то же.

Отключение прогона. Иногда прогон приходится исключить из расчёта:
прерывался, шёл на другом коде, оказался явно out of family. Список
отключённых хранится в experiments/results/excluded_runs.json и действует
сразу везде. ВАЖНО: сводка всегда считается ДВАЖДЫ — с учётом отключений
и по всем прогонам. В окне показан первый набор, но разница между ними
выводится отдельной строкой со стрелкой: влияние отключения должно быть
видно сразу, а не выясняться при постороннем вопросе.
"""
import csv
import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / 'experiments' / 'results'
RUNS = PROJECT_ROOT / 'runs'
EXCLUDED_FILE = RESULTS / 'excluded_runs.json'

# Целевая метрика работы: сколько мелких объектов находится на ночных
# кадрах. Узкое место трекера — пропуски (81% всех потерь), а худшая
# клетка базовой модели — мелкие объекты ночью.
TARGET_METRIC = 'AR50_small'
TARGET_SUBSET = 'night'


def _epochs(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, encoding='utf-8') as f:
            return [float(r['metrics/mAP50(B)']) for r in csv.DictReader(f)]
    except (OSError, KeyError, ValueError):
        return []


def run_key(group: str, seed: int) -> str:
    return f'{group}/seed{seed}'


def load_excluded() -> dict[str, str]:
    """{ключ прогона: заметка}. Заметка необязательна и может быть пустой."""
    if not EXCLUDED_FILE.exists():
        return {}
    try:
        with open(EXCLUDED_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return {str(k): str(v or '') for k, v in data.items()}
    except (OSError, ValueError):
        return {}


def save_excluded(data: dict[str, str]) -> None:
    EXCLUDED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXCLUDED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)


def toggle(key: str, note: str = '') -> bool:
    """Переключает учёт прогона. Возвращает новое состояние: True — учитывается."""
    data = load_excluded()
    if key in data:
        del data[key]
        save_excluded(data)
        return True
    data[key] = note
    save_excluded(data)
    return False


def _subset_values(metric: str, subset: str) -> dict[str, float]:
    """{имя модели: значение} по всем таблицам поднаборов, что есть на диске."""
    out: dict[str, float] = {}
    for f in sorted(RESULTS.glob('*.csv')):
        try:
            with open(f, encoding='utf-8') as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames or 'поднабор' not in rd.fieldnames:
                    continue
                for r in rd:
                    if r['поднабор'] == subset and r['метрика'] == metric:
                        out[r['модель']] = float(r['значение'])
        except (OSError, ValueError, KeyError):
            continue
    return out


def _track_values(metric: str, subset: str) -> dict[str, float]:
    """Метрики трекинга. Читаются из ВСЕХ таблиц, а не только из одной.

    Расчёт по кнопке пишет в свой файл (tracking_full_night.csv), а раньше
    здесь смотрели только probe_tracking.csv — числа лежали на диске, но в
    окно не попадали. Устройство таблиц одинаковое, различаются только
    названия метрик, поэтому сборщик у них общий.
    """
    return _subset_values(metric, subset)


def collect_runs() -> list[dict]:
    """Все прогоны с диска: группа, сид, эпох, итог, пик, целевая метрика.

    Имя модели в таблицах поднаборов не совпадает с каталогом прогона:
    пробы считались то как «drb_ad», то как «drb_ad_seed123». Проверяются
    оба написания.
    """
    excluded = load_excluded()
    target = _subset_values(TARGET_METRIC, TARGET_SUBSET)
    mota = _track_values('MOTA', TARGET_SUBSET)
    runs: list[dict] = []
    roots = [(RUNS / 'init_probe', True), (RUNS, False)]
    seen: set[Path] = set()
    for root, is_probe in roots:
        if not root.exists():
            continue
        for grp in sorted(root.iterdir()):
            if not grp.is_dir() or grp.name.startswith('_') or grp.name in (
                    'init_probe', 'detect', 'track', 'eval_probe'):
                continue
            for sd in sorted(grp.glob('seed*')):
                if not sd.is_dir() or sd in seen or not sd.name[4:].isdigit():
                    continue
                seen.add(sd)
                curve = _epochs(sd / 'results.csv')
                # Прогоны короче пробы — это smoke-проверки окружения, а не
                # результаты: в сводке им делать нечего.
                if len(curve) < 12:
                    continue
                seed = int(sd.name[4:])
                key = run_key(grp.name, seed)
                # Имя в таблицах поднаборов: сначала с сидом, потом без.
                tv = target.get(f'{grp.name}_seed{seed}')
                if tv is None and seed == 42:
                    tv = target.get(grp.name)
                mv = mota.get(f'{grp.name}_seed{seed}')
                if mv is None and seed == 42:
                    mv = mota.get(grp.name)
                runs.append({
                    'key': key, 'group': grp.name, 'seed': seed,
                    'probe': is_probe, 'epochs': len(curve),
                    'final': curve[-1], 'best': max(curve),
                    'target': tv, 'mota': mv,
                    'on': key not in excluded,
                    'note': excluded.get(key, ''),
                })
    return runs


def _stat(vals: list[float]) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {'n': len(vals), 'mean': sum(vals) / len(vals),
            'sd': st.stdev(vals) if len(vals) > 1 else 0.0,
            'lo': min(vals), 'hi': max(vals)}


def summarize(runs: list[dict], groups: list[str], field: str,
              only_on: bool) -> dict | None:
    sel = [r for r in runs if r['group'] in groups and (r['on'] or not only_on)]
    return _stat([r.get(field) for r in sel])


# Что нужно посчитать для каждой величины и чем.
#   fresh  — берётся прямо из кривых обучения, считать нечего;
#   subset — прогон весов по ночному срезу (evaluate_probes.py);
#   track  — то же плюс трекер и метрики MOT (долго).
BLOCK_JOBS = {
    'overall': None,
    'target': 'subset',
    'tracking': 'track',
}


def model_name(run: dict) -> str:
    """Как прогон называется в таблицах метрик: <группа>_seed<сид>."""
    return f"{run['group']}_seed{run['seed']}"


def _weights_of(run: dict) -> Path:
    """Веса прогона. Сначала в его собственном каталоге, потом в копии.

    Копия в weights/ делается ПОСЛЕ обучения, поэтому у идущего прогона её
    ещё нет — а best.pt в каталоге прогона Ultralytics пишет по ходу, и его
    уже можно оценивать. Без этой проверки идущий прогон значился как «без
    весов», то есть якобы непригодный к расчёту.
    """
    for base in (RUNS, RUNS / 'init_probe'):
        w = base / run['group'] / f"seed{run['seed']}" / 'weights' / 'best.pt'
        if w.exists():
            return w
    return PROJECT_ROOT / 'weights' / model_name(run) / 'best.pt'


# Две независимые пары сравнения. Смешивать их нельзя: расписание обучения
# разное (на двенадцати эпохах мозаика работает две эпохи из двенадцати, на
# ста — девяносто из ста), и контроль у каждой свой.
LENGTHS = [
    ('short', '12 эпох', ['base12'], ['drb_ad']),
    ('full', '100 эпох', ['baseline'], ['drb_ad100']),
]


def section_of(run: dict) -> str:
    """К какой длине обучения относится прогон. Порог тот же, что в таблице."""
    return 'full' if run['epochs'] > 12 else 'short'

METRICS = [
    ('overall', 'Общая точность', 'best',
     'mAP@0.50 по всему набору · лучшая эпоха · берётся из кривых обучения'),
    ('target', 'Целевая клетка', 'target',
     'полнота по мелким объектам ночью · ради этого работа'),
    ('tracking', 'Трекинг', 'mota',
     'MOTA ночью · главная метрика работы'),
]


def block_state(runs: list[dict], metric_id: str, section: str,
                field: str) -> dict:
    """Свежесть величины и список прогонов, которые надо посчитать.

    Берутся ВСЕ учитываемые прогоны этой длины, а не только та пара, что
    сравнивается. Иначе выходило расхождение: в таблице у w0, w2 и w4
    пусто, а сводка писала «актуально», потому что смотрела лишь на
    base12 и drb_ad. Лишние числа сравнению не мешают, а дыры в таблице
    мешают читать.

    Отключённые не в счёт: раз прогон не участвует в расчёте, досчитывать
    его незачем.

    Устаревшим считается прогон, чьи веса новее таблицы с метрикой: значит
    обучение продолжилось, а число осталось от прошлого раза.
    """
    job = BLOCK_JOBS.get(metric_id)
    sel = [r for r in runs if section_of(r) == section and r['on']]
    if not sel:
        return {'job': job, 'missing': [], 'stale': [], 'ready': 0,
                'total': 0, 'why': 'все прогоны этой длины отключены'}
    if job is None:
        return {'job': None, 'missing': [], 'stale': [], 'ready': len(sel),
                'total': len(sel), 'why': ''}
    table_mt = 0.0
    for f in RESULTS.glob('*.csv'):
        try:
            table_mt = max(table_mt, f.stat().st_mtime)
        except OSError:
            pass
    missing, stale, noweights = [], [], 0
    for r in sel:
        w = _weights_of(r)
        if not w.exists():
            noweights += 1
            continue
        if r.get(field) is None:
            missing.append(model_name(r))
        else:
            try:
                if w.stat().st_mtime > table_mt:
                    stale.append(model_name(r))
            except OSError:
                pass
    # Прогоны без сохранённых весов посчитать нельзя вообще: модели больше
    # нет. Молчать об этом нельзя — иначе в таблице остаётся необъяснимая
    # дыра, а сводка говорит «актуально».
    return {'job': job, 'missing': missing, 'stale': stale,
            'noweights': noweights,
            'ready': len(sel) - len(missing) - noweights,
            'total': len(sel), 'why': ''}


def metric_blocks(runs: list[dict]) -> list[dict]:
    """Три величины, каждая — отдельно по двенадцати и по ста эпохам.

    Раньше сводка знала только двенадцатиэпошную пару, и стоэпошные прогоны
    в неё не попадали вовсе. Теперь у каждой длины своя строка, свой набор
    прогонов и своя кнопка расчёта — считать их вместе нельзя.
    """
    out = []
    for mid, title, field, sub in METRICS:
        block = {'id': mid, 'title': title, 'sub': sub, 'lens': []}
        for lid, llabel, ctl, exp in LENGTHS:
            st = block_state(runs, mid, lid, field)
            st['pair'] = ' и '.join(ctl + exp)
            c = summarize(runs, ctl, field, True)
            e = summarize(runs, exp, field, True)
            c_all = summarize(runs, ctl, field, False)
            e_all = summarize(runs, exp, field, False)
            delta = (e['mean'] - c['mean']) if (c and e) else None
            delta_all = (e_all['mean'] - c_all['mean']) if (c_all and e_all) else None
            block['lens'].append({
                'id': f'{mid}_{lid}', 'label': llabel,
                'control': c, 'block': e, 'delta': delta,
                'delta_all': delta_all, 'state': st,
            })
        out.append(block)
    return out
