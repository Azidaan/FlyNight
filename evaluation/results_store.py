"""Единый сбор результатов всех прогонов и учёт отключённых.

Зачем отдельный модуль. Прогонов набралось два десятка, их числа лежат в
трёх местах (кривые обучения, таблицы по поднаборам, таблица трекинга), и
каждое окно собирало их по-своему. Здесь один сборщик на всех — окно
результатов, окно графика и панель читают одно и то же.

Отключение прогона. Иногда прогон приходится исключить из расчёта:
прерывался, шёл на другом коде, оказался явно out of family. Список
отключённых хранится в experiments/results/excluded_runs.json и действует
сразу везде. ВАЖНО: сводка всегда считается ДВАЖДЫ — с учётом отключений
и по всем прогонам. Второе число никуда не прячется, чтобы влияние
отключения было видно сразу, а не выяснилось при постороннем вопросе.
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
    p = RESULTS / 'probe_tracking.csv'
    out: dict[str, float] = {}
    if not p.exists():
        return out
    try:
        with open(p, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if r['поднабор'] == subset and r['метрика'] == metric:
                    out[r['модель']] = float(r['значение'])
    except (OSError, ValueError, KeyError):
        return {}
    return out


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


def metric_blocks(runs: list[dict]) -> list[dict]:
    """Три интересующие величины, каждая — с учётом отключений и без.

    Сравниваются двенадцатиэпошные пробы: контроль base12 против подъёма по
    кадру drb_ad. Это единственная пара, где обе стороны посчитаны на
    нескольких сидах в одинаковых условиях.
    """
    pairs = [
        ('Общая точность', 'best', ['base12'], ['drb_ad'],
         'mAP@0.50 по всему набору · лучшая эпоха'),
        ('Целевая клетка', 'target', ['base12'], ['drb_ad'],
         'полнота по мелким объектам ночью · ради этого работа'),
        ('Трекинг', 'mota', ['base12'], ['drb_ad'],
         'MOTA ночью · пока не посчитано'),
    ]
    out = []
    for title, field, ctl, exp, sub in pairs:
        block = {'title': title, 'sub': sub, 'rows': []}
        for label, only_on in (('с учётом отключений', True), ('все прогоны', False)):
            c = summarize(runs, ctl, field, only_on)
            e = summarize(runs, exp, field, only_on)
            block['rows'].append({
                'label': label, 'control': c, 'block': e,
                'delta': (e['mean'] - c['mean']) if (c and e) else None,
            })
        out.append(block)
    return out
