"""Проверки адекватности результатов после каждого шага конвейера.

Нужны для автономного прогона (scripts/run_full_pipeline.py): десять суток
обучений без человека — это десять суток, за которые тихая деградация
(NaN в лоссе, оборванный сид, вырожденные α, метрика в неверном диапазоне)
успеет утащить за собой все последующие шаги. Проверки читают только файлы
на диске, GPU не трогают — их безопасно запускать параллельно с обучением.

Уровни находок:
    fail — результат негоден, автономный прогон останавливается;
    warn — подозрительно, но продолжать можно (пишется в журнал);
    ok   — проверка пройдена (печатается для наглядности).

Осознанное ограничение: проверки ловят поломки, а не подтверждают научный
результат. «DRB лучше baseline» проверкой быть не может — это исход
эксперимента; здесь ловится только катастрофическое падение, означающее
ошибку в коде или сорванное обучение.

Использование:
    python scripts/sanity_checks.py --step baseline
    python scripts/sanity_checks.py --all
"""
import argparse
import csv
import math
import sys
import zipfile
from pathlib import Path
from typing import Callable, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS3 = (42, 123, 2024)
SUBSETS = ('normal', 'night', 'rain', 'fog')

# Абсолютный «пол» mAP50: ниже — обучение развалилось, а не «получилось хуже».
# Здоровый baseline на BDD100K MOT к 28-й эпохе давал 0.44, финальный ~0.45-0.55.
MAP50_FLOOR = 0.20
# Разброс финального mAP50 между сидами, выше которого стоит присмотреться.
SEED_SPREAD_WARN = 0.05
# Доля эпох от запланированных, ниже которой обучение считается оборванным.
MIN_EPOCH_FRACTION = 0.9
# Минимальный размер чекпойнта: обрезанный файл меньше.
MIN_CKPT_MB = 5.0


class Finding(NamedTuple):
    level: str          # 'ok' | 'warn' | 'fail'
    text: str


def ok(t: str) -> Finding:
    return Finding('ok', t)


def warn(t: str) -> Finding:
    return Finding('warn', t)


def fail(t: str) -> Finding:
    return Finding('fail', t)


# ————— вспомогательные чтения —————

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _f(value: str | None) -> float | None:
    """float или None (пустая строка, 'nan', мусор)."""
    if value is None or value == '':
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _planned_epochs(run_dir: Path) -> int | None:
    """Сколько эпох было заказано — из args.yaml, который пишет Ultralytics."""
    args_yaml = run_dir / 'args.yaml'
    if not args_yaml.exists():
        return None
    for line in args_yaml.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('epochs:'):
            try:
                return int(line.split(':', 1)[1].strip())
            except ValueError:
                return None
    return None


def _checkpoint_findings(ckpt: Path) -> list[Finding]:
    """Целостность чекпойнта: размер, zip-структура, загрузка torch."""
    out: list[Finding] = []
    size_mb = ckpt.stat().st_size / 1024 / 1024
    if size_mb < MIN_CKPT_MB:
        return [fail(f'{ckpt.name}: файл {size_mb:.1f} МБ — похоже на обрыв записи')]
    if not zipfile.is_zipfile(ckpt):
        return [fail(f'{ckpt.name}: не открывается как архив — чекпойнт битый')]
    try:
        import torch
    except ImportError:
        out.append(warn(f'{ckpt.name}: torch недоступен — глубокая проверка пропущена'))
        return out
    try:
        ck = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    except Exception as e:                                  # noqa: BLE001
        return [fail(f'{ckpt.name}: torch.load упал — {type(e).__name__}: {e}')]
    if not isinstance(ck, dict) or 'model' not in ck:
        return [fail(f'{ckpt.name}: в чекпойнте нет ключа model')]
    out.append(ok(f'{ckpt.name}: {size_mb:.0f} МБ, загружается'))
    return out


# ————— проверки обучающих шагов —————

def check_training(run_name: str, seeds: tuple[int, ...],
                   root: Path = PROJECT_ROOT) -> list[Finding]:
    """Веса целы, эпохи добежали, лоссы без NaN, mAP выше пола, сиды согласны."""
    out: list[Finding] = []
    finals: dict[int, float] = {}

    for seed in seeds:
        ckpt = root / 'weights' / f'{run_name}_seed{seed}' / 'best.pt'
        if not ckpt.exists():
            out.append(fail(f'сид {seed}: нет весов {ckpt.relative_to(root)}'))
            continue
        out.extend(_checkpoint_findings(ckpt))

        run_dir = root / 'runs' / run_name / f'seed{seed}'
        results = run_dir / 'results.csv'
        if not results.exists():
            out.append(fail(f'сид {seed}: нет журнала обучения results.csv'))
            continue
        rows = _read_csv(results)
        if not rows:
            out.append(fail(f'сид {seed}: results.csv пуст'))
            continue

        planned = _planned_epochs(run_dir)
        if planned and len(rows) < planned * MIN_EPOCH_FRACTION:
            out.append(fail(f'сид {seed}: пройдено {len(rows)} эпох из {planned} '
                            f'— обучение оборвалось'))
        elif planned:
            out.append(ok(f'сид {seed}: {len(rows)}/{planned} эпох'))

        # NaN/inf в любой числовой колонке — верный признак разошедшегося обучения
        bad_cols = set()
        for row in rows:
            for key, value in row.items():
                if key in ('epoch', 'time') or value in (None, ''):
                    continue
                if _f(value) is None:
                    bad_cols.add(key)
        if bad_cols:
            out.append(fail(f'сид {seed}: NaN/inf в колонках {sorted(bad_cols)}'))

        last, first = rows[-1], rows[0]
        map50 = _f(last.get('metrics/mAP50(B)'))
        if map50 is None:
            out.append(fail(f'сид {seed}: в results.csv нет финального mAP50'))
        else:
            finals[seed] = map50
            if map50 < MAP50_FLOOR:
                out.append(fail(f'сид {seed}: финальный mAP50={map50:.3f} ниже '
                                f'порога {MAP50_FLOOR} — обучение развалилось'))
            else:
                out.append(ok(f'сид {seed}: финальный mAP50={map50:.3f}'))

        box_first, box_last = _f(first.get('train/box_loss')), _f(last.get('train/box_loss'))
        if box_first is not None and box_last is not None and box_last >= box_first:
            out.append(fail(f'сид {seed}: train/box_loss не убыл '
                            f'({box_first:.3f} → {box_last:.3f})'))

        # финал заметно хуже лучшей эпохи — не фатально (best.pt берёт лучшую),
        # но означает переобучение, о котором стоит знать
        best_seen = max((_f(r.get('metrics/mAP50(B)')) or 0.0) for r in rows)
        if map50 is not None and best_seen - map50 > 0.03:
            out.append(warn(f'сид {seed}: финал {map50:.3f} против лучшей эпохи '
                            f'{best_seen:.3f} — признак переобучения'))

    if len(finals) > 1:
        spread = max(finals.values()) - min(finals.values())
        text = ('разброс mAP50 между сидами '
                f'{spread:.3f} ({", ".join(f"{s}:{v:.3f}" for s, v in finals.items())})')
        out.append(warn(text + ' — необычно много') if spread > SEED_SPREAD_WARN
                   else ok(text))
    return out


def check_drb_alphas(run_name: str, seeds: tuple[int, ...],
                     root: Path = PROJECT_ROOT) -> list[Finding]:
    """α ветвей DRB: в (0,1), не вырождены, не примёрзли к начальному значению."""
    out: list[Finding] = []
    for seed in seeds:
        path = root / 'runs' / run_name / f'seed{seed}' / 'drb_alphas.csv'
        if not path.exists():
            out.append(warn(f'сид {seed}: нет drb_alphas.csv — динамика α не записана'))
            continue
        rows = _read_csv(path)
        if not rows:
            out.append(warn(f'сид {seed}: drb_alphas.csv пуст'))
            continue
        for col in ('alpha_lqa', 'alpha_fga'):
            vals = [_f(r.get(col)) for r in rows]
            if any(v is None for v in vals):
                out.append(fail(f'сид {seed}: NaN в {col}'))
                continue
            lo, hi = min(vals), max(vals)
            if lo <= 0.0 or hi >= 1.0:
                out.append(fail(f'сид {seed}: {col} вышла за (0,1): [{lo:.3f}, {hi:.3f}]'))
            elif hi - lo < 1e-4:
                out.append(warn(f'сид {seed}: {col} не менялась за обучение '
                                f'({lo:.4f}) — ветвь могла не получать градиент'))
            else:
                out.append(ok(f'сид {seed}: {col} ∈ [{lo:.3f}, {hi:.3f}]'))
    return out


# ————— проверки шагов оценки —————

def check_summary(run_name: str, expected_seeds: int,
                  root: Path = PROJECT_ROOT) -> list[Finding]:
    """Сводная таблица оценки: поднаборы на месте, метрики в диапазоне."""
    out: list[Finding] = []
    path = root / 'experiments' / 'results' / f'{run_name}_summary.csv'
    if not path.exists():
        return [fail(f'нет таблицы {path.name}')]
    rows = _read_csv(path)
    if not rows:
        return [fail(f'{path.name} пуст')]

    seen = {(r['subset'], r['metric']): r for r in rows}
    for subset in SUBSETS:
        key = (subset, 'mAP50')
        if key not in seen:
            out.append(fail(f'{path.name}: нет строки {subset}/mAP50'))
            continue
        mean = _f(seen[key].get('mean'))
        if mean is None:
            out.append(fail(f'{path.name}: {subset}/mAP50 не число'))
        elif not 0.0 <= mean <= 1.0:
            out.append(fail(f'{path.name}: {subset}/mAP50={mean:.3f} вне [0,1]'))
        elif mean < MAP50_FLOOR:
            out.append(fail(f'{path.name}: {subset}/mAP50={mean:.3f} ниже пола'))
        else:
            out.append(ok(f'{subset}: mAP50={mean:.3f}'))

    n_seeds = {_f(r.get('n_seeds')) for r in rows if r.get('n_seeds')}
    if n_seeds and max(n_seeds) < expected_seeds:
        out.append(warn(f'{path.name}: усреднение по {int(max(n_seeds))} сидам '
                        f'вместо {expected_seeds}'))

    for stress in ('night', 'rain', 'fog'):
        if (f'delta_{stress}', 'mAP50') not in seen:
            out.append(warn(f'{path.name}: нет Δ-строки для {stress}'))

    # AP по размерам не может быть кратно ниже общего mAP50-95: если это так,
    # значит эталон COCO и предсказания разошлись в соглашениях (нумерация
    # классов, координаты) и метрика считает мимо.
    overall = _f(seen.get(('normal', 'mAP50_95'), {}).get('mean'))
    ap_large = _f(seen.get(('normal', 'AP_large'), {}).get('mean'))
    if overall and ap_large is not None:
        if ap_large < overall * 0.3:
            out.append(fail(f'AP_large={ap_large:.4f} при mAP50-95={overall:.3f} — '
                            f'метрики по размерам считаются мимо эталона'))
        else:
            out.append(ok(f'AP по размерам согласован с общим (large {ap_large:.3f})'))

    # содержательная проверка: сложные условия должны быть не легче обычных
    normal = _f(seen.get(('normal', 'mAP50'), {}).get('mean'))
    night = _f(seen.get(('night', 'mAP50'), {}).get('mean'))
    if normal is not None and night is not None and night > normal + 0.02:
        out.append(warn(f'ночь ({night:.3f}) заметно лучше обычных условий '
                        f'({normal:.3f}) — проверить состав поднаборов'))
    return out


def check_tables(names: list[str], root: Path = PROJECT_ROOT) -> list[Finding]:
    """Таблицы сравнения существуют, парсятся и не пусты."""
    out: list[Finding] = []
    for name in names:
        path = root / 'experiments' / 'results' / name
        if not path.exists():
            out.append(fail(f'нет таблицы {name}'))
            continue
        rows = _read_csv(path)
        out.append(ok(f'{name}: {len(rows)} строк') if rows
                   else fail(f'{name} пуст'))
    return out


def check_no_collapse(run_name: str, baseline: str = 'baseline',
                      max_drop: float = 0.05,
                      root: Path = PROJECT_ROOT) -> list[Finding]:
    """Конфигурация не рухнула относительно baseline (ловит ошибку, не исход).

    Улучшение здесь НЕ требуется: превзойдёт DRB baseline или нет — это
    результат эксперимента. Проверяется только катастрофическое падение.
    """
    res = root / 'experiments' / 'results'
    try:
        cur = {r['subset']: _f(r['mean']) for r in _read_csv(res / f'{run_name}_summary.csv')
               if r['metric'] == 'mAP50'}
        base = {r['subset']: _f(r['mean']) for r in _read_csv(res / f'{baseline}_summary.csv')
                if r['metric'] == 'mAP50'}
    except (FileNotFoundError, KeyError):
        return [warn(f'{run_name}: не с чем сравнить — сводка baseline недоступна')]

    out: list[Finding] = []
    for subset in SUBSETS:
        a, b = cur.get(subset), base.get(subset)
        if a is None or b is None:
            continue
        delta = a - b
        if delta < -max_drop:
            out.append(fail(f'{subset}: {run_name} хуже baseline на {-delta:.3f} '
                            f'mAP50 — похоже на ошибку интеграции'))
        else:
            out.append(ok(f'{subset}: Δ к baseline {delta:+.3f}'))
    return out


# ————— проверки трекинга и финала —————

def check_tracking_grid(root: Path = PROJECT_ROOT) -> list[Finding]:
    """Сетка порогов: метрики в диапазоне, оптимум не упёрся в край сетки."""
    out: list[Finding] = []
    path = root / 'experiments' / 'results' / 'tracking_param_search.csv'
    if not path.exists():
        return [fail('нет tracking_param_search.csv')]
    rows = _read_csv(path)
    if not rows:
        return [fail('tracking_param_search.csv пуст')]

    hotas = [_f(r.get('HOTA')) for r in rows]
    if any(h is None for h in hotas):
        out.append(fail('в сетке есть строки без HOTA'))
    valid = [h for h in hotas if h is not None]
    if valid and (min(valid) < 0 or max(valid) > 100):
        out.append(fail(f'HOTA вне [0,100]: [{min(valid):.2f}, {max(valid):.2f}]'))
    elif valid:
        out.append(ok(f'{len(rows)} конфигураций, HOTA ∈ '
                      f'[{min(valid):.2f}, {max(valid):.2f}]'))

    # оптимум на границе сетки означает, что искать надо было шире
    for subset in {r['subset'] for r in rows}:
        sub = [r for r in rows if r['subset'] == subset and _f(r.get('HOTA')) is not None]
        if len(sub) < 2:
            continue
        best = max(sub, key=lambda r: _f(r['HOTA']))
        thr = [_f(r['track_high_thresh']) for r in sub]
        buf = [_f(r['track_buffer']) for r in sub]
        b_thr, b_buf = _f(best['track_high_thresh']), _f(best['track_buffer'])
        edges = []
        if b_thr in (min(thr), max(thr)):
            edges.append(f'порог {b_thr:g}')
        if b_buf in (min(buf), max(buf)):
            edges.append(f'буфер {b_buf:g}')
        if edges:
            out.append(warn(f'{subset}: оптимум на краю сетки ({", ".join(edges)}) '
                            f'— возможно, стоит расширить диапазон'))
    return out


EXPECTED_MODES = ('standard', 'brightness', 'lqa', 'control_global')


def check_tracking_modes(root: Path = PROJECT_ROOT) -> list[Finding]:
    """Режимы трекинга: все четыре на месте, посчитаны подобранными параметрами.

    Таблица в длинном формате (mode, subset, metric, value) — метрика лежит
    в строке, а не в колонке.
    """
    out: list[Finding] = []
    path = root / 'experiments' / 'results' / 'tracking_three_modes.csv'
    if not path.exists():
        return [fail('нет tracking_three_modes.csv')]
    rows = _read_csv(path)
    if not rows:
        return [fail('tracking_three_modes.csv пуст')]

    modes = {r.get('mode') for r in rows if r.get('mode')}
    missing = [m for m in EXPECTED_MODES if m not in modes]
    if missing:
        out.append(fail(f'не посчитаны режимы: {", ".join(missing)} — '
                        f'сравнение неполное'))
    else:
        out.append(ok(f'все четыре режима на месте ({len(rows)} строк)'))

    hota: dict[tuple[str, str], float] = {}
    for r in rows:
        value = _f(r.get('value'))
        mode, subset, metric = r.get('mode'), r.get('subset'), r.get('metric')
        if metric == 'HOTA':
            if value is None:
                out.append(fail(f'{mode}/{subset}: HOTA не число'))
            elif not 0 <= value <= 100:
                out.append(fail(f'{mode}/{subset}: HOTA={value} вне [0,100]'))
            else:
                hota[(mode, subset)] = value

    # параметры должны быть записаны — доказательство, что автоподбор сработал
    tagged = {(r.get('mode'), r.get('subset')) for r in rows
              if r.get('metric') == 'tau_used'}
    pairs = {(m, s) for (m, s) in hota}
    if pairs - tagged:
        out.append(fail('в таблице нет строк tau_used — режимы посчитаны '
                        'старой версией скрипта, без подобранного порога'))
    else:
        taus = {r.get('value') for r in rows if r.get('metric') == 'tau_used'}
        out.append(ok(f'использованные пороги: {", ".join(sorted(taus))}'))

    by_mode: dict[str, list[float]] = {}
    for (mode, _subset), value in hota.items():
        by_mode.setdefault(mode, []).append(value)
    means = {m: sum(v) / len(v) for m, v in by_mode.items() if v}
    for mode, value in sorted(means.items(), key=lambda kv: -kv[1]):
        out.append(ok(f'{mode}: средний HOTA {value:.2f}'))
    if 'standard' in means:
        for mode, value in means.items():
            if mode != 'standard' and value < means['standard'] - 5:
                out.append(warn(f'режим {mode} хуже standard на '
                                f'{means["standard"] - value:.1f} HOTA — проверить'))
    return out


def check_final(root: Path = PROJECT_ROOT) -> list[Finding]:
    """Финальные таблицы: производительность положительная, сводка не пуста.

    Колонки performance_table.csv задаёт utils/benchmark.py:
    model, fps_b1, latency_ms_b1, ... — отсюда имена ниже.
    """
    out = check_tables(['performance_table.csv', 'final/main_results.csv'], root)
    perf = root / 'experiments' / 'results' / 'performance_table.csv'
    if perf.exists():
        for r in _read_csv(perf):
            name = r.get('model', '?')
            fps = _f(r.get('fps_b1'))
            if fps is None:
                out.append(fail(f'{name}: в таблице нет числового fps_b1'))
            elif fps <= 0:
                out.append(fail(f'{name}: FPS={fps} — замер не удался'))
            else:
                out.append(ok(f'{name}: {fps:.1f} FPS (батч 1)'))
    return out


# ————— карта проверок по шагам конвейера —————

CHECKS: dict[str, Callable[[], list[Finding]]] = {
    'baseline': lambda: check_training('baseline', SEEDS3),
    'eval_baseline': lambda: check_summary('baseline', 3),
    'drb': lambda: check_training('drb_full', SEEDS3) + check_drb_alphas('drb_full', SEEDS3),
    'eval_drb': lambda: (check_summary('drb_full', 3)
                         + check_no_collapse('drb_full')
                         + check_tables(['drb_vs_baseline.csv'])),
    'drb_aug': lambda: (check_training('drb_aug_full', SEEDS3)
                        + check_drb_alphas('drb_aug_full', SEEDS3)),
    'eval_drb_aug': lambda: (check_summary('drb_aug_full', 3)
                             + check_no_collapse('drb_aug_full')
                             + check_tables(['drb_aug_vs_drb_vs_baseline.csv'])),
    'ablation': lambda: (check_training('ablation_lqa_only', (42, 123))
                         + check_training('ablation_fga_only', (42, 123))
                         + check_training('ablation_drb_input', (42,))),
    'eval_ablation': lambda: (check_summary('ablation_lqa_only', 2)
                              + check_summary('ablation_fga_only', 2)
                              + check_tables(['ablation_synergy.csv',
                                              'ablation_paradigm.csv'])),
    'track_grid': lambda: (check_tracking_grid()
                           + check_tables(['brightness_optimal_tau.csv'])),
    'track_modes': lambda: (check_tracking_modes()
                            + check_tables(['lqa_k_sweep.csv',
                                            'confidence_qi_correlation.csv',
                                            'q_aggregation_ablation.csv'])),
    'final': lambda: check_final(),
}


def check_step(step_id: str) -> list[Finding]:
    """Находки по шагу. Неизвестный шаг — пустой список (не мешаем прогону)."""
    checker = CHECKS.get(step_id)
    if checker is None:
        return []
    try:
        return checker()
    except Exception as e:                                  # noqa: BLE001
        return [fail(f'проверка упала: {type(e).__name__}: {e}')]


def format_findings(findings: list[Finding]) -> list[str]:
    marks = {'ok': '[OK ]', 'warn': '[WARN]', 'fail': '[FAIL]'}
    return [f'{marks[f.level]} {f.text}' for f in findings]


def worst_level(findings: list[Finding]) -> str:
    if any(f.level == 'fail' for f in findings):
        return 'fail'
    return 'warn' if any(f.level == 'warn' for f in findings) else 'ok'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step', help=f'шаг конвейера: {", ".join(CHECKS)}')
    ap.add_argument('--all', action='store_true', help='все шаги подряд')
    args = ap.parse_args()

    if not args.step and not args.all:
        ap.error('нужен --step <id> или --all')
    steps = list(CHECKS) if args.all else [args.step]

    worst = 'ok'
    for step in steps:
        if step not in CHECKS:
            print(f'[WARN] неизвестный шаг {step}')
            continue
        findings = check_step(step)
        print(f'\n=== {step} ===')
        for line in format_findings(findings):
            print(' ', line)
        level = worst_level(findings)
        if level == 'fail' or (level == 'warn' and worst == 'ok'):
            worst = level
    print(f'\n--- итог проверок: {worst.upper()} ---')
    return 1 if worst == 'fail' else 0


if __name__ == '__main__':
    raise SystemExit(main())
