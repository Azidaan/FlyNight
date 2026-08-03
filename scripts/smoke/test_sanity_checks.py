"""Smoke-тест проверок адекватности (scripts/sanity_checks.py).

Проверки нужны для десятидневного автономного прогона без человека, поэтому
здесь они сами проверяются на синтетических артефактах: здоровое обучение
должно проходить без замечаний, а каждая типовая поломка — ловиться.

Запуск: python scripts/smoke/test_sanity_checks.py
"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sanity_checks as sc  # noqa: E402

FAILURES: list[str] = []
RESULT_COLS = ['epoch', 'time', 'train/box_loss', 'train/cls_loss',
               'metrics/precision(B)', 'metrics/recall(B)',
               'metrics/mAP50(B)', 'metrics/mAP50-95(B)']


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def levels(findings: list[sc.Finding]) -> str:
    return sc.worst_level(findings)


def texts(findings: list[sc.Finding]) -> str:
    return ' | '.join(f.text for f in findings)


# ————— конструирование синтетических артефактов —————

def make_checkpoint(path: Path, small: bool = False, corrupt: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if small:
        path.write_bytes(b'x' * 1024)
        return
    if corrupt:
        path.write_bytes(b'NOT-A-ZIP' * 700_000)          # крупный, но не архив
        return
    import torch
    torch.save({'model': torch.zeros(1_500_000), 'epoch': -1}, str(path))


def make_results(path: Path, epochs: int = 10, final_map: float = 0.45,
                 nan_at: int | None = None, rising_loss: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / 'args.yaml').write_text('epochs: 10\nbatch: 16\n', encoding='utf-8')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(RESULT_COLS)
        for e in range(1, epochs + 1):
            frac = e / max(epochs, 1)
            box = 1.6 + 0.3 * frac if rising_loss else 1.6 - 0.3 * frac
            m50 = final_map * frac
            if e == epochs:
                m50 = final_map
            row = [e, e * 100, f'{box:.5f}', '0.9', '0.7', '0.4',
                   f'{m50:.5f}', f'{m50 * 0.63:.5f}']
            if nan_at is not None and e == nan_at:
                row[2] = 'nan'
            w.writerow(row)


def make_alphas(path: Path, frozen: bool = False, out_of_range: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'layer_idx', 'alpha_lqa', 'alpha_fga'])
        for e in range(1, 11):
            a1 = 0.5 if frozen else 0.5 + 0.01 * e
            a2 = 1.0 if out_of_range else 0.4 + 0.005 * e
            w.writerow([e, 16, f'{a1:.6f}', f'{a2:.6f}'])


def make_summary(path: Path, drop_subset: str | None = None,
                 bad_value: bool = False, night_better: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {'normal': 0.46, 'night': 0.38, 'rain': 0.40, 'fog': 0.36}
    if night_better:
        values['night'] = 0.52
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'metric', 'mean', 'std', 'n_seeds'])
        for subset, value in values.items():
            if subset == drop_subset:
                continue
            w.writerow([subset, 'mAP50', '1.7' if bad_value else f'{value:.5f}',
                        '0.01', 3])
        for stress in ('night', 'rain', 'fog'):
            w.writerow([f'delta_{stress}', 'mAP50', '-0.08', '0.01', 3])


def make_grid(path: Path, best_at_edge: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'track_high_thresh', 'track_buffer', 'HOTA',
                    'IDF1', 'MOTA', 'IDSw'])
        for thr in (0.3, 0.4, 0.5):
            for buf in (15, 30, 60):
                # оптимум либо в углу сетки, либо в середине
                hota = (40 + thr * 10 + buf * 0.1) if best_at_edge else (
                    50 - abs(thr - 0.4) * 20 - abs(buf - 30) * 0.1)
                w.writerow(['night', thr, buf, f'{hota:.2f}', '55', '48', '120'])


# ————— сами проверки —————

def test_training(root: Path) -> None:
    run = 'baseline'
    weights = root / 'weights' / f'{run}_seed42' / 'best.pt'
    results = root / 'runs' / run / 'seed42' / 'results.csv'

    make_checkpoint(weights)
    make_results(results)
    f = sc.check_training(run, (42,), root)
    check('здоровое обучение — без замечаний', levels(f) == 'ok', texts(f))

    make_results(results, nan_at=5)
    f = sc.check_training(run, (42,), root)
    check('NaN в лоссе ловится', levels(f) == 'fail'
          and any('NaN' in x.text for x in f))

    make_results(results, epochs=3)
    f = sc.check_training(run, (42,), root)
    check('оборванное обучение ловится', levels(f) == 'fail'
          and any('оборвалось' in x.text for x in f))

    make_results(results, final_map=0.05)
    f = sc.check_training(run, (42,), root)
    check('развалившееся обучение (низкий mAP) ловится', levels(f) == 'fail'
          and any('ниже' in x.text for x in f))

    make_results(results, rising_loss=True)
    f = sc.check_training(run, (42,), root)
    check('растущий лосс ловится', levels(f) == 'fail'
          and any('не убыл' in x.text for x in f))

    make_results(results)
    make_checkpoint(weights, small=True)
    f = sc.check_training(run, (42,), root)
    check('обрезанный чекпойнт ловится', levels(f) == 'fail'
          and any('обрыв записи' in x.text for x in f))

    make_checkpoint(weights, corrupt=True)
    f = sc.check_training(run, (42,), root)
    check('битый чекпойнт ловится', levels(f) == 'fail'
          and any('битый' in x.text for x in f))

    # разброс между сидами
    make_checkpoint(weights)
    for seed, m in ((123, 0.45), (2024, 0.30)):
        make_checkpoint(root / 'weights' / f'{run}_seed{seed}' / 'best.pt')
        make_results(root / 'runs' / run / f'seed{seed}' / 'results.csv', final_map=m)
    f = sc.check_training(run, (42, 123, 2024), root)
    check('разброс между сидами замечается', levels(f) == 'warn'
          and any('разброс' in x.text for x in f), texts(f))


def test_alphas(root: Path) -> None:
    path = root / 'runs' / 'drb_full' / 'seed42' / 'drb_alphas.csv'
    make_alphas(path)
    f = sc.check_drb_alphas('drb_full', (42,), root)
    check('живые α проходят', levels(f) == 'ok', texts(f))

    make_alphas(path, frozen=True)
    f = sc.check_drb_alphas('drb_full', (42,), root)
    check('примёрзшая α замечается', levels(f) == 'warn'
          and any('не менялась' in x.text for x in f))

    make_alphas(path, out_of_range=True)
    f = sc.check_drb_alphas('drb_full', (42,), root)
    check('α вне (0,1) ловится', levels(f) == 'fail')


def test_summary(root: Path) -> None:
    path = root / 'experiments' / 'results' / 'baseline_summary.csv'
    make_summary(path)
    f = sc.check_summary('baseline', 3, root)
    check('нормальная сводка проходит', levels(f) == 'ok', texts(f))

    make_summary(path, drop_subset='fog')
    f = sc.check_summary('baseline', 3, root)
    check('потерянный поднабор ловится', levels(f) == 'fail'
          and any('fog' in x.text for x in f))

    make_summary(path, bad_value=True)
    f = sc.check_summary('baseline', 3, root)
    check('mAP вне [0,1] ловится', levels(f) == 'fail'
          and any('вне [0,1]' in x.text for x in f))

    make_summary(path, night_better=True)
    f = sc.check_summary('baseline', 3, root)
    check('ночь лучше дня — замечание', levels(f) == 'warn'
          and any('ночь' in x.text for x in f), texts(f))


def test_collapse(root: Path) -> None:
    res = root / 'experiments' / 'results'
    make_summary(res / 'baseline_summary.csv')
    make_summary(res / 'drb_full_summary.csv')
    f = sc.check_no_collapse('drb_full', root=root)
    check('одинаковые конфигурации — без замечаний', levels(f) == 'ok', texts(f))

    with open(res / 'drb_full_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['subset', 'metric', 'mean', 'std', 'n_seeds'])
        for subset in sc.SUBSETS:
            w.writerow([subset, 'mAP50', '0.25', '0.01', 3])
    f = sc.check_no_collapse('drb_full', root=root)
    check('обвал относительно baseline ловится', levels(f) == 'fail'
          and any('хуже baseline' in x.text for x in f))


def test_grid(root: Path) -> None:
    path = root / 'experiments' / 'results' / 'tracking_param_search.csv'
    make_grid(path, best_at_edge=False)
    f = sc.check_tracking_grid(root)
    check('оптимум внутри сетки — без замечаний', levels(f) == 'ok', texts(f))

    make_grid(path, best_at_edge=True)
    f = sc.check_tracking_grid(root)
    check('оптимум на краю сетки — замечание', levels(f) == 'warn'
          and any('краю сетки' in x.text for x in f), texts(f))


def make_modes_table(path: Path, modes=('standard', 'brightness', 'lqa',
                                        'control_global'),
                     with_params: bool = True, bad_hota: bool = False) -> None:
    """Длинный формат (mode, subset, metric, value) — как пишет реальный скрипт."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'subset', 'metric', 'value'])
        for mode in modes:
            for subset in ('normal', 'night'):
                hota = 140.0 if bad_hota else (48.0 if mode == 'standard' else 50.0)
                w.writerow([mode, subset, 'HOTA', f'{hota:.2f}'])
                w.writerow([mode, subset, 'IDF1', '55.0'])
                if with_params:
                    w.writerow([mode, subset, 'tau_used', '0.4'])
                    w.writerow([mode, subset, 'buffer_used', '30'])


def test_modes(root: Path) -> None:
    path = root / 'experiments' / 'results' / 'tracking_three_modes.csv'
    make_modes_table(path)
    f = sc.check_tracking_modes(root)
    check('полная таблица режимов проходит', levels(f) == 'ok', texts(f))

    make_modes_table(path, modes=('standard', 'lqa', 'control_global'))
    f = sc.check_tracking_modes(root)
    check('пропавший режим brightness ловится', levels(f) == 'fail'
          and any('brightness' in x.text for x in f))

    make_modes_table(path, with_params=False)
    f = sc.check_tracking_modes(root)
    check('расчёт без подобранного порога ловится', levels(f) == 'fail'
          and any('tau_used' in x.text for x in f))

    make_modes_table(path, bad_hota=True)
    f = sc.check_tracking_modes(root)
    check('HOTA вне [0,100] ловится', levels(f) == 'fail'
          and any('вне [0,100]' in x.text for x in f))


def test_final(root: Path) -> None:
    res = root / 'experiments' / 'results'
    (res / 'final').mkdir(parents=True, exist_ok=True)
    (res / 'final' / 'main_results.csv').write_text('a,b\n1,2\n', encoding='utf-8')
    perf = res / 'performance_table.csv'
    cols = ['model', 'fps_b1', 'latency_ms_b1', 'gpu_mem_mb_b1']

    with open(perf, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(cols); w.writerow(['baseline', '237.4', '4.2', '900'])
    f = sc.check_final(root)
    check('нормальная таблица производительности проходит', levels(f) == 'ok',
          texts(f))

    with open(perf, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(cols); w.writerow(['baseline', '0', '4.2', '900'])
    f = sc.check_final(root)
    check('нулевой FPS ловится', levels(f) == 'fail'
          and any('замер не удался' in x.text for x in f))


def test_tuned_params(root: Path) -> None:
    """Оптимум читается из артефактов, а не берётся по умолчанию."""
    from scripts import tuned_params as tp
    res = root / 'experiments' / 'results'
    res.mkdir(parents=True, exist_ok=True)

    check('без артефактов — умолчание',
          tp.resolve_tau('night', root=root)[0] == tp.DEFAULT_TAU
          and tp.resolve_k(root=root)[0] == tp.DEFAULT_K)

    with open(res / 'tracking_param_search.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'track_high_thresh', 'track_buffer', 'HOTA'])
        w.writerow(['night', '0.3', '15', '41.0'])
        w.writerow(['night', '0.6', '60', '47.5'])       # лучший для night
        w.writerow(['normal', '0.4', '30', '52.0'])      # лучший для normal
        w.writerow(['normal', '0.6', '60', '50.0'])
    tau, buf, src = tp.resolve_tau('night', root=root)
    check('τ берётся из сетки для своего поднабора',
          (tau, buf) == (0.6, 60), f'{tau}/{buf} ({src})')
    tau_n, buf_n, _ = tp.resolve_tau('normal', root=root)
    check('у другого поднабора свой оптимум', (tau_n, buf_n) == (0.4, 30),
          f'{tau_n}/{buf_n}')
    check('ручное значение перекрывает подобранное',
          tp.resolve_tau('night', override=0.55, root=root)[0] == 0.55)

    with open(res / 'lqa_k_sweep.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['k', 'subset', 'tau_base', 'track_buffer', 'HOTA'])
        for k, h_norm, h_night in ((0.0, 50.0, 46.0), (0.3, 51.5, 49.0),
                                   (0.4, 49.0, 45.0)):
            w.writerow([k, 'normal', '0.4', 30, h_norm])
            w.writerow([k, 'night', '0.6', 60, h_night])
    k, src = tp.resolve_k(root=root)
    check('k берётся по лучшему среднему HOTA', k == 0.3, f'{k} ({src})')

    with open(res / 'brightness_coefficients.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['a', 'b', 'r2', 'n_videos', 'split'])
        w.writerow(['0.5500', '0.2500', '0.31', '120', 'val'])
    check('коэффициенты яркости читаются',
          tp.brightness_coefficients(root) == (0.55, 0.25))

    (res / 'brightness_coefficients.csv').unlink()
    notes = root / 'notes'
    notes.mkdir(parents=True, exist_ok=True)
    (notes / 'brightness_to_threshold.md').write_text(
        'ерунда\n\n    tau_base(видео) = 0.4200 * brightness + 0.1900\n\nещё\n',
        encoding='utf-8')
    check('запасной разбор заметки работает',
          tp.brightness_coefficients(root) == (0.42, 0.19))


def test_runner_dry_run() -> None:
    """Автономный прогон видит все шаги и не падает на разборе аргументов."""
    import subprocess
    py = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
    proc = subprocess.run(
        [str(py), str(PROJECT_ROOT / 'scripts' / 'run_full_pipeline.py'), '--dry-run'],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=180)
    check('--dry-run отрабатывает', proc.returncode == 0,
          (proc.stderr or '')[-200:])
    for step in ('baseline', 'eval_baseline', 'track_modes', 'final'):
        check(f'--dry-run показывает шаг {step}', step in proc.stdout)

    proc = subprocess.run(
        [str(py), str(PROJECT_ROOT / 'scripts' / 'run_full_pipeline.py'),
         '--only', 'нет-такого-шага'],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=180)
    check('неизвестный шаг отвергается', proc.returncode == 2)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='sanity_', dir=str(PROJECT_ROOT / 'runs')))
    try:
        print('=== обучение ===');      test_training(tmp)
        print('=== α блока DRB ===');   test_alphas(tmp)
        print('=== сводки оценки ==='); test_summary(tmp)
        print('=== обвал против baseline ==='); test_collapse(tmp)
        print('=== сетка трекера ===');  test_grid(tmp)
        print('=== режимы трекинга ==='); test_modes(tmp)
        print('=== финальные таблицы ==='); test_final(tmp)
        print('=== подобранные параметры ==='); test_tuned_params(tmp)
        print('=== автономный прогон ==='); test_runner_dry_run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\n--- {"OK: проверки адекватности работают" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
