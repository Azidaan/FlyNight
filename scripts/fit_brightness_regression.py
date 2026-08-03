"""Регрессия «яркость видео → оптимальный порог» (Этап 8, подэтап 8.3).

Для каждого видео из union стресс-поднаборов перебираются пороги сетки
(ассоциация на кэше детекций + HOTA этого одного видео), выбирается
оптимальный, затем линейная регрессия по точкам (brightness, optimal_tau):

    tau_base(видео) = a * brightness + b

Результаты:
    experiments/results/brightness_optimal_tau.csv — per-video точки (кэшируется,
        повторный запуск досчитывает только недостающие видео);
    notes/brightness_to_threshold.md — коэффициенты a, b, R² (фиксация решения).

По плану низкий R² — не блокер: направление зависимости важнее точной
формулы (второй пункт новизны сравнивается с LQA-guided, а не с яркостью).

Использование (после построения кэша grid search-ем или three_modes):
    python scripts/fit_brightness_regression.py --cache runs/det_cache
    python scripts/fit_brightness_regression.py --cache runs/det_cache_q --subsets night rain
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from scripts.run_tracker_grid_search import subset_videos  # noqa: E402
from tracking.brightness_estimator import get_brightness_map  # noqa: E402
from tracking.brightness_regression import fit_brightness_to_threshold  # noqa: E402
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
NOTES = PROJECT_ROOT / 'notes'
POINTS_CSV = RESULTS / 'brightness_optimal_tau.csv'
WORK = PROJECT_ROOT / 'runs' / 'brightness_fit'
SUBSETS = ('normal', 'night', 'rain', 'fog')


def load_done() -> dict[str, tuple[float, float]]:
    """video_id -> (optimal_tau, hota) из прежних запусков."""
    if not POINTS_CSV.exists():
        return {}
    with open(POINTS_CSV, newline='', encoding='utf-8') as f:
        return {r['video_id']: (float(r['optimal_tau']), float(r['hota']))
                for r in csv.DictReader(f)}


def optimal_tau_for_video(cache_dir: str, vid: str, split: str,
                          thresholds: list[float]) -> tuple[float, float]:
    """Перебор порогов для одного видео; возвращает (лучший tau, HOTA)."""
    best = (thresholds[0], -1.0)
    for tau in thresholds:
        out_dir = WORK / f't{tau:g}'
        run_association(cache_dir, [vid], out_dir, split,
                        args=make_tracker_args(track_high_thresh=tau))
        try:
            m = evaluate_tracking_bdd(out_dir, [vid],
                                      out_dir / f'_eval_{vid}.json', split)
            hota = float(m.get('HOTA', -1.0))
        except Exception as e:  # повреждённое видео — пропускаем порог
            print(f'[WARN] {vid} tau={tau}: {e}')
            hota = -1.0
        if hota > best[1]:
            best = (tau, hota)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', required=True, help='директория кэша детекций')
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.3, 0.4, 0.5, 0.6])
    args = ap.parse_args()

    by_subset = subset_videos(args.split, args.subsets)
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    done = load_done()
    todo = [v for v in all_ids if v not in done]
    print(f'[INFO] видео: {len(all_ids)}, уже посчитано: {len(done)}, '
          f'осталось: {len(todo)}')

    POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not POINTS_CSV.exists()
    with open(POINTS_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['video_id', 'optimal_tau', 'hota'])
        for i, vid in enumerate(todo, 1):
            tau, hota = optimal_tau_for_video(args.cache, vid, args.split,
                                              args.thresholds)
            done[vid] = (tau, hota)
            w.writerow([vid, f'{tau:g}', f'{hota:.5f}'])
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f'[INFO] {i}/{len(todo)}', flush=True)

    # Регрессия по всем накопленным точкам
    bright = get_brightness_map(list(done), args.split)
    xs = [bright[v] for v in done]
    ys = [done[v][0] for v in done]
    a, b, r2 = fit_brightness_to_threshold(xs, ys)
    print(f'[OK ] tau = {a:.4f} * brightness + {b:.4f}   (R² = {r2:.3f}, '
          f'n = {len(xs)})')

    NOTES.mkdir(exist_ok=True)
    md = NOTES / 'brightness_to_threshold.md'
    md.write_text(f"""# Формула адаптации порога по яркости (Этап 8, подэтап 8.3)

Дата: авто-генерация fit_brightness_regression.py.

    tau_base(видео) = {a:.4f} * brightness + {b:.4f}

- R² = {r2:.3f}, точек (видео): {len(xs)}
- Сетка порогов: {args.thresholds}
- Поднаборы: {', '.join(args.subsets)} (split {args.split})
- Яркость: среднее V-канала HSV, нормированное в [0, 1]
  (tracking/brightness_estimator.py)
- Точки: experiments/results/brightness_optimal_tau.csv

Использование в режиме brightness (Этап 9):
    python scripts/run_tracking_three_modes.py --model ... --brightness-coef {a:.4f} {b:.4f}

Замечание плана: низкий R² не блокирует работу — важно направление
зависимости; формула — «базовый уровень» адаптации для сравнения
с LQA-guided (per-detection).
""", encoding='utf-8')
    print(f'[OK ] {md}')

    # машиночитаемая копия: отсюда коэффициенты берёт Этап 9 автоматически
    # (заметка выше — для человека, разбирать её регулярным выражением хрупко)
    coef_csv = RESULTS / 'brightness_coefficients.csv'
    coef_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(coef_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['a', 'b', 'r2', 'n_videos', 'split'])
        w.writerow([f'{a:.6f}', f'{b:.6f}', f'{r2:.4f}', len(xs), args.split])
    print(f'[OK ] {coef_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
