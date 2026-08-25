"""Корреляция confidence детектора и q_i (Этап 9, п. 9.5).

Эмпирическое подтверждение независимости сигналов — ключевой аргумент
защиты второго пункта новизны. Целевая интерпретация плана:
    Pearson/Spearman 0.3–0.6 — связаны, но не дублируют (нужный результат);
    > 0.9 — q_i дублирует confidence (пункт 2 под угрозой);
    < 0.2 — сигналы не связаны вовсе (подозрение на ошибку в LQA).

Данные берутся из q-кэша (inference_with_q) — отдельный инференс не нужен.
По плану достаточно поднабора Night (сотни тысяч точек).

Выходы:
    experiments/results/confidence_qi_correlation.csv  — pearson/spearman по поднаборам
    experiments/results/confidence_qi_pairs_night.csv  — сырые пары (для scatter)
    figs/tracking/q_vs_confidence_scatter.png          — scatter-плот
    experiments/results/divergence_frames.csv          — топ кадров с расхождением
                                                          |conf − q| (материал для защиты)

Использование:
    python scripts/correlation_experiment.py --model weights/drb_aug_full_seed42/best.pt
    python scripts/correlation_experiment.py --model ... --subsets night rain
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tracker_grid_search import subset_videos  # noqa: E402
from tracking.detection_cache import load_video_cache  # noqa: E402
from tracking.inference_with_q import build_detection_cache_with_q  # noqa: E402

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
FIGS = PROJECT_ROOT / 'figs' / 'tracking'
Q_CACHE = PROJECT_ROOT / 'runs' / 'det_cache_q'


def collect_pairs(split: str, video_ids: list[str], q_key: str,
                  min_conf: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Пары (conf, q) всех детекций поднабора + метки (video, frame_idx)."""
    confs, qs, frames = [], [], []
    for vid in video_ids:
        cache = load_video_cache(Q_CACHE, split, vid)
        if q_key not in cache:
            raise KeyError(f'{vid}: нет {q_key} в кэше — модель без DRB?')
        m = cache['conf'] >= min_conf
        confs.append(cache['conf'][m])
        qs.append(cache[q_key][m])
        frames.extend((vid, int(fi)) for fi in cache['frame_idx'][m])
    return np.concatenate(confs), np.concatenate(qs), frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=['night'])
    ap.add_argument('--agg', default='mean')
    ap.add_argument('--min-conf', type=float, default=0.1,
                    help='шумовые детекции ниже порога не учитываются')
    ap.add_argument('--top-divergent', type=int, default=30)
    ap.add_argument('--device', default=0,
                    help="0 — видеокарта, 'cpu' — мимо неё, чтобы не мешать "
                         'идущему обучению (обучения этот опыт не требует)')
    ap.add_argument('--max-videos', type=int, default=0,
                    help='ограничить число видео поднабора (0 — все); '
                         'для быстрой прикидки знака и разброса хватает 5')
    args = ap.parse_args()

    from scipy.stats import pearsonr, spearmanr

    by_subset = subset_videos(args.split, args.subsets)
    if args.max_videos:
        by_subset = {k: sorted(v)[:args.max_videos] for k, v in by_subset.items()}
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    build_detection_cache_with_q(args.model, all_ids, args.split, Q_CACHE,
                                 device=args.device)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    q_key = f'q_{args.agg}'

    corr_csv = RESULTS / 'confidence_qi_correlation.csv'
    with open(corr_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'n_pairs', 'pearson', 'pearson_p',
                         'spearman', 'spearman_p'])
        for subset in args.subsets:
            conf, q, frames = collect_pairs(args.split, by_subset[subset],
                                            q_key, args.min_conf)
            pr, pp = pearsonr(conf, q)
            sr, sp = spearmanr(conf, q)
            writer.writerow([subset, len(conf), f'{pr:.4f}', f'{pp:.2e}',
                             f'{sr:.4f}', f'{sp:.2e}'])
            print(f'[OK ] {subset}: n={len(conf)}, '
                  f'Pearson={pr:.3f}, Spearman={sr:.3f}')

            if subset == args.subsets[0]:
                # сырые пары (подвыборка до 200К строк для CSV разумного размера)
                idx = np.random.default_rng(42).permutation(len(conf))[:200_000]
                pairs_csv = RESULTS / f'confidence_qi_pairs_{subset}.csv'
                with open(pairs_csv, 'w', newline='', encoding='utf-8') as pf:
                    w2 = csv.writer(pf)
                    w2.writerow(['conf', 'q'])
                    w2.writerows(zip(conf[idx].round(5), q[idx].round(5)))

                # scatter
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.hexbin(conf, q, gridsize=60, bins='log', cmap='viridis')
                ax.set_xlabel('confidence детектора')
                ax.set_ylabel(f'q_i ({args.agg} по рамке)')
                ax.set_title(f'{subset}: Pearson={pr:.3f}, Spearman={sr:.3f} '
                             f'(n={len(conf)})')
                fig.tight_layout()
                fig.savefig(FIGS / 'q_vs_confidence_scatter.png', dpi=150)
                plt.close(fig)

                # топ дивергентных детекций: |conf − q| максимален
                div = np.abs(conf - q)
                top = np.argsort(-div)[:args.top_divergent]
                div_csv = RESULTS / 'divergence_frames.csv'
                with open(div_csv, 'w', newline='', encoding='utf-8') as df:
                    w3 = csv.writer(df)
                    w3.writerow(['video_id', 'frame_idx', 'conf', 'q',
                                 'divergence'])
                    for i in top:
                        vid, fi = frames[i]
                        w3.writerow([vid, fi, f'{conf[i]:.4f}',
                                     f'{q[i]:.4f}', f'{div[i]:.4f}'])
                print(f'[OK ] пары: {pairs_csv}')
                print(f'[OK ] scatter: {FIGS / "q_vs_confidence_scatter.png"}')
                print(f'[OK ] дивергентные кадры: {div_csv}')

    print(f'[OK ] {corr_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
