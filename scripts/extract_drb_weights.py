"""Извлечение финальных α₁, α₂ из обученных DRB-моделей (Этап 5).

Для каждого сида и каждого уровня пирамиды достаёт σ(w_LQA), σ(w_FGA)
из чекпойнта. Выход — experiments/results/drb_learned_weights.csv со
столбцами: run, seed, layer_idx, level, alpha_lqa, alpha_fga.

Уровни именуются P3/P4/P5 по порядку head.f (возрастание индекса слоя
соответствует убыванию пространственного разрешения).

Использование:
    python scripts/extract_drb_weights.py                 # drb_full, все сиды
    python scripts/extract_drb_weights.py --runs ablation_lqa_only ablation_fga_only
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo_neck import iter_drb  # noqa: E402

RESULTS = PROJECT_ROOT / 'experiments' / 'results'
SEEDS = (42, 123, 2024)
LEVELS = ('P3', 'P4', 'P5')


def extract_from_checkpoint(best_pt: Path) -> list[dict]:
    """[{layer_idx, level, alpha_lqa, alpha_fga}] из одного чекпойнта."""
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    rows = []
    pairs = sorted(iter_drb(model.model), key=lambda t: (t[0] is None, t[0]))
    for n, (idx, drb) in enumerate(pairs):
        a1, a2 = drb.alphas
        level = LEVELS[n] if len(pairs) == 3 else ('input' if idx == 0 else f'L{idx}')
        rows.append({'layer_idx': idx, 'level': level,
                     'alpha_lqa': round(a1, 6), 'alpha_fga': round(a2, 6)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', default=['drb_full'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--out', default=str(RESULTS / 'drb_learned_weights.csv'))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in args.runs:
        for seed in args.seeds:
            best = PROJECT_ROOT / 'weights' / f'{run}_seed{seed}' / 'best.pt'
            if not best.exists():
                print(f'[WARN] нет {best} — пропуск')
                continue
            for r in extract_from_checkpoint(best):
                rows.append({'run': run, 'seed': seed, **r})
                print(f'[OK ] {run} seed{seed} {r["level"]}: '
                      f'α₁={r["alpha_lqa"]:.4f} α₂={r["alpha_fga"]:.4f}')

    if not rows:
        print('[WARN] ни одного чекпойнта не найдено')
        return 1
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[OK ] {out_path} ({len(rows)} строк)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
