"""Кривые обучения и динамика весов ветвей DRB (Этап 10, подэтап 10.3).

Два типа графиков:
1. mAP50-95 по эпохам для нескольких конфигураций на одном полотне
   (results.csv Ultralytics) — сходимость baseline vs DRB vs DRB+aug;
2. динамика α₁ = σ(w_LQA), α₂ = σ(w_FGA) по эпохам и уровням пирамиды
   (drb_alphas.csv из train_drb.py) — аргумент специализации ветвей.

Использование:
    python visualization/plot_training_curves.py --curves baseline drb_full
    python visualization/plot_training_curves.py --alphas drb_full --seed 42
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGS = PROJECT_ROOT / 'figs' / 'final'
SEEDS = (42, 123, 2024)
LEVEL_BY_ORDER = ('P3', 'P4', 'P5')


def read_results_csv(path: Path) -> tuple[list[int], list[float]]:
    """(эпохи, mAP50-95) из results.csv Ultralytics."""
    epochs, maps = [], []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = next((k for k in r if 'mAP50-95' in k), None)
            if key is None:
                continue
            epochs.append(int(float(r['epoch'])))
            maps.append(float(r[key]))
    return epochs, maps


def plot_curves(runs: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        for seed in SEEDS:
            rc = PROJECT_ROOT / 'runs' / run / f'seed{seed}' / 'results.csv'
            if not rc.exists():
                continue
            ep, mp = read_results_csv(rc)
            ax.plot(ep, mp, label=f'{run} seed{seed}', alpha=0.8)
    ax.set_xlabel('эпоха')
    ax.set_ylabel('mAP@0.50:0.95 (val)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / 'training_curves_all_models.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_alphas(run: str, seed: int) -> Path:
    path = PROJECT_ROOT / 'runs' / run / f'seed{seed}' / 'drb_alphas.csv'
    by_layer: dict[int, dict[str, list]] = defaultdict(
        lambda: {'epoch': [], 'a1': [], 'a2': []})
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d = by_layer[int(r['layer_idx'])]
            d['epoch'].append(int(r['epoch']))
            d['a1'].append(float(r['alpha_lqa']))
            d['a2'].append(float(r['alpha_fga']))

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, (idx, d) in enumerate(sorted(by_layer.items())):
        level = LEVEL_BY_ORDER[n] if len(by_layer) == 3 else f'L{idx}'
        ax.plot(d['epoch'], d['a1'], label=f'{level} α₁ (LQA)', linestyle='-')
        ax.plot(d['epoch'], d['a2'], label=f'{level} α₂ (FGA)', linestyle='--')
    ax.axhline(0.5, color='gray', lw=0.8, alpha=0.5)
    ax.set_xlabel('эпоха')
    ax.set_ylabel('α = σ(w)')
    ax.set_title(f'{run} seed{seed}: динамика весов ветвей DRB')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f'learned_weights_dynamics_{run}_seed{seed}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--curves', nargs='+', default=None,
                    help='конфигурации для кривых обучения')
    ap.add_argument('--alphas', default=None,
                    help='конфигурация для динамики α')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.curves:
        print(f'[OK ] {plot_curves(args.curves)}')
    if args.alphas:
        print(f'[OK ] {plot_alphas(args.alphas, args.seed)}')
    if not args.curves and not args.alphas:
        print('укажите --curves и/или --alphas')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
