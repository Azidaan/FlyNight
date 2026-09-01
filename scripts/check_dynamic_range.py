"""Съедает ли батчнорм разницу дня и ночи по размаху сигнала?

Вопрос. Замер спектров 21.08 показал: ночь в BDD100K — это потеря КОНТРАСТА
(размаха), а не рост шума. Контраст лечится не эквалайзером (частотная ось,
это была FGA), а сжатием-растяжением динамического диапазона (уровневая ось,
не пробовали ни разу). Но внутри YOLO в каждом блоке стоит батчнорм, который
выравнивает статистики. Возможно, он уже съел всю разницу, и чинить нечего.

Что меряем. Размах сигнала в ДВУХ точках на одних и тех же кадрах:
    вход   — сама картинка, до сети;
    шея    — выходы уровней, куда смотрит голова (после всех нормировок).
Если отношение день/ночь на входе большое, а в шее близко к единице — батчнорм
разницу съел, уровневая ось закрыта. Если разница доживает до шеи — есть что
чинить.

Обучение не нужно, видеокарта нужна ненадолго. Модель — контрольная, без блока.

Запуск: python scripts/check_dynamic_range.py
        python scripts/check_dynamic_range.py --frames 120
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_q_signal import pick_frames  # noqa: E402

WEIGHTS = PROJECT_ROOT / 'weights' / 'baseline_seed42' / 'best.pt'


def spread_stats(a: np.ndarray) -> dict[str, float]:
    """Меры размаха. p95−p05 устойчивее к выбросам, чем max−min."""
    return {
        'скво': float(a.std()),
        'p95-p05': float(np.percentile(a, 95) - np.percentile(a, 5)),
        'сред': float(a.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=int, default=80, help='кадров на условие')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--layers', default='head', choices=['head', 'all'],
                    help='head — только входы головы; all — каждый слой, '
                         'чтобы найти, ГДЕ именно стирается разница')
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    import cv2
    import torch
    from ultralytics import YOLO

    if not WEIGHTS.exists():
        raise SystemExit(f'нет весов контроля: {WEIGHTS}')
    model = YOLO(str(WEIGHTS)).model.eval().cuda()
    head_idx = (list(range(len(model.model) - 1)) if args.layers == 'all'
                else list(model.model[-1].f))

    grabbed: dict[int, torch.Tensor] = {}
    handles = []
    for idx in head_idx:
        def _hook(m, i, o, idx=idx):
            grabbed[idx] = o.detach()
        handles.append(model.model[idx].register_forward_hook(_hook))

    acc: dict[str, dict[str, list]] = {}
    for cond in ('normal', 'night'):
        frames = pick_frames(cond, args.frames)
        print(f'[INFO] {cond}: кадров {len(frames)}', flush=True)
        acc[cond] = {'вход': [], **{f'слой {i:02d}': [] for i in head_idx}}
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.resize(img, (args.imgsz, args.imgsz))
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            acc[cond]['вход'].append(spread_stats(g))
            t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
            t = (t.float() / 255.0).unsqueeze(0).cuda()
            with torch.no_grad():
                model(t)
            for idx in head_idx:
                acc[cond][f'слой {idx:02d}'].append(
                    spread_stats(grabbed[idx].float().cpu().numpy()))
    for h in handles:
        h.remove()

    print('\n%-12s %10s %10s %10s %10s %10s' % (
        'точка', 'день p95-p05', 'ночь', 'отн. н/д', 'день скво', 'ночь скво'))
    ratios = {}
    for point in acc['normal']:
        d = np.mean([s['p95-p05'] for s in acc['normal'][point]])
        n = np.mean([s['p95-p05'] for s in acc['night'][point]])
        ds = np.mean([s['скво'] for s in acc['normal'][point]])
        ns = np.mean([s['скво'] for s in acc['night'][point]])
        ratios[point] = n / d if d else float('nan')
        print('%-12s %10.4f %10.4f %10.3f %10.4f %10.4f'
              % (point, d, n, ratios[point], ds, ns))

    print('\n=== приговор ===')
    r_in = ratios.get('вход', float('nan'))
    # Приговор считается ТОЛЬКО по входам головы. В режиме all в ratios лежат
    # и ранние слои, где разница ещё жива; усреднение по ним дало бы обратный
    # вывод — это уже один раз ввело в заблуждение.
    head_f = list(model.model[-1].f)
    neck = [ratios[k] for k in (f'слой {i:02d}' for i in head_f) if k in ratios]
    r_neck = float(np.mean(neck)) if neck else float('nan')
    print(f'  размах ночь/день НА ВХОДЕ:  {r_in:.3f}')
    print(f'  размах ночь/день В ШЕЕ:     {r_neck:.3f}  (среднее по уровням)')
    lost = (1 - abs(1 - r_neck) / abs(1 - r_in)) * 100 if abs(1 - r_in) > 1e-6 else float('nan')
    print(f'  съедено нормировками:       {lost:.0f}% разницы')
    if abs(1 - r_neck) < 0.05:
        print('\n  БАТЧНОРМ СЪЕЛ РАЗНИЦУ. Уровневая ось закрыта: в шее день и ночь')
        print('  по размаху неразличимы, сжимать-растягивать нечего.')
    else:
        print('\n  РАЗНИЦА ДОЖИВАЕТ ДО ШЕИ. Уровневая ось жива: есть что чинить,')
        print('  и это НЕ то, чем занималась FGA (та работала по частотам).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
