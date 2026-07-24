"""Замеры производительности моделей (Этап 10, подэтап 10.1).

Обёртка над utils_nn.py (measure_fps, count_params_flops_*) с выходом
в структуру, пригодную для performance_table.csv раздела 3.2 диссертации:
FPS (batch 1 и 16), latency, GPU-память, параметры (по компонентам), GFLOPs.

Использование как модуль:
    from utils.benchmark import benchmark_model, count_params_by_component
Как скрипт:
    python utils/benchmark.py --model weights/baseline_seed42/best.pt
    python utils/benchmark.py --model weights/drb_aug_full_seed42/best.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_model(model_path: str, input_size: tuple[int, int] = (640, 640),
                    batch_sizes: tuple[int, ...] = (1, 16),
                    warmup_iter: int = 10, measure_iter: int = 100,
                    device: str = 'cuda') -> dict:
    """FPS / latency / пик GPU-памяти для каждого batch size.

    Возвращает {batch: {'fps': float, 'latency_ms': float, 'gpu_mem_mb': float}}.
    FPS считается по изображениям в секунду (batch * iters / time).
    """
    from ultralytics import YOLO
    det_model = YOLO(model_path).model.to(device).eval()

    out: dict[int, dict] = {}
    h, w = input_size
    for bs in batch_sizes:
        x = torch.randn(bs, 3, h, w, device=device)
        with torch.no_grad():
            for _ in range(warmup_iter):
                det_model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(measure_iter):
                det_model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        out[bs] = {
            'fps': bs * measure_iter / elapsed,
            'latency_ms': elapsed / measure_iter * 1000,
            'gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024**2,
        }
    return out


def count_params_by_component(model_path: str) -> dict[str, int]:
    """Параметры по компонентам: backbone / neck / drb / head / всего.

    Границы для yolo26s: backbone — слои до SPPF включительно (0..10),
    head — последний слой, neck — остальное. DRB считается отдельно
    (и вычитается из neck).
    """
    from ultralytics import YOLO

    from models.yolo_neck import DRBInputLayer, DRBWrappedLayer

    det_model = YOLO(model_path).model
    layers = list(det_model.model)
    counts = {'backbone': 0, 'neck': 0, 'drb': 0, 'head': 0}
    backbone_end = 10  # SPPF в yolo26s; для других масштабов уточнить
    for i, layer in enumerate(layers):
        n_drb = 0
        target = layer
        if isinstance(layer, (DRBWrappedLayer, DRBInputLayer)):
            n_drb = sum(p.numel() for p in layer.drb.parameters())
            target = layer.inner
        n = sum(p.numel() for p in target.parameters())
        counts['drb'] += n_drb
        if i == len(layers) - 1:
            counts['head'] += n
        elif i <= backbone_end:
            counts['backbone'] += n
        else:
            counts['neck'] += n
    counts['total'] = sum(counts.values())
    return counts


def gflops(model_path: str, input_size: tuple[int, int] = (640, 640)) -> float:
    """GFLOPs одного forward через thop (fvcore — как перекрёстная проверка)."""
    from thop import profile
    from ultralytics import YOLO
    det_model = YOLO(model_path).model.eval()
    device = next(det_model.parameters()).device
    x = torch.randn(1, 3, *input_size, device=device)
    flops, _ = profile(det_model, inputs=(x,), verbose=False)
    return flops / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', default=None, help='имя строки в CSV (default: имя папки весов)')
    ap.add_argument('--out', default=str(
        PROJECT_ROOT / 'experiments' / 'results' / 'performance_table.csv'))
    args = ap.parse_args()

    tag = args.tag or Path(args.model).parent.name
    print(f'[INFO] бенчмарк {tag}: {args.model}')
    bench = benchmark_model(args.model)
    params = count_params_by_component(args.model)
    gf = gflops(args.model)

    import csv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['model', 'fps_b1', 'latency_ms_b1', 'gpu_mem_mb_b1',
                        'fps_b16', 'gpu_mem_mb_b16', 'gflops',
                        'params_total', 'params_backbone', 'params_neck',
                        'params_drb', 'params_head'])
        w.writerow([tag,
                    f'{bench[1]["fps"]:.1f}', f'{bench[1]["latency_ms"]:.2f}',
                    f'{bench[1]["gpu_mem_mb"]:.0f}',
                    f'{bench[16]["fps"]:.1f}', f'{bench[16]["gpu_mem_mb"]:.0f}',
                    f'{gf:.2f}', params['total'], params['backbone'],
                    params['neck'], params['drb'], params['head']])
    print(f'[OK ] b1: {bench[1]["fps"]:.1f} FPS, {bench[1]["latency_ms"]:.2f} ms; '
          f'GFLOPs={gf:.2f}; params={params["total"]:,} (DRB {params["drb"]:,})')
    print(f'[OK ] {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
