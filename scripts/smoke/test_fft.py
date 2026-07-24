"""Smoke-тест torch.fft на GPU (Этап 1, блок 4).

FFT нужен для частотной ветви FGA блока DRB. На RTX 5070 (sm_120, CUDA 13)
проверяем, что rfft2/irfft2 работают, численно корректны и быстры.

Проверки:
1. Roundtrip: irfft2(rfft2(x)) == x с точностью fp32.
2. Умножение спектра на маску (как в FGA) не ломает форму и градиенты.
3. Замер времени на типичном для DRB размере тензора.
"""
import sys
import time

import torch

TOL = 1e-4


def main() -> int:
    if not torch.cuda.is_available():
        print('[FAIL] CUDA недоступна — запустите сначала test_cuda.py')
        return 1

    device = torch.device('cuda')
    failures: list[str] = []

    # 1. Roundtrip на размерах, характерных для P3/P4/P5 при входе 640x640.
    for shape in [(2, 64, 80, 80), (2, 128, 40, 40), (2, 256, 20, 20)]:
        x = torch.randn(*shape, device=device)
        spec = torch.fft.rfft2(x)
        y = torch.fft.irfft2(spec, s=x.shape[-2:])
        err = (x - y).abs().max().item()
        status = 'OK ' if err < TOL else 'FAIL'
        print(f'[{status}] roundtrip {shape}: max_err={err:.2e}')
        if err >= TOL:
            failures.append(f'roundtrip {shape}: ошибка {err:.2e} >= {TOL}')

    # 2. FGA-подобная операция: обучаемая маска, умножение спектра, backward.
    x = torch.randn(2, 64, 80, 80, device=device, requires_grad=True)
    spec = torch.fft.rfft2(x)
    mask = torch.ones(64, 80, 41, device=device, requires_grad=True)  # 41 = 80//2+1
    y = torch.fft.irfft2(spec * mask, s=x.shape[-2:])
    y.sum().backward()
    grad_ok = x.grad is not None and mask.grad is not None \
        and torch.isfinite(x.grad).all() and torch.isfinite(mask.grad).all()
    print(f'[{"OK " if grad_ok else "FAIL"}] FGA-операция: форма {tuple(y.shape)}, градиенты конечны')
    if not grad_ok:
        failures.append('градиенты через rfft2/irfft2 не проходят или содержат NaN/Inf')

    # 3. Скорость: 100 итераций rfft2+маска+irfft2 на (16, 64, 80, 80).
    x = torch.randn(16, 64, 80, 80, device=device)
    m = torch.ones(64, 80, 41, device=device)
    for _ in range(5):
        _ = torch.fft.irfft2(torch.fft.rfft2(x) * m, s=x.shape[-2:])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = torch.fft.irfft2(torch.fft.rfft2(x) * m, s=x.shape[-2:])
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 100 * 1000
    print(f'[INFO] FGA-цикл (batch 16, 64x80x80): {dt:.3f} мс/итерация')

    if failures:
        print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
        for f in failures:
            print(f'  [FAIL] {f}')
        return 1

    print('\n--- OK: torch.fft работает на GPU ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
