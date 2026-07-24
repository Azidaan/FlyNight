"""Unit-тесты блока DRB (Этап 4, подэтап 4.5).

Проверки по критерию готовности плана, для трёх уровней пирамиды P3/P4/P5:
- формы: x_out совпадает со входом, Q имеет форму (B,1,H,W);
- значения Q в [0,1];
- backward без NaN, градиенты доходят до w_LQA, w_FGA, маски FGA и Conv LQA;
- α₁, α₂ на старте равны 0.5 (инициализация нулями);
- один шаг оптимизатора сдвигает w_LQA/w_FGA (веса учатся);
- FGA с маской-единицами ≈ тождественное преобразование (нейтральный старт);
- работает и на CPU, и на CUDA; под autocast (AMP) NaN не появляются.

Запуск: python tests/test_drb.py  (без pytest — стиль smoke-скриптов проекта).
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.drb import DRB, FGA  # noqa: E402

# (каналы, H, W) уровней пирамиды при входе 640×640 (страйды 8/16/32);
# каналы условные — DRB не зависит от конкретного масштаба YOLO.
PYRAMID = {'P3': (64, 80, 80), 'P4': (128, 40, 40), 'P5': (256, 20, 20)}
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def test_level(level: str, c: int, h: int, w: int, device: str) -> None:
    torch.manual_seed(0)
    drb = DRB(c, h, w).to(device)
    x = torch.randn(2, c, h, w, device=device, requires_grad=True)

    x_out, q = drb(x)
    check(f'{level}/{device} форма x_out', x_out.shape == x.shape, str(tuple(x_out.shape)))
    check(f'{level}/{device} форма Q', q.shape == (2, 1, h, w), str(tuple(q.shape)))
    check(f'{level}/{device} Q в [0,1]',
          bool((q >= 0).all() and (q <= 1).all()),
          f'min={q.min():.3f}, max={q.max():.3f}')

    a1, a2 = drb.alphas
    check(f'{level}/{device} старт α₁=α₂=0.5', abs(a1 - .5) < 1e-6 and abs(a2 - .5) < 1e-6,
          f'α₁={a1:.3f}, α₂={a2:.3f}')

    loss = x_out.square().mean() + q.mean()
    loss.backward()
    grads = {
        'вход x': x.grad,
        'w_LQA': drb.w_lqa.grad,
        'w_FGA': drb.w_fga.grad,
        'маска FGA': drb.fga.mask.grad,
        'Conv LQA': drb.lqa.conv_merge.weight.grad,
    }
    for gname, g in grads.items():
        check(f'{level}/{device} градиент {gname}',
              g is not None and torch.isfinite(g).all().item())

    # один шаг оптимизатора — w_LQA/w_FGA должны сдвинуться
    opt = torch.optim.SGD(drb.parameters(), lr=0.5)
    opt.step()
    a1_new, a2_new = drb.alphas
    check(f'{level}/{device} веса ветвей учатся',
          abs(a1_new - .5) > 1e-6 or abs(a2_new - .5) > 1e-6,
          f'α₁ 0.5→{a1_new:.4f}, α₂ 0.5→{a2_new:.4f}')


def test_fga_identity(device: str) -> None:
    c, h, w = 32, 40, 40
    fga = FGA(c, h, w).to(device)
    x = torch.randn(2, c, h, w, device=device)
    err = (fga(x) - x).abs().max().item()
    check(f'FGA/{device} нейтральный старт (маска единиц ≈ identity)',
          err < 1e-4, f'max_err={err:.2e}')


def test_amp_cuda() -> None:
    c, h, w = 64, 80, 80
    drb = DRB(c, h, w).cuda()
    x = torch.randn(2, c, h, w, device='cuda')
    with torch.autocast('cuda'):
        x_out, q = drb(x)
        loss = x_out.float().square().mean() + q.float().mean()
    loss.backward()
    ok = torch.isfinite(x_out).all().item() and torch.isfinite(drb.fga.mask.grad).all().item()
    check('AMP/cuda forward+backward без NaN', bool(ok))


def main() -> int:
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for level, (c, h, w) in PYRAMID.items():
            test_level(level, c, h, w, device)
        test_fga_identity(device)
    if 'cuda' in devices:
        test_amp_cuda()

    n_params = sum(p.numel() for p in DRB(256, 80, 80).parameters())
    print(f'\n[INFO] параметров DRB на P3 (C=256, 80×80): {n_params:,} '
          f'(маска FGA доминирует: 256×80×41)')

    if FAILURES:
        print(f'\n--- ПРОВАЛЕНО {len(FAILURES)} ПРОВЕРОК ---')
        return 1
    print('\n--- OK: все проверки DRB пройдены ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
