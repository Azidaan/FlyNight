"""Unit-тесты блока DRB (Этап 4, подэтап 4.5).

Проверки по критерию готовности плана, для трёх уровней пирамиды P3/P4/P5:
- формы: x_out совпадает со входом, Q имеет форму (B,1,H,W);
- значения Q в [0,1];
- backward без NaN, градиенты доходят до w_LQA, w_FGA, маски FGA и Conv LQA;
- α₁, α₂ на старте равны 0.5 (инициализация нулями);
- один шаг оптимизатора сдвигает w_LQA/w_FGA (веса учатся);
- FGA с маской-единицами ≈ тождественное преобразование (нейтральный старт);
- работает и на CPU, и на CUDA; под autocast (AMP) NaN не появляются.

Запуск: python tests/test_drb.py или pytest для этого файла.
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


def check_level(level: str, c: int, h: int, w: int, device: str) -> None:
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
        'отклонение маски FGA': drb.fga.delta.grad,
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


def check_fga_identity(device: str) -> None:
    c, h, w = 32, 40, 40
    fga = FGA(c, h, w).to(device)
    x = torch.randn(2, c, h, w, device=device)
    err = (fga(x) - x).abs().max().item()
    check(f'FGA/{device} нейтральный старт (маска единиц ≈ identity)',
          err < 1e-4, f'max_err={err:.2e}')


def check_amp_cuda() -> None:
    c, h, w = 64, 80, 80
    drb = DRB(c, h, w).cuda()
    x = torch.randn(2, c, h, w, device='cuda')
    with torch.autocast('cuda'):
        x_out, q = drb(x)
        loss = x_out.float().square().mean() + q.float().mean()
    loss.backward()
    ok = torch.isfinite(x_out).all().item() and torch.isfinite(drb.fga.delta.grad).all().item()
    check('AMP/cuda forward+backward без NaN', bool(ok))


def _decay_only(module) -> None:
    """Прогон оптимизатора без сигнала обучения: работает только штраф."""
    for _ in range(200):
        for p in module.parameters():
            p.grad = torch.zeros_like(p)
        yield


def check_mask_neutral_under_decay() -> None:
    """Штраф за величину больше не гасит маску частот.

    Регрессия: раньше хранилась сама маска (старт — единицы), и штраф тянул
    её к нулю, то есть к «вырезать все частоты». За 12 эпох она уезжала с
    1.0 до 0.71, причём одинаково при вкладе ветви 50%, 12% и 2% — то есть
    это было затухание, а не обучение, и блок отставал от контроля на
    0.025 mAP50. Теперь хранится отклонение от единицы, сжимать нечего.
    """
    drb = DRB(16, 8, 8)
    opt = torch.optim.SGD(drb.parameters(), lr=0.1, weight_decay=0.5)
    for _ in _decay_only(drb):
        opt.step()
    dev = (drb.fga.mask3d - 1.0).abs().max().item()
    check('маска остаётся нейтральной под штрафом', dev < 1e-6, f'отклонение {dev:.2e}')


def check_block_excluded_from_decay() -> None:
    """Маска и веса ветвей выводятся из-под штрафа в оптимизаторе.

    Отклонение маски спасает её саму, но логит α спасать нечем: его нейтраль
    там, куда его поставили, а не в нуле. Поэтому параметры блока переносятся
    в группу без штрафа — иначе стартовый вклад ветвей уползал к 50%
    независимо от того, каким его задали.
    """
    from models.yolo_neck.neck_drb import separate_block_params

    drb = DRB(16, 8, 8)
    with torch.no_grad():
        drb.w_lqa.fill_(-2.0)
        drb.w_fga.fill_(-2.0)
    opt = torch.optim.SGD(drb.parameters(), lr=0.1, weight_decay=0.5)
    moved = separate_block_params(opt, drb)
    check('из-под штрафа выведены маска и два веса ветвей', moved == 3, str(moved))
    check('свёртка LQA штраф сохранила — она обычный слой', any(
        g['weight_decay'] == 0.5
        and any(p is drb.lqa.conv_merge.weight for p in g['params'])
        for g in opt.param_groups))
    for _ in _decay_only(drb):
        opt.step()
    w = drb.w_lqa.mean().item()          # вес ветви теперь вектор по каналам
    check('стартовый вклад ветви не уползает', abs(w + 2.0) < 1e-6,
          f'w_lqa {w:+.4f} (задавали -2.0)')


def check_adaptive_survives_other_sizes() -> None:
    """Адаптивный подъём работает на картах любого размера, не только обучающего.

    Регрессия 21.08: разметка колец бралась для обучающего размера, а
    проверка в Ultralytics идёт прямоугольными пачками, где карты признаков
    другой формы. Прогон падал на проверке после первой эпохи и делал это
    пять раз подряд, теряя по получасу на попытку.
    """
    fga = FGA(4, 80, 80)
    fga.enable_adaptive()
    x = torch.randn(2, 4, 80, 80)
    check('обучающий размер работает', tuple(fga(x).shape) == (2, 4, 80, 80))
    for h, w in ((48, 80), (80, 48), (32, 32)):
        try:
            out = fga(torch.randn(2, 4, h, w))
            ok = tuple(out.shape) == (2, 4, h, w)
        except RuntimeError as exc:
            ok = False
            print(f'      {h}x{w}: {exc}')
        check(f'размер {h}x{w} работает', ok)


def check_gated_fusion_is_identity() -> None:
    """В режиме поправки блок тождественен при ЛЮБОМ вкладе ветвей.

    Прежняя форма прибавляла выход ветви: при нейтральной маске это копия
    входа, и при вкладе 50% получалось почти 2x. Прямой замер 20.08: точность
    рамок предобученной головы падала вдвое ещё до начала обучения. Поправка
    (разность с входом) при нейтральной маске равна нулю независимо от вклада,
    поэтому подбирать стартовый вклад больше не требуется.
    """
    drb = DRB(8, 16, 16)
    x = torch.randn(2, 8, 16, 16)

    drb.fusion = 'add'
    dev_add = (drb(x)[0] - x).abs().max().item()
    check('прежняя форма при нейтральной маске меняет признаки',
          dev_add > 0.1, f'{dev_add:.3e}')

    drb.fusion = 'gated'
    for w in (0.0, 3.0):                       # вклад 50% и 95%
        with torch.no_grad():
            drb.w_lqa.fill_(w)
            drb.w_fga.fill_(w)
        dev = (drb(x)[0] - x).abs().max().item()
        a = 1 / (1 + pow(2.718281828, -w))
        check(f'поправка тождественна при вкладе {a:.0%}', dev < 1e-5,
              f'{dev:.3e}')


def check_optimizer_compatible_shapes() -> None:
    """Формы параметров должны перевариваться оптимизатором MuSGD.

    Регрессия: Ultralytics при длинных обучениях выбирает MuSGD и шлёт в
    ветвь Muon ВСЕ параметры с ndim >= 2, а та умеет только 2 и 4 измерения
    (4 — свёрточные фильтры, они разворачиваются в матрицу). Трёхмерная
    маска FGA роняла обучение на первом же шаге оптимизатора, причём уже
    после сканирования датасета — то есть через полторы минуты после старта.
    """
    for variant in ('both', 'lqa', 'fga'):
        drb = DRB(256, 40, 40, branches=variant)
        for name, p in drb.named_parameters():
            ok = p.ndim in (1, 2, 4)
            check(f'{variant}: {name} формы {tuple(p.shape)} пригодна для MuSGD', ok,
                  '' if ok else f'ndim={p.ndim} — Muon умеет только 2 и 4')


def main() -> int:
    FAILURES.clear()
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for level, (c, h, w) in PYRAMID.items():
            check_level(level, c, h, w, device)
        check_fga_identity(device)
    if 'cuda' in devices:
        check_amp_cuda()

    check_optimizer_compatible_shapes()
    check_mask_neutral_under_decay()
    check_block_excluded_from_decay()
    check_gated_fusion_is_identity()
    check_adaptive_survives_other_sizes()

    n_params = sum(p.numel() for p in DRB(256, 80, 80).parameters())
    print(f'\n[INFO] параметров DRB на P3 (C=256, 80×80): {n_params:,} '
          f'(маска радиальная: 256 каналов × 24 кольца)')

    if FAILURES:
        print(f'\n--- ПРОВАЛЕНО {len(FAILURES)} ПРОВЕРОК ---')
        return 1
    print('\n--- OK: все проверки DRB пройдены ---')
    return 0


def test_drb_checks() -> None:
    assert main() == 0, FAILURES


if __name__ == '__main__':
    sys.exit(main())
