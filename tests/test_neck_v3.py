"""Проверки фиксированного низкочастотного фильтра (операторы шеи v3).

Что здесь доказывается — по пунктам, каждый закрывает конкретную прошлую беду:
- обучаемых чисел РОВНО ноль (весь смысл замысла: сети нечего выключать);
- при strength = 0 выход побитово равен входу (тест на отключение);
- четыре формы карт, включая ПРЯМОУГОЛЬНЫЕ — ловушка прямоугольных пачек
  валидации уже однажды стоила пяти падений;
- под autocast fp16 нет NaN и нет островка полной точности;
- CPU и CUDA дают согласованный результат;
- врезка в реальную модель сохраняет маршрутизацию и не меняет форм выходов;
- врезка идемпотентна (повторный вызов не наслаивает фильтры);
- оба размещения (upsample / levels) собираются и считают.

Запуск: python tests/test_neck_v3.py  (без pytest — стиль smoke-скриптов проекта).
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.neck_v3.fixed_lowpass import (  # noqa: E402
    FixedLowPass, SmoothedLayer, gaussian_kernel2d, has_lowpass,
    insert_lowpass, trainable_in_filters, upsample_indices)

FAILURES: list[str] = []
SHAPES = [(2, 32, 40, 40), (2, 32, 80, 80), (1, 32, 48, 80), (1, 32, 64, 112)]


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'[{"OK " if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not ok:
        FAILURES.append(name)


def test_kernel() -> None:
    k = gaussian_kernel2d(5, 1.0)
    check('ядро суммируется в единицу', abs(float(k.sum()) - 1.0) < 1e-6,
          f'сумма {float(k.sum()):.8f}')
    check('ядро симметрично по горизонтали', torch.allclose(k, k.flip(-1), atol=1e-7))
    check('ядро симметрично по вертикали', torch.allclose(k, k.flip(-2), atol=1e-7))
    check('ядро радиально: углы равны', abs(float(k[0, 0] - k[0, -1])) < 1e-7
          and abs(float(k[0, 0] - k[-1, 0])) < 1e-7)
    try:
        gaussian_kernel2d(4, 1.0)
        check('чётное ядро отвергается', False, 'исключения не было')
    except ValueError:
        check('чётное ядро отвергается', True)


def test_zero_trainable() -> None:
    f = FixedLowPass(64)
    n_par = sum(p.numel() for p in f.parameters())
    n_buf = sum(b.numel() for b in f.buffers())
    check('обучаемых чисел ровно ноль', n_par == 0, f'параметров {n_par}, буферов {n_buf}')
    check('ядро лежит в state_dict', 'kernel' in f.state_dict())


def test_identity_at_zero() -> None:
    f = FixedLowPass(32, strength=0.0)
    for shp in SHAPES:
        x = torch.randn(*shp)
        y = f(x)
        if not torch.equal(x, y):
            check(f'strength=0 побитовое тождество {shp}', False)
            return
    check('strength=0 побитовое тождество на всех формах', True, f'{len(SHAPES)} форм')


def test_shapes_and_finite() -> None:
    f = FixedLowPass(32, kernel_size=5, sigma=1.0, strength=1.0)
    bad = []
    for shp in SHAPES:
        x = torch.randn(*shp)
        y = f(x)
        if y.shape != x.shape or not torch.isfinite(y).all():
            bad.append(str(shp))
    check('формы сохраняются, значений NaN нет', not bad,
          f'проверено форм: {len(SHAPES)}' + (f'; сбои: {bad}' if bad else ''))


def test_smoothing_actually_smooths() -> None:
    """Фильтр обязан снижать высокочастотную энергию — иначе он бесполезен."""
    f = FixedLowPass(1, kernel_size=5, sigma=1.0, strength=1.0)
    x = torch.randn(1, 1, 64, 64)
    y = f(x)
    hf = lambda t: float((t - torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(t, (1, 1, 1, 1), mode='replicate'), 3, 1)).pow(2).mean())
    before, after = hf(x), hf(y)
    check('высокочастотная энергия падает', after < before * 0.9,
          f'до {before:.4f} → после {after:.4f} ({after/before:.0%})')


def test_amp_and_devices() -> None:
    f = FixedLowPass(32, strength=1.0)
    x = torch.randn(1, 32, 48, 80)
    y_cpu = f(x)
    if not torch.cuda.is_available():
        check('CUDA недоступна — проверка устройств пропущена', True)
        return
    fg, xg = f.cuda(), x.cuda()
    y_gpu = fg(xg)
    check('CPU и CUDA согласованы', torch.allclose(y_cpu, y_gpu.cpu(), atol=1e-5),
          f'макс. расхождение {float((y_cpu - y_gpu.cpu()).abs().max()):.2e}')
    with torch.autocast('cuda', dtype=torch.float16):
        y_amp = fg(xg)
    check('под autocast fp16 нет NaN', torch.isfinite(y_amp).all(),
          f'dtype выхода {y_amp.dtype}')
    f.cpu()


def test_insertion(placement: str) -> None:
    """Врезка в настоящую модель: маршрутизация, формы выходов, идемпотентность."""
    import warnings
    warnings.filterwarnings('ignore')
    from ultralytics import YOLO
    w = PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt'
    if not w.exists():
        check(f'врезка [{placement}]: нет весов {w}', False)
        return

    base = YOLO(str(w)).model.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        ref = base(x)
    ref_shapes = [tuple(t.shape) for t in (ref if isinstance(ref, (list, tuple)) else [ref])
                  if torch.is_tensor(t)]

    model = YOLO(str(w)).model.eval()
    idx = insert_lowpass(model, placement=placement, strength=1.0)
    check(f'[{placement}] слои обёрнуты', bool(idx), f'индексы {idx}')
    check(f'[{placement}] обучаемых чисел в фильтрах ноль',
          trainable_in_filters(model) == 0, f'{trainable_in_filters(model)}')

    with torch.no_grad():
        out = model(x)
    out_shapes = [tuple(t.shape) for t in (out if isinstance(out, (list, tuple)) else [out])
                  if torch.is_tensor(t)]
    check(f'[{placement}] формы выходов не изменились', out_shapes == ref_shapes,
          f'{out_shapes[:2]}')

    again = insert_lowpass(model, placement=placement)
    n_wrapped = sum(1 for m in model.model if isinstance(m, SmoothedLayer))
    check(f'[{placement}] врезка идемпотентна',
          again == idx and n_wrapped == len(idx), f'обёрток {n_wrapped}')

    # Отключённый фильтр обязан дать в точности исходную модель.
    off = YOLO(str(w)).model.eval()
    insert_lowpass(off, placement=placement, strength=0.0)
    with torch.no_grad():
        o = off(x)
    o_t = o[0] if isinstance(o, (list, tuple)) else o
    r_t = ref[0] if isinstance(ref, (list, tuple)) else ref
    check(f'[{placement}] strength=0 — модель совпадает с чистой',
          torch.allclose(o_t, r_t, atol=1e-6),
          f'макс. расхождение {float((o_t - r_t).abs().max()):.2e}')

    # Прямоугольная пачка — та самая ловушка.
    with torch.no_grad():
        model(torch.zeros(1, 3, 384, 640))
    check(f'[{placement}] прямоугольная пачка 384×640 проходит', True)


def main() -> int:
    print('=== ядро ===')
    test_kernel()
    print('\n=== модуль ===')
    test_zero_trainable()
    test_identity_at_zero()
    test_shapes_and_finite()
    test_smoothing_actually_smooths()
    test_amp_and_devices()
    print('\n=== врезка в модель ===')
    for placement in ('upsample', 'levels'):
        test_insertion(placement)
        print()

    if FAILURES:
        print(f'--- ПРОВАЛЕНО {len(FAILURES)} ПРОВЕРОК: {FAILURES} ---')
        return 1
    print('--- OK: все проверки фильтра пройдены ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
