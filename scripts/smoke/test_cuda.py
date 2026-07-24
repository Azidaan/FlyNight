"""Smoke-тест GPU-окружения (Этап 1, блок 1).

Выводит: версию PyTorch (с суффиксом сборки), доступность CUDA, имя GPU,
compute capability, версии CUDA и cuDNN, расположение интерпретатора и кэшей.

Проверяет:
- сборка PyTorch содержит '+cu' (CUDA), а не '+cpu';
- torch.cuda.is_available() == True;
- имя GPU содержит '5070' (защита от случайной работы на другой карте);
- интерпретатор лежит внутри папки проекта (локальное окружение);
- кэши pip/torch/ultralytics указывают внутрь папки проекта.

При нарушении любого условия завершается с ненулевым кодом возврата.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    failures: list[str] = []

    print(f'python           : {sys.version.split()[0]}')
    print(f'executable       : {sys.executable}')
    if not str(Path(sys.executable).resolve()).lower().startswith(str(PROJECT_ROOT).lower()):
        failures.append(
            f'Интерпретатор вне папки проекта: {sys.executable} '
            f'(ожидается внутри {PROJECT_ROOT})'
        )

    for var in ('PIP_CACHE_DIR', 'TORCH_HOME', 'YOLO_CONFIG_DIR'):
        val = os.environ.get(var, '')
        print(f'{var:<17}: {val or "<не задана>"}')
        if not val or not val.lower().startswith(str(PROJECT_ROOT).lower()):
            failures.append(
                f'Переменная {var} не указывает внутрь проекта '
                f'(проверьте sitecustomize.py в .venv/Lib/site-packages)'
            )

    import torch

    print(f'torch            : {torch.__version__}')
    if '+cpu' in torch.__version__ or '+cu' not in torch.__version__:
        failures.append(
            f'PyTorch собран без CUDA: {torch.__version__}. '
            f'Переустановите с --index-url https://download.pytorch.org/whl/cu130'
        )

    available = torch.cuda.is_available()
    print(f'cuda.is_available: {available}')
    if not available:
        failures.append('torch.cuda.is_available() == False — GPU недоступен')
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f'GPU              : {name}')
        print(f'capability       : sm_{cap[0]}{cap[1]}')
        print(f'CUDA (torch)     : {torch.version.cuda}')
        print(f'cuDNN            : {torch.backends.cudnn.version()}')
        if '5070' not in name:
            failures.append(f'Ожидалась RTX 5070, обнаружено: {name}')

    if failures:
        print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
        for f in failures:
            print(f'  [FAIL] {f}')
        return 1

    print('\n--- OK: окружение корректно ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
