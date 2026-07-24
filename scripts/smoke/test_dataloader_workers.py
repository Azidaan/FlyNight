"""Smoke-тест DataLoader на Windows (Этап 1, блок 4).

На Windows DataLoader с num_workers > 0 использует spawn и может зависать
в debug-режиме. Проверяем оба сценария и функции-помощники, которые затем
переедут в datasets/bdd100k_loader.py:

- detect_debug_mode() — определение debug через sys.gettrace() / env var;
- get_dataloader_kwargs(is_debug) — безопасные параметры DataLoader.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset


def detect_debug_mode() -> bool:
    """Debug-режим: активный трейсер (debugpy/pdb) или явная переменная окружения."""
    return sys.gettrace() is not None or os.environ.get('FORCE_DEBUG_MODE') == '1'


def get_dataloader_kwargs(is_debug: bool) -> dict:
    """Безопасные параметры DataLoader для Windows."""
    if is_debug:
        return {'num_workers': 0, 'persistent_workers': False}
    return {'num_workers': 4, 'persistent_workers': True}


class _ToyDataset(Dataset):
    def __len__(self) -> int:
        return 64

    def __getitem__(self, idx: int):
        return torch.full((3, 32, 32), float(idx)), idx


def run_loader(num_workers: int) -> float:
    kwargs = {'num_workers': num_workers}
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    loader = DataLoader(_ToyDataset(), batch_size=8, **kwargs)
    t0 = time.perf_counter()
    total = 0
    for batch, idx in loader:
        total += batch.shape[0]
    dt = time.perf_counter() - t0
    assert total == 64, f'прочитано {total} примеров вместо 64'
    return dt


def main() -> int:
    is_debug = detect_debug_mode()
    print(f'[INFO] debug-режим: {is_debug}')
    print(f'[INFO] get_dataloader_kwargs -> {get_dataloader_kwargs(is_debug)}')

    dt0 = run_loader(num_workers=0)
    print(f'[OK ] num_workers=0: 64 примера за {dt0:.2f} с')

    dt4 = run_loader(num_workers=4)
    print(f'[OK ] num_workers=4: 64 примера за {dt4:.2f} с (включая старт воркеров)')

    print('\n--- OK: DataLoader работает в обоих режимах ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
