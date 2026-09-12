"""Изолированная загрузка официального FlyVis на Windows.

FlyVis 1.2.0 используется как внешний зрительный модуль. Его код и зависимости
размещаются в ``tools/flyvis-runtime`` (каталог исключён из Git), а официальные
контрольные точки — в ``runs/flyvis``. Основное окружение проекта при этом не
переустанавливается и сохраняет рабочий Torch/CUDA стек YOLO26.

Datamate 1.0.0 открывает новый HDF5-файл, пытается прочитать ещё не созданный
набор ``data`` и в обработчике ошибки удаляет всё ещё открытый файл. Linux
разрешает удалить открытый файл, Windows возвращает WinError 32. Функция ниже
исправляет только управление файловым дескриптором; данные и вычисления FlyVis
не изменяются.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "tools" / "flyvis-runtime"
DEFAULT_DATA_DIR = PROJECT_ROOT / "runs" / "flyvis"


def _patch_datamate_hdf5_writer() -> None:
    """Заменить небезопасную для Windows запись нового HDF5-файла."""
    if os.name != "nt":
        return

    import h5py
    import numpy as np
    import datamate.directory as directory_module
    import datamate.io as io_module

    if getattr(io_module._write_h5, "_flyvis_windows_safe", False):
        return

    def safe_write_h5(path: Path, value: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        array = np.asarray(value)
        with h5py.File(path, mode="w", libver="latest") as handle:
            handle.create_dataset("data", data=array)
            handle.flush()
            handle.swmr_mode = True
            assert handle.swmr_mode

    safe_write_h5._flyvis_windows_safe = True  # type: ignore[attr-defined]
    io_module._write_h5 = safe_write_h5
    # directory.py импортирует функцию по имени, поэтому меняем обе ссылки.
    directory_module._write_h5 = safe_write_h5


def activate_flyvis(
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
):
    """Подключить изолированный FlyVis и вернуть импортированный модуль.

    Функцию следует вызывать в отдельном процессе: FlyVis при импорте задаёт
    глобальное устройство Torch. Диагностический стенд именно так и запускается,
    поэтому состояние YOLO26 в основном процессе не затрагивается.
    """
    runtime_dir = Path(runtime_dir).resolve()
    data_dir = Path(data_dir).resolve()
    if not (runtime_dir / "flyvis").is_dir():
        raise FileNotFoundError(
            f"FlyVis runtime не найден: {runtime_dir}. "
            "Сначала выполните scripts/setup_flyvis_runtime.py."
        )
    if not (data_dir / "results" / "flow").is_dir():
        raise FileNotFoundError(
            f"Официальные контрольные точки FlyVis не найдены: {data_dir / 'results'}"
        )

    runtime_text = str(runtime_dir)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    os.environ["FLYVIS_ROOT_DIR"] = str(data_dir)

    _patch_datamate_hdf5_writer()
    import flyvis

    return flyvis

