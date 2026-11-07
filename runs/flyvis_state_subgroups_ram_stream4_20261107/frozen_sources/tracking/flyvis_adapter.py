"""Изолированная загрузка официального FlyVis на Windows.

FlyVis 1.2.0 используется как внешний зрительный модуль. Его код и зависимости
размещаются в ``tools/flyvis-runtime`` (каталог исключён из Git), а официальные
контрольные точки — в ``runs/flyvis``. Основное окружение проекта при этом не
переустанавливается и сохраняет рабочий Torch/CUDA стек YOLO26.

Datamate 1.0.0 открывает новый HDF5-файл, пытается прочитать ещё не созданный
набор ``data`` и в обработчике ошибки удаляет всё ещё открытый файл. Linux
разрешает удалить открытый файл, Windows возвращает WinError 32. Функция ниже
закрывает дескрипторы до публикации и заменяет файл только после полной записи.
Формат массива ``data`` и вычисления FlyVis не изменяются.
"""

from __future__ import annotations

import os
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "tools" / "flyvis-runtime"
DEFAULT_DATA_DIR = PROJECT_ROOT / "runs" / "flyvis"


def _patch_datamate_hdf5_writer() -> None:
    """Публиковать HDF5 целиком после успешной записи и закрытия дескрипторов."""
    if os.name != "nt":
        return

    import h5py
    import numpy as np
    import datamate.directory as directory_module
    import datamate.io as io_module

    if getattr(io_module._write_h5, "_flyvis_windows_safe", False):
        return

    def safe_write_h5(path: Path, value: Any) -> None:
        array = np.asarray(value)
        requested = Path(path)
        # mode='w' follows file symlinks. Replacing the link itself would change
        # that contract, so publish to its resolved target instead.
        path = requested.resolve()
        if requested.is_symlink() and path.is_dir():
            raise IsADirectoryError(f"HDF5 path is a symlink to a directory: {requested}")
        if path.is_file() and path.stat().st_nlink > 1:
            # Replacing one name would silently detach it from the other hard
            # links; updating their shared file in place cannot be atomic.
            raise OSError(f"Atomic HDF5 write cannot replace a target with multiple hard links: {path}")
        requested.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        removed_empty_directory = False
        try:
            # Unsupported dtypes, flush errors and SWMR errors only affect this
            # private file. All HDF5 handles close before the atomic replace.
            with h5py.File(temporary, mode="w", libver="latest") as handle:
                handle.create_dataset("data", data=array)
                handle.flush()
                handle.swmr_mode = True
                assert handle.swmr_mode
            if path.is_dir():
                # Preserve datamate's empty-directory fallback. rmdir refuses
                # nonempty directories; no existing directory is renamed.
                path.rmdir()
                removed_empty_directory = True
            try:
                os.replace(temporary, path)
            except BaseException:
                if removed_empty_directory and not path.exists():
                    path.mkdir()
                raise
        finally:
            temporary.unlink(missing_ok=True)

    safe_write_h5._flyvis_windows_safe = True  # type: ignore[attr-defined]
    io_module._write_h5 = safe_write_h5
    # directory.py импортирует функцию по имени, поэтому меняем обе ссылки.
    directory_module._write_h5 = safe_write_h5


def activate_flyvis(
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    device: Any = None,
):
    """Подключить изолированный FlyVis и вернуть импортированный модуль.

    Функцию следует вызывать в отдельном процессе: FlyVis при импорте задаёт
    глобальное устройство Torch. Диагностический стенд именно так и запускается,
    поэтому состояние YOLO26 в основном процессе не затрагивается. ``device``
    явно задаёт устройство нового вычислительного процесса: обновляются также
    копии устройства в инициализаторе и декодере FlyVis. ``None`` сохраняет
    штатный автоматический выбор при первом импорте и уже выбранное устройство
    при повторном вызове. Уже созданные модели эта функция не переносит;
    переключение устройства требует отдельного процесса без живых моделей.

    Повторная активация другого runtime/data_dir в том же процессе запрещена:
    импортированные классы сохраняют прежние корневые пути и устройство.
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

    # Changing an environment variable does not reload an imported package or
    # defaults such as NetworkView(root_dir=flyvis.results_dir).
    loaded = sys.modules.get("flyvis")
    if loaded is not None:
        loaded_runtime = Path(loaded.__file__).resolve().parent.parent
        loaded_data = Path(loaded.root_dir).resolve()
        if loaded_runtime != runtime_dir or loaded_data != data_dir:
            raise RuntimeError(
                "FlyVis уже импортирован из другого runtime/data_dir; "
                "запустите отдельный процесс вместо незаметной смены весов. "
                f"Загружено: {loaded_runtime}, {loaded_data}; "
                f"запрошено: {runtime_dir}, {data_dir}"
            )

    requested_device = None
    if device is not None:
        import torch

        requested_device = torch.device(device)
        if requested_device.type not in {"cpu", "cuda"}:
            raise ValueError("FlyVis поддерживает явный выбор cpu или cuda")
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Запрошена CUDA, но она недоступна")
            if requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
                raise ValueError(f"Недоступное устройство CUDA: {requested_device}")

    runtime_text = str(runtime_dir)
    if loaded is None:
        # An existing entry later in sys.path must not let a different installed
        # FlyVis win on the first import.
        if runtime_text in sys.path:
            sys.path.remove(runtime_text)
        sys.path.insert(0, runtime_text)
    elif runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    os.environ["FLYVIS_ROOT_DIR"] = str(data_dir)

    _patch_datamate_hdf5_writer()
    import flyvis

    if requested_device is not None:
        # Both modules use `from flyvis import device`, so changing flyvis.device
        # alone leaves the RNG and decoder masks on the old device.
        for module in (
            flyvis,
            importlib.import_module("flyvis.network.initialization"),
            importlib.import_module("flyvis.task.decoder"),
        ):
            module.device = requested_device
        torch.set_default_device(requested_device)

    return flyvis

