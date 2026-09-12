from pathlib import Path

import numpy as np
import pytest

from tracking.flyvis_adapter import DEFAULT_RUNTIME_DIR, _patch_datamate_hdf5_writer


@pytest.mark.skipif(
    not (DEFAULT_RUNTIME_DIR / "flyvis").is_dir(),
    reason="изолированный FlyVis runtime не установлен",
)
def test_windows_hdf5_writer_closes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(DEFAULT_RUNTIME_DIR))
    _patch_datamate_hdf5_writer()

    import datamate.io as io_module
    import h5py

    path = tmp_path / "array.h5"
    expected = np.arange(6, dtype=np.float32)
    io_module._write_h5(path, expected)
    with h5py.File(path, "r", libver="latest", swmr=True) as handle:
        np.testing.assert_array_equal(handle["data"][:], expected)

    # Повторная запись проверяет, что предыдущий файловый дескриптор закрыт.
    io_module._write_h5(path, expected + 1)
    with h5py.File(path, "r", libver="latest", swmr=True) as handle:
        np.testing.assert_array_equal(handle["data"][:], expected + 1)

