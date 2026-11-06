"""The FlyVis Windows HDF5 writer must preserve data on a failed write."""
from pathlib import Path

import numpy as np
import pytest

from tracking.flyvis_adapter import DEFAULT_RUNTIME_DIR, _patch_datamate_hdf5_writer

pytestmark=pytest.mark.skipif(not (DEFAULT_RUNTIME_DIR/'flyvis').is_dir(),reason='local FlyVis runtime not installed')


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.syspath_prepend(str(DEFAULT_RUNTIME_DIR))
    _patch_datamate_hdf5_writer()
    import datamate.io
    return datamate.io._write_h5


def test_invalid_hdf5_value_preserves_existing_file(writer,tmp_path:Path):
    path=tmp_path/'existing.h5'
    writer(path,np.arange(6,dtype=np.float32))
    before=path.read_bytes()
    with pytest.raises(TypeError):
        writer(path,np.array([object()],dtype=object))
    assert path.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['existing.h5']


def test_invalid_hdf5_value_does_not_publish_partial_new_file(writer,tmp_path:Path):
    path=tmp_path/'new.h5'
    with pytest.raises(TypeError):
        writer(path,np.array([object()],dtype=object))
    assert not path.exists()
    assert list(tmp_path.iterdir())==[]
