"""The FlyVis Windows HDF5 writer must preserve data on a failed write."""
from pathlib import Path
import os
import sys
import types

import numpy as np
import pytest

from tracking.flyvis_adapter import DEFAULT_RUNTIME_DIR, _patch_datamate_hdf5_writer

pytestmark=pytest.mark.skipif(os.name!='nt',reason='Windows-specific datamate adapter')


@pytest.fixture
def writer(monkeypatch):
    # Exercise real h5py while isolating datamate's two imported aliases.
    # h5py is bundled with the project runtime rather than the main venv.
    if DEFAULT_RUNTIME_DIR.exists():
        monkeypatch.syspath_prepend(str(DEFAULT_RUNTIME_DIR))
    pytest.importorskip('h5py')
    package=types.ModuleType('datamate');package.__path__=[]
    io=types.ModuleType('datamate.io');directory=types.ModuleType('datamate.directory')
    io._write_h5=lambda *args:None
    directory._write_h5=io._write_h5
    package.io=io;package.directory=directory
    for module in (package,io,directory):
        monkeypatch.setitem(sys.modules,module.__name__,module)
    _patch_datamate_hdf5_writer()
    assert directory._write_h5 is io._write_h5
    return io._write_h5


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


@pytest.mark.parametrize('value',[
    np.arange(6,dtype=np.float32).reshape(2,3),np.arange(3,dtype=np.int64),
    np.array([True,False]),np.array([1+2j],dtype=np.complex64),
    np.array([b'one',b'two']),np.array(3.25,dtype=np.float64),np.empty((0,2),np.float32),
])
def test_valid_data_dtype_shape_and_swmr_are_preserved(writer,tmp_path,value):
    import h5py
    path=tmp_path/'value.h5'
    for _ in range(2):
        writer(path,value)
        with h5py.File(path,'r',libver='latest',swmr=True) as handle:
            assert handle.swmr_mode
            assert handle['data'].dtype==value.dtype
            np.testing.assert_array_equal(handle['data'][()],value)
        assert sorted(p.name for p in tmp_path.iterdir())==['value.h5']


@pytest.mark.parametrize('existing',[True,False])
def test_publish_failure_preserves_old_file_or_absence(writer,tmp_path,monkeypatch,existing):
    path=tmp_path/'value.h5'
    if existing:writer(path,np.arange(4))
    before=path.read_bytes() if existing else None
    def refused(*args):raise PermissionError('test: replacement refused')
    monkeypatch.setattr(os,'replace',refused)
    with pytest.raises(PermissionError,match='replacement refused'):
        writer(path,np.arange(8))
    assert path.read_bytes()==before if existing else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir())==(['value.h5'] if existing else [])


def test_open_windows_hdf5_reader_prevents_replacement_without_data_loss(writer,tmp_path):
    import h5py
    path=tmp_path/'value.h5';value=np.arange(4)
    writer(path,value);before=path.read_bytes()
    with h5py.File(path,'r',swmr=True) as reader:
        with pytest.raises(OSError):writer(path,np.arange(8))
        np.testing.assert_array_equal(reader['data'][()],value)
        assert path.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['value.h5']
    writer(path,np.arange(8))


def test_empty_directory_is_removed_only_after_successful_write(writer,tmp_path,monkeypatch):
    import h5py
    path=tmp_path/'value.h5';path.mkdir()
    with pytest.raises(TypeError):writer(path,np.array([object()],dtype=object))
    assert path.is_dir() and list(path.iterdir())==[]
    original_replace=os.replace
    monkeypatch.setattr(os,'replace',lambda *args:(_ for _ in ()).throw(PermissionError('refused')))
    with pytest.raises(PermissionError):writer(path,np.arange(4))
    assert path.is_dir() and list(path.iterdir())==[]
    monkeypatch.setattr(os,'replace',original_replace)
    writer(path,np.arange(4))
    with h5py.File(path,'r',swmr=True) as reader:
        np.testing.assert_array_equal(reader['data'][()],np.arange(4))
    assert sorted(p.name for p in tmp_path.iterdir())==['value.h5']


def test_nonempty_directory_is_never_removed_or_renamed(writer,tmp_path):
    path=tmp_path/'value.h5';path.mkdir();(path/'keep').write_bytes(b'preserved')
    with pytest.raises(OSError):writer(path,np.arange(4))
    assert (path/'keep').read_bytes()==b'preserved'
    assert sorted(p.name for p in tmp_path.iterdir())==['value.h5']


@pytest.mark.parametrize('stage',['flush','swmr'])
def test_flush_or_swmr_failure_preserves_existing_data(writer,tmp_path,monkeypatch,stage):
    import h5py
    path=tmp_path/'value.h5';writer(path,np.arange(5));before=path.read_bytes()
    def refused(*args):raise RuntimeError('test: '+stage+' failed')
    if stage=='flush':
        monkeypatch.setattr(h5py.File,'flush',refused)
    else:
        original=h5py.File.swmr_mode
        monkeypatch.setattr(h5py.File,'swmr_mode',property(original.fget,refused))
    with pytest.raises(RuntimeError,match=stage+' failed'):writer(path,np.arange(8))
    assert path.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['value.h5']


def test_hardlinked_target_is_rejected_instead_of_splitting_aliases(writer,tmp_path):
    path=tmp_path/'value.h5';alias=tmp_path/'alias.h5'
    writer(path,np.arange(5));os.link(path,alias);before=path.read_bytes()
    with pytest.raises(OSError,match='hard link'):writer(path,np.arange(8))
    assert path.read_bytes()==alias.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['alias.h5','value.h5']


def test_resolved_file_link_target_is_published_without_replacing_requested_path(writer,tmp_path,monkeypatch):
    # Contract test independent of Windows's symlink-creation privilege. The
    # native symlink test below is kept separately and reports a skip honestly.
    target=tmp_path/'target.h5';link=tmp_path/'link.h5'
    writer(target,np.arange(4));link.write_bytes(b'link-object-placeholder')
    resolve,is_symlink=Path.resolve,Path.is_symlink
    monkeypatch.setattr(Path,'resolve',lambda p,*a,**kw:target if p==link else resolve(p,*a,**kw))
    monkeypatch.setattr(Path,'is_symlink',lambda p:True if p==link else is_symlink(p))
    writer(link,np.arange(8))
    import h5py
    with h5py.File(target,'r',swmr=True) as reader:
        np.testing.assert_array_equal(reader['data'][()],np.arange(8))
    assert link.read_bytes()==b'link-object-placeholder'


def symlink(link,target):
    try:link.symlink_to(target,target_is_directory=target.is_dir())
    except OSError as error:
        if getattr(error,'winerror',None)==1314:
            pytest.skip('Creating symbolic links requires Windows privilege')
        raise


def test_file_symlink_stays_a_link_and_updates_target(writer,tmp_path,monkeypatch):
    import h5py
    target=tmp_path/'target.h5';link=tmp_path/'link.h5'
    writer(target,np.arange(4));symlink(link,target)
    writer(link,np.arange(7))
    assert link.is_symlink() and link.resolve()==target.resolve()
    with h5py.File(target,'r',swmr=True) as reader:
        np.testing.assert_array_equal(reader['data'][()],np.arange(7))
    before=target.read_bytes()
    with pytest.raises(TypeError):writer(link,np.array([object()],dtype=object))
    assert link.is_symlink() and target.read_bytes()==before
    monkeypatch.setattr(os,'replace',lambda *args:(_ for _ in ()).throw(PermissionError('refused')))
    with pytest.raises(PermissionError):writer(link,np.arange(9))
    assert link.is_symlink() and target.read_bytes()==before
    assert sorted(p.name for p in tmp_path.iterdir())==['link.h5','target.h5']


def test_symlink_to_directory_never_removes_target(writer,tmp_path):
    target=tmp_path/'target';target.mkdir();link=tmp_path/'link.h5';symlink(link,target)
    with pytest.raises(IsADirectoryError):writer(link,np.arange(3))
    assert link.is_symlink() and target.is_dir() and list(target.iterdir())==[]


@pytest.mark.skipif(not (DEFAULT_RUNTIME_DIR/'datamate').is_dir(),reason='vendor datamate absent')
def test_real_datamate_directory_writer_and_reader_contract(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(DEFAULT_RUNTIME_DIR))
    _patch_datamate_hdf5_writer()
    from datamate import Directory
    directory=Directory(tmp_path/'record')
    directory.array=np.arange(6,dtype=np.float32)
    np.testing.assert_array_equal(directory.array[()],np.arange(6,dtype=np.float32))
    path=directory.path/'array.h5';before=path.read_bytes()
    with pytest.raises(TypeError):directory.array=np.array([object()],dtype=object)
    assert path.read_bytes()==before
