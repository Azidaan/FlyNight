import hashlib
import os
from scripts.z13_center_hota_service_20261021 import verify_files,fingerprint
import pytest


def fixture_files(root):
    files=['code.py','video1/cache.npz','video2/cache.npz']
    for name in files:
        path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'original')
    return ({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},{n:fingerprint(root/n) for n in files})


@pytest.mark.parametrize('path,video,full',[('code.py',None,False),('video1/cache.npz','video1',False),('video2/cache.npz',None,True)])
def test_hash_scope_detects_changed_bytes_even_with_preserved_size_and_time(tmp_path,path,video,full):
    expected,stats=fixture_files(tmp_path);target=tmp_path/path;st=target.stat();target.write_bytes(b'modified');os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns))
    with pytest.raises(RuntimeError,match='Frozen execution input changed'):verify_files(tmp_path,expected,stats,video,full)


def test_changed_stat_of_unselected_video_forces_hash_and_unchanged_inputs_pass(tmp_path):
    expected,stats=fixture_files(tmp_path);assert verify_files(tmp_path,expected,stats,'video1')==2
    target=tmp_path/'video2/cache.npz';st=target.stat();target.write_bytes(b'changed!');os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns+100000))
    with pytest.raises(RuntimeError):verify_files(tmp_path,expected,stats,'video1')
