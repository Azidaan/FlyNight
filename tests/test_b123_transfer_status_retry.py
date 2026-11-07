from pathlib import Path
import pytest
from scripts import b123_supported_transfer_recovery1_20261027 as recovery


def test_transient_write_lock_retries_and_persists(tmp_path,monkeypatch):
    original=Path.replace;calls=[]
    def replace(path,target):
        calls.append(target)
        if len(calls)==1:raise PermissionError('Windows sharing violation')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',replace);monkeypatch.setattr(recovery.time,'sleep',lambda _:None)
    assert recovery.write(tmp_path/'status.json',{'done':500}) and len(calls)==2


def test_status_failure_is_bounded_but_scientific_result_is_not_silently_dropped(tmp_path,monkeypatch):
    def replace(path,target):raise PermissionError('persistent lock')
    monkeypatch.setattr(Path,'replace',replace);monkeypatch.setattr(recovery.time,'sleep',lambda _:None)
    assert recovery.write(tmp_path/'status.json',{'done':500}) is False
    with pytest.raises(PermissionError):recovery.write(tmp_path/'report.json',{'complete':True})
