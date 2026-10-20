import pytest
from scripts import z13_observed_memory_execution_v2 as service


def test_transient_atomic_save_retries_same_target(monkeypatch):
    pauses=[];monkeypatch.setattr(service.time,'sleep',pauses.append)
    class File:
        calls=0
        def replace(self,target):
            assert target=='status.json'
            self.calls+=1
            if self.calls<3:raise PermissionError('reader sharing lock')
    f=File();service.replace_with_retry(f,'status.json')
    assert f.calls==3 and pauses==[.025,.025]


def test_persistent_write_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(service.time,'sleep',lambda _:None)
    class File:
        calls=0
        def replace(self,target):
            self.calls+=1
            raise PermissionError('still locked')
    f=File()
    with pytest.raises(PermissionError):service.replace_with_retry(f,'status.json')
    assert f.calls==40
