"""Проверки после исправления формата ссылки, в остатке исходных600с."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from datetime import datetime
import sys, inspect, time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN, service


def main():
    folder=RUN/'b2_causal960_checks_recovery1'
    p=service.read(folder/'protocol.json')
    assert service.sha(ROOT/'scripts/check_b2_causal960_20261028.py') == p['original_script_sha256']
    assert time.time() < datetime.fromisoformat(p['original_deadline']).timestamp()-30
    original=service.read(RUN/'b2_causal960_checks/controller_result.json')
    assert original['exit_code']==1 and original['all_owned_ended']
    assert service.sha(RUN/'b2_causal960_checks/controller_result.json')==p['original_controller_sha256']
    assert not any(service.matching(k,v) for k,v in original['owned'].items())
    from scripts import check_b2_causal960_20261028 as old
    source=inspect.getsource(old.main)
    oldtext="folder = RUN / 'b2_causal960_checks'"
    assert source.count(oldtext)==1
    ns=vars(old).copy()
    exec(compile(source.replace(oldtext,"folder = RUN / 'b2_causal960_checks_recovery1'"),__file__,'exec'),ns)
    ns['main']()


if __name__=='__main__':main()
