"""Проверка исправленной загрузки, с сохранением двух прежних отказов."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    old=RUN/'b2_cpu_loader_recovery2';launch=service.read(old/'launch.json')
    assert not service.matching(launch['pid'],launch['created'])
    service.write(old/'failure_verified.json',dict(exit_code=1,actual_pid_ended=True,launch=launch,
        error="ImportError: cannot import name 'initialization' from 'flyvis.network.network'",
        location='tracking/b2_cpu_loader_recovery2.py:7',
        source_sha256=service.sha(ROOT/'tracking/b2_cpu_loader_recovery2.py'),updated=service.now()))
    source=ROOT/'scripts/check_b2_cpu_loader_recovery1_20261029.py'
    code=source.read_text(encoding='utf-8').replace('b2_cpu_loader_recovery1','b2_cpu_loader_recovery3')
    ns={'__name__':'b2_cpu_check_recovery3','__file__':str(source)}
    exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
