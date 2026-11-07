"""Проверки каталога/причинных входов/градиента до нового обучения."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    f=RUN/'history_model_checks'
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_history_pair_model.py','-q']]):
        with (f/f'check_{i}.log').open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=260)
        assert result.returncode==0
    service.write(f/'report.json',dict(passed=True,checks=2,updated=service.now()))


if __name__=='__main__':main()
