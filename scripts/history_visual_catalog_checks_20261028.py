"""Подготовка нового GPU-кэша: сначала каталог и причинные проверки."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,subprocess
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    f=RUN/'history_visual_checks';end=datetime.fromisoformat(service.read(f/'execution_deadline.json')['deadline']).timestamp();checks=[]
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_history_training_data.py','tests/test_history_representations.py','tests/test_causal_motion_history.py','-q']]):
        service.write(f/'status.json',dict(state='running',step=i,command=args,updated=service.now()),required=False)
        with (f/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-10))
        checks.append(dict(command=args,exit_code=r.returncode));assert r.returncode==0
    service.write(f/'report.json',dict(passed=True,checks=checks,updated=service.now()))

if __name__=='__main__':main()
