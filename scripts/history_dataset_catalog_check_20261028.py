"""Проверки новой карточки подготовки данных до CPU-экспорта пар."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess,time
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'history_dataset_checks';end=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp();checks=[]
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_history_training_data.py','tests/test_history_representations.py','-q']]):
        service.write(folder/'status.json',dict(state='running',command=args,updated=service.now()),required=False)
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-10))
        checks.append(dict(command=args,exit_code=r.returncode));assert r.returncode==0
    service.write(folder/'report.json',dict(passed=True,checks=checks,updated=service.now()))

if __name__=='__main__':main()
