"""Supervised, authorized window; only own child PID is stopped at deadline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/fly_efference_eye_v2'
deadline=datetime.fromisoformat('2026-10-12T07:40:00+03:00').timestamp()
env={**os.environ,'EFFERENCE_DEADLINE':str(deadline),'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'3','MKL_NUM_THREADS':'3','OPENBLAS_NUM_THREADS':'3','FOVEA_THREADS':'3'}


def run(script,name):
    seconds=deadline-time.time()-60
    if seconds<180: return 'not_started_deadline'
    with (OUT/f'{name}.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (OUT/'child.json').write_text(json.dumps({'pid':child.pid,'stage':name,'deadline':deadline}),encoding='utf-8')
        try: return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            return 'stopped_deadline'


result={'started':time.time(),'deadline':deadline,'gate_exit':run('fly_efference_v2.py','gate')}
if result['gate_exit']==0:
    gate=json.loads((OUT/'gate.json').read_text(encoding='utf-8'))
    result['gate_passed']=gate['all_passed']
    if gate['all_passed']:
        result['stage1_exit']=run('fly_efference_stage1_v2.py','stage1')
result['finished']=time.time()
(OUT/'night_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
