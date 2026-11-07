"""Finite hidden cached-data chain; no detached unregistered children."""
import os,sys,time,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,write
if __name__=='__main__':
    me=psutil.Process();result=dict(passed=False,owned={str(me.pid):me.create_time()},steps=[])
    stages=[('unit',['scripts/check_flyvis_readout_comparison_20261107.py','unit']),
        ('validator',['scripts/check_flyvis_readout_comparison_20261107.py','validator']),
        ('catalog',['scripts/check_flyvis_readout_comparison_20261107.py','catalog']),
        ('data',['scripts/flyvis_readout_comparison_20261107.py','--data']),
        ('train',['scripts/flyvis_readout_comparison_20261107.py','--train']),
        ('audit',['scripts/check_flyvis_readout_comparison_20261107.py','audit'])]
    for stage,args in stages:
        with (OUT/(stage+'.log')).open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            result['owned'][str(child.pid)]=psutil.Process(child.pid).create_time()
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):result['owned'].setdefault(str(p.pid),p.create_time())
                except psutil.NoSuchProcess:pass
                write(OUT/'chain_status.json',dict(stage=stage,updated=time.time(),owned=result['owned']))
                time.sleep(.3)
            code=child.wait()
        result['steps'].append(dict(stage=stage,code=code))
        if code:
            result['failed']=stage;write(OUT/'controller.json',result);raise SystemExit(code)
    result['passed']=True;write(OUT/'controller.json',result)
