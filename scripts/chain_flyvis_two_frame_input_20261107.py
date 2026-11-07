"""One finite hidden chain, with PID/birth tracking and preserved stage errors."""
import os,sys,time,subprocess,json
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,write
if __name__=='__main__':
    result=dict(passed=False,owned={},steps=[])
    for name,args in [
        ('validator',['scripts/check_flyvis_two_frame_input_20261107.py','validator']),
        ('catalog',['scripts/check_flyvis_two_frame_input_20261107.py','catalog']),
        ('gpu',['scripts/flyvis_two_frame_input_20261107.py']),
        ('audit',['scripts/flyvis_two_frame_input_20261107.py','--audit'])]:
        with (OUT/(name+'.log')).open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            result['owned'][str(child.pid)]=psutil.Process(child.pid).create_time()
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):
                        result['owned'].setdefault(str(p.pid),p.create_time())
                except psutil.NoSuchProcess:
                    pass
                write(OUT/'chain_status.json',dict(stage=name,owned=result['owned']))
                time.sleep(.3)
            code=child.wait()
        result['steps'].append(dict(stage=name,code=code))
        if code!=0:
            result['failed']=name;write(OUT/'controller.json',result);raise SystemExit(code)
    result['passed']=True;write(OUT/'controller.json',result)
