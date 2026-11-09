"""Finite sequential continuation; all scientific children have their own guard."""
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,6,35,tzinfo=timezone(timedelta(hours=3)))
PREDECESSOR=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/993e0fec2811486c8e7c5930677ae16a/memory_guard.json'
record=dict(started=datetime.now().astimezone().isoformat(),own_pid=os.getpid(),own_birth=psutil.Process().create_time(),stages=[],state='waiting_for_retina_guard',recovery='The old queue.py shadowed Python stdlib queue during Torch import. Renamed controller; completed retinal, mathematics and map registration are reused. No model inference had started.')
def save():(OUT/'sequence_resume1_status.json').write_text(json.dumps(record,indent=2),encoding='utf8')
save();psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
try:
    while True:
        r=json.loads(PREDECESSOR.read_text(encoding='utf8'))
        if r['returncode'] is not None:
            assert r['returncode']==0 and r['reason'] is None,r
            break
        if datetime.now(STOP.tzinfo)>=STOP:raise RuntimeError('Scientific save/stop deadline reached while waiting')
        time.sleep(3)
    assert (OUT/'retinal_complete.json').exists()
    assert (OUT/'balanced_math_audit.json').exists() and (OUT/'map_protocol.json').exists()
    stages=[('recover_maps.py','run'),('compare.py','register'),('compare.py','run'),('compare.py','analyze')]
    for script,action in stages:
        if datetime.now(STOP.tzinfo)>=STOP:raise RuntimeError('No new scientific stage after save/stop')
        label=Path(script).stem+('_'+action if action else '')+'_resume1';cmd=[sys.executable,'-u',str(OUT/script)]+([action] if action else [])
        with (OUT/(label+'.log')).open('xb') as out,(OUT/(label+'.err')).open('xb') as err:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);proc=psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS)
            item=dict(stage=label,pid=child.pid,birth=proc.create_time(),started=datetime.now().astimezone().isoformat(),returncode=None);record['stages'].append(item);record['state']='running';save()
            while child.poll() is None:time.sleep(2)
            item.update(returncode=child.returncode,ended=datetime.now().astimezone().isoformat());save()
            assert child.returncode==0,(label,child.returncode)
        print('FINISHED',label,flush=True)
    record['state']='science_complete';save()
except BaseException as error:
    record.update(state='failed',error=repr(error));save();raise
finally:
    record['ended']=datetime.now().astimezone().isoformat();save()
