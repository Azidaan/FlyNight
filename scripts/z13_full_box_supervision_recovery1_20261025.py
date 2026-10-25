"""Batch native costs and reuse two finished videos inside original CPU budget."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import inspect
import hashlib
import time
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_full_box_supervision_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=base.RUN
REC=RUN/'cache/recovery1'


def source():
    s=inspect.getsource(base.cache_worker)
    old='from tracking.z13_box_action_labels import full_box_rows'
    assert s.count(old)==1
    s=s.replace(old,'from tracking.z13_box_action_batch import FullBoxCosts\n    full_box_rows=FullBoxCosts()')
    old="dest=RUN/'cache/videos'/(v+'.json');assert not dest.exists()"
    new="""dest=RUN/'cache/videos'/(v+'.json')
        if dest.exists():
            assert sha(dest)==execution['completed'][v]
            record=read(dest);entries=record['entries'];assert record['checks']['raw_replay_exact']
            manifest[v]=sha(dest);total+=len(entries);known+=sum(x['kind']>=0 for x in entries.values())
            continue"""
    assert s.count(old)==1;s=s.replace(old,new)
    assert s.count('if i==1:')==1;s=s.replace('if i==1:','if len(times)==2:')
    s=s.replace("RUN/'cache/price.json'","REC/'price.json'")
    old="assert total==37296 and known==26600"
    assert s.count(old)==1
    s=s.replace(old,old+"\n    assert full_box_rows.checked>=128\n    write(REC/'actual_cost_parity.json',dict(actual_action_rows_checked=full_box_rows.checked,native_scalar_and_batch_bit_exact=True))")
    return s


def worker():
    execution=read(REC/'execution.json')
    for n,h in execution['sha256'].items():assert sha(ROOT/n)==h
    s=source();assert hashlib.sha256(s.encode()).hexdigest()==execution['source_sha256']
    ns={**vars(base),'execution':execution,'REC':REC};exec(compile(s,__file__,'exec'),ns);ns['cache_worker']()
    write(REC/'status.json',dict(state='completed',stage='Метки полной рамки сохранены',updated=now().isoformat()))


def prepare():
    assert not REC.exists()
    ps=WindowsProcesses();e=read(RUN/'cache/controller_result.json');assert e['exit_code']==1 and e['all_owned_ended']
    assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert read(RUN/'cache/price.json')['proceed'] is False
    assert read(RUN/'batch_cost_parity.json')['native_costs_bit_exact']
    p=read(RUN/'cache/protocol.json');assert now()<base.datetime.fromisoformat(p['save_at'])
    paths=['tracking/z13_box_action_batch.py',Path(__file__).relative_to(ROOT).as_posix(),'runs/z13_full_box_supervision_20261025/batch_cost_parity.json']
    completed={f.stem:sha(f) for f in (RUN/'cache/videos').glob('*.json')};assert len(completed)==2
    write(REC/'execution.json',dict(at=now().isoformat(),cause='Conservative original1200s price failed after2complete videos; no settings/tolerance changes.',fix='Batch exact native distances for all candidate boxes once per original detection-list identity; same geometry, IoU/fused confidence and Hungarian. Old2videos immutable and skipped. Recheck same2x128+60 price formula on first2new videos at new execution cost, inside original save/hard deadline.',completed=completed,source_sha256=hashlib.sha256(source().encode()).hexdigest(),sha256={n:sha(ROOT/n) for n in paths}))
    q=dict(p,stop_request_path=(RUN/'cache/stop.request').relative_to(ROOT).as_posix())
    write(REC/'protocol.json',q);write(REC/'status.json',dict(state='ready',stage='Пакетный точный расчёт без повторения2видео',updated=now().isoformat()))
    base.journal('служебная пакетная версия в прежнем бюджете','Первый контроль стоимости cache/price.json не пройден:1544,032с против1170,337с. Исходные ошибка/контроллер/код сохранены,2видео не повторяются. '+read(REC/'execution.json')['fix']+' Доказательство численного равенства batch_cost_parity.json; новая версия отдельно хеширована, исходные сроки '+p['save_at']+' / '+p['hard_stop']+' не продлены.')


def pipeline():
    ps=WindowsProcesses();write(REC/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (REC/'controller.stdout.log').open('w') as out,(REC/'controller.stderr.log').open('w') as err:
            proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(REC),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            while proc.poll() is None:
                try:write(RUN/'status.json',dict(read(RUN/'cache/status.json'),service='recovery1',observer_at=now().isoformat()))
                except (OSError,ValueError):pass
                time.sleep(2)
            proc.wait()
        e=read(REC/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and read(RUN/'cache/report.json')['complete']
        base.phase_protocol('training');folder=RUN/'training'
        with (folder/'controller.stdout.log').open('w') as out,(folder/'controller.stderr.log').open('w') as err:
            proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(ROOT/'scripts/z13_full_box_supervision_20261025.py')],cwd=ROOT,env={**os.environ,'Z13_FULL_BOX_PHASE':'training'},stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            while proc.poll() is None:
                try:write(RUN/'status.json',dict(read(folder/'status.json'),phase='training',observer_at=now().isoformat()))
                except (OSError,ValueError):pass
                time.sleep(2)
            proc.wait()
        e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and read(folder/'report.json')['complete']
        write(REC/'compute_complete.json',dict(at=now().isoformat(),both_phases_complete=True))
    except Exception as error:write(REC/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


def finish():
    execution=read(REC/'execution.json')
    for n,h in execution['sha256'].items():assert sha(ROOT/n)==h
    s=inspect.getsource(base.finish)
    old="e=read(RUN/phase/'controller_result.json')"
    assert s.count(old)==1
    s=s.replace(old,"e=read((REC if phase=='cache' else RUN/phase)/'controller_result.json')")
    ns={**vars(base),'REC':REC};exec(compile(s,__file__,'exec'),ns);ns['finish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--finish' in sys.argv:finish()
