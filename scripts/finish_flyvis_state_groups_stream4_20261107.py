"""Sequential completion of the authorized RAM-bounded experiment."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,os,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_state_subgroups_ram_stream4_20261107"
PY=ROOT/".venv/Scripts/python.exe"
def save(name,value):
    p=OUT/name;q=p.with_suffix(".next");q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(q,p)
def ended(owned):
    for pid,birth in owned.items():
        try:
            if abs(psutil.Process(int(pid)).create_time()-birth)<.01:return False
        except psutil.NoSuchProcess:pass
    return True
def wait_probe():
    result=OUT/"execution_probe/controller_result.json"
    while not result.exists():
        assert datetime.now(ZoneInfo("Europe/Moscow")).isoformat()<"2026-11-07T04:29:00+03:00"
        time.sleep(1)
    value=json.loads(result.read_text());assert value["exit_code"]==0 and value["all_owned_ended"],value
    launch=json.loads((OUT/"probe.controller.launch.json").read_text())
    owned={**value["owned"],str(launch["pid"]):launch["birth"]}
    for _ in range(30):
        if ended(owned):break
        time.sleep(1)
    assert ended(owned)
    proof=json.loads((OUT/"batch_equivalence.json").read_text())
    assert proof["passed"] and proof["boundary_max_diff"]<=1e-10 and proof["full_field_max_diff"]<=1e-10
    save("probe_ended_verified.json",dict(passed=True,owned=owned,all_owned_ended=True,equivalence=proof))
def run(args,name):
    assert psutil.virtual_memory().available>=12*1024**3
    save("postprocess.stage.json",dict(stage=name,started_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat()))
    own={};peak=0;minimum=psutil.virtual_memory().available;reason=None
    with (OUT/(name+".log")).open("w",encoding="utf-8") as log:
        proc=subprocess.Popen([str(PY)]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,
                              env={**os.environ,"CUDA_VISIBLE_DEVICES":""})
        parent=psutil.Process(proc.pid);own[proc.pid]=parent.create_time()
        if name=="run_controller":
            save("controller.launch.json",dict(pid=proc.pid,birth=own[proc.pid]))
        while proc.poll() is None:
            try:
                for x in [parent]+parent.children(recursive=True):own.setdefault(x.pid,x.create_time())
            except psutil.NoSuchProcess:pass
            private=0
            for pid,birth in own.items():
                try:
                    x=psutil.Process(pid)
                    if abs(x.create_time()-birth)<.01:private+=getattr(x.memory_info(),"private",0)
                except psutil.NoSuchProcess:pass
            peak=max(peak,private);available=psutil.virtual_memory().available;minimum=min(minimum,available)
            if available<12*1024**3:reason="system_memory_reserve";break
            if datetime.now(ZoneInfo("Europe/Moscow")).isoformat()>="2026-11-07T04:29:00+03:00":reason="deadline";break
            time.sleep(.5)
        if reason:
            for pid,birth in reversed(list(own.items())):
                try:
                    x=psutil.Process(pid)
                    if abs(x.create_time()-birth)<.01:x.kill()
                except psutil.NoSuchProcess:pass
        code=proc.wait()
    for _ in range(20):
        if ended(own):break
        time.sleep(.5)
    save(name+".ended.json",dict(exit_code=code,reason=reason,owned=own,all_owned_ended=ended(own),peak_private=peak,minimum_available=minimum))
    assert code==0 and reason is None and ended(own),(name,code,reason)
def bounded_code(code):
    return ["-c","import os;os.environ['CUDA_VISIBLE_DEVICES']='';from tracking.windows_memory_job import MemoryJob;job=MemoryJob(4*1024**3);"+code]
def main():
    save("chain.launch.json",dict(pid=os.getpid(),birth=psutil.Process().create_time(),started_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat()))
    wait_probe()
    run(["scripts/control_flyvis_state_ram_stream4_20261107.py","run"],"run_controller")
    p=json.loads((OUT/"controller_result.json").read_text());assert p["exit_code"]==0 and p["all_owned_ended"]
    run(["scripts/control_flyvis_state_ram_stream4_20261107.py","audit"],"audit_controller")
    a=json.loads((OUT/"independent_audit.json").read_text());assert a["passed"] and a["cases"]==64
    prefix="from pathlib import Path;out=Path('E:/Dissertation/runs/flyvis_state_subgroups_ram_stream4_20261107');"
    run(bounded_code(prefix+"from scripts import record_flyvis_state_subgroups_20261107 as m;m.OUT=out;m.main()"),"record")
    run(bounded_code("import runpy;runpy.run_path('scripts/validate_project_history.py',run_name='__main__')"),"catalog_check")
    run(bounded_code("import pytest;raise SystemExit(pytest.main(['tests/test_windows_memory_job.py','tests/test_flyvis_state_subgroups.py','tests/test_flyvis_state_transplant.py','tests/test_project_history.py','tests/test_obsidian_fundamental_overview.py','-q']))"),"record_tests")
    run(bounded_code(prefix+"from scripts import plot_flyvis_state_subgroups_20261107 as m;m.OUT=out;m.main()"),"plot")
    run(bounded_code(prefix+"from scripts import finalize_flyvis_state_subgroups_20261107 as m;m.OUT=out;m.main()"),"finalize")
    save("postprocess.stage.json",dict(stage="complete",ended_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat()))
    save("chain_result.json",dict(complete=True,all_stages_ended=True,ended_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat()))
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        save("chain_failure.json",dict(error=repr(exc),time_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat()))
        raise
