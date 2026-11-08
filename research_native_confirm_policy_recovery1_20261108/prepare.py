"""Fix signed aggregation before comparison; preserve frozen original source."""
import sys,time,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    assert not (RUN/'comparison_registration.json').exists() and not (RUN/'comparison_report.json').exists()
    old=ROOT/'research_native_confirm_policy_20261108/compare.py';source=old.read_text(encoding='utf-8');needle="for arm in ARMS:results[split][arm]['totals']=dict(sum((Counter(v) for v in results[split][arm]['per_video'].values()),Counter()))"
    replacement="for arm in ARMS:\n            vv=list(results[split][arm]['per_video'].values());kk=set().union(*(v.keys() for v in vv));results[split][arm]['totals']={k:sum(v.get(k,0) for v in vv) for k in kk}"
    assert source.count(needle)==1;new=Path(__file__).parent/'compare.py';new.write_text(source.replace(needle,replacement),encoding='utf-8',newline='\n');ast.parse(new.read_text(encoding='utf-8'))
    # A negative per-video J must reduce the total, not disappear through Counter+.
    vv=[{'correct':4,'foreign':1,'J':3},{'correct':0,'foreign':2,'J':-2}];kk=set().union(*(v.keys() for v in vv));check={k:sum(v.get(k,0) for v in vv) for k in kk};assert check['J']==1 and check['J']==check['correct']-check['foreign']
    oldplanner=ROOT/'research_native_confirm_policy_20261108/comparison_planner_controller.py';planner=oldplanner.read_text(encoding='utf-8');assert planner.count("script='research_native_confirm_policy_20261108/compare.py'")==1
    newplanner=Path(__file__).parent/'comparison_planner_controller.py';newplanner.write_text(planner.replace("script='research_native_confirm_policy_20261108/compare.py'","script='research_native_confirm_policy_recovery1_20261108/compare.py'"),encoding='utf-8',newline='\n');ast.parse(newplanner.read_text(encoding='utf-8'))
    receipt=read(RUN/'comparison_planner_started.json');root=psutil.Process(receipt['pid']);assert abs(root.create_time()-receipt['created'])<.01
    owned=[root,*root.children(recursive=True)];verified=[]
    for p in owned:
        if 'python' in p.name().lower():assert any('comparison_planner_controller.py' in a for a in p.cmdline()),(p.pid,p.cmdline())
        verified.append(dict(pid=p.pid,created=p.create_time(),command=p.cmdline()))
    for info in reversed(verified):
        try:
            p=psutil.Process(info['pid']);assert abs(p.create_time()-info['created'])<.01;p.terminate();p.wait(timeout=5)
        except psutil.NoSuchProcess:pass
    write(RUN/'comparison_preflight_correction.json',dict(passed=True,before_any_comparison=True,reason='Counter addition drops negative per-video J. Replace only signed total aggregation; no scientific output existed and no ready calculation is repeated.',signed_probe=check,old_source_sha256=sha(old),new_source_sha256=sha(new),stopped_only_waiting_service=verified),new=True)
    with (RUN/'comparison_planner_recovery1.log').open('x',encoding='utf-8') as out,(RUN/'comparison_planner_recovery1.err').open('x',encoding='utf-8') as err:p=subprocess.Popen([sys.executable,'-u','research_native_confirm_policy_recovery1_20261108/comparison_planner_controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    r=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_native_confirm_policy_recovery1_20261108/comparison_planner_controller.py');write(RUN/'comparison_planner_recovery1_started.json',r,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(r)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    print(r)


if __name__=='__main__':main()
