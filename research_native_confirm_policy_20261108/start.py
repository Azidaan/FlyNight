import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    from research_native_confirm_policy_20261108.register import main as register
    register();qs=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows'];videos=sorted({q['video'] for q in qs if q['split']=='evaluate26'});exports=read(BASE/'export_complete.json')['videos'];entries={}
    for v in videos:
        raw=f'runs/train_cache/{v}/detections/raw_video.npz';baseline=f'runs/fly_signatures/{v}/signatures.npz';export=exports[v]['path'];assert sha(ROOT/export)==exports[v]['sha256'];entries[v]=dict(raw=raw,baseline=baseline,export=export,sources={n:sha(ROOT/n) for n in [raw,baseline,export,str(Path(export).with_suffix('.npz')).replace('\\','/')]})
    sources=read(RUN/'fit_phase.json')['sources'].copy();sources['research_singleton_native_confirm_20261108/native_trace.py']=sha(ROOT/'research_singleton_native_confirm_20261108/native_trace.py')
    write(RUN/'evaluate26_metadata_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,videos=entries,native_config=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config'],labels_opened=False),new=True)
    with (RUN/'controller.log').open('x',encoding='utf-8') as out,(RUN/'controller.err').open('x',encoding='utf-8') as err:p=subprocess.Popen([sys.executable,'-u','research_native_confirm_policy_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_native_confirm_policy_20261108/controller.py');write(RUN/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(RUN/'deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True);print(root)


if __name__=='__main__':main()
