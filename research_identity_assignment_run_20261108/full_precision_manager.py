"""Sequential isolated workers; native-precision recovery, unchanged scientific protocol."""
import os,sys,time,subprocess,shutil,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
FULL=RUN/'full_precision'


def output_json(arm,video):return FULL/'outputs'/arm/(video+'.json')
def raw_path(p,arm,video):
    return ROOT/p['videos'][video]['raw_array'] if arm=='B0' else FULL/'rollouts'/p['aliases'].get(arm,arm)/(video+'.npz')


def metric_copy(source,target,video):
    folder=FULL/'metrics'/target;folder.mkdir(parents=True,exist_ok=True)
    old=read(FULL/'metrics'/source/(video+'.json'));assert read(output_json(source,video))==read(output_json(target,video))
    details=folder/(video+'.details.json.gz');shutil.copyfile(FULL/'metrics'/source/(video+'.details.json.gz'),details)
    write(folder/(video+'.json'),dict(old,arm=target,reused_from=source,output_sha256=sha(output_json(target,video))),new=True)


def verify_receipt(arm,video):
    r=read(FULL/'rollouts'/arm/(video+'.receipt.json'));assert r['passed']
    for n,h in {**r['artifacts'],**r['feature_files']}.items():assert sha(ROOT/n)==h


def launch(role,video,script,args):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,9,7,39,30,tzinfo=MSK)
    folder=FULL/'children';folder.mkdir(parents=True,exist_ok=True);name=role+'_'+video
    assert not (folder/(name+'.log')).exists(),'Versioned recovery required after failure.'
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS if os.name=='nt' else 0
        proc=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/'+script,*args],cwd=ROOT,stdout=out,stderr=err,creationflags=flags)
        write(folder/(name+'.started.json'),dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),role=role,video=video),new=True)
        code=proc.wait()
    reports=[line for line in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if line.startswith('SYSTEM_MEMORY_GUARD ')]
    assert reports,(role,video,'No guardian receipt',code)
    report=json.loads(reports[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            process=psutil.Process(int(pid))
            if abs(process.create_time()-birth)<.01:
                try:process.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(role,video,code,report)


def release_scratch(video):
    path=FULL/'processed'/(video+'.float64');marker=FULL/'processed'/(video+'.json');receipt=FULL/'processed'/(video+'.released.json')
    if receipt.exists():assert not path.exists();return
    assert (FULL/'completed'/(video+'.json')).exists()
    assert path.resolve().parent==(FULL/'processed').resolve() and path.name==video+'.float64'
    meta=read(marker);assert sha(path)==meta['sha256'];gc.collect();path.unlink()
    write(receipt,dict(released=True,bytes=meta['bytes'],used_sha256=meta['sha256'],
        reason='Registered disposable decoded input scratch. All used 721-cell views, recurrent inputs, outputs, metrics, source movie and resume checkpoints retained.'),new=True)


def main():
    verify_phase('full_precision_execution_phase.json');p=read(FULL/'protocol.json');began=time.perf_counter()
    assert read(FULL/'technical_gate.json')['passed'] and read(FULL/'zero_parity.json')['passed']
    for index,(video,entry) in enumerate(p['videos'].items()):
        completed=FULL/'completed'/(video+'.json')
        if completed.exists():
            for n,h in read(completed)['artifacts'].items():assert sha(ROOT/n)==h
            release_scratch(video);continue
        started=time.perf_counter()
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h
        if not output_json('B0',video).exists():write(output_json('B0',video),read(ROOT/entry['baseline']),new=True)
        if not (FULL/'rollouts/G_R1'/(video+'.receipt.json')).exists():launch('cpu',video,'full_precision_worker.py',['cpu',video])
        for arm in ['G','G_R1']:verify_receipt(arm,video)
        for arm,source in p['aliases'].items():
            assert p['parameters'][arm]['beta']==0 and p['parameters'][arm]['alpha']==p['parameters'][source]['alpha']
            if not output_json(arm,video).exists():write(output_json(arm,video),read(output_json(source,video)),new=True)
        if not (FULL/'rollouts/G_PERM'/(video+'.receipt.json')).exists():launch('gpu',video,'full_precision_worker.py',['gpu',video])
        for arm in ['G_FLYVIS','G_PERM']:verify_receipt(arm,video)
        done=[];arrays={}
        for arm in p['parameters']:
            metric=FULL/'metrics'/arm/(video+'.json')
            if metric.exists():done.append(arm);continue
            with np.load(raw_path(p,arm,video),allow_pickle=False) as z:array=z['outputs']
            same=None
            for prior in done:
                if prior not in arrays:
                    with np.load(raw_path(p,prior,video),allow_pickle=False) as z:arrays[prior]=z['outputs']
                if np.array_equal(array,arrays[prior]):same=prior;break
            arrays[arm]=array
            if same is not None:metric_copy(same,arm,video)
            else:
                old=RUN/'full';reuse=None
                for prior in p['parameters']:
                    mp=old/'metrics'/prior/(video+'.json');rp=old/'rollouts'/p['aliases'].get(prior,prior)/(video+'.npz')
                    if prior=='B0':rp=ROOT/entry['raw_array']
                    if not mp.exists() or not rp.exists():continue
                    with np.load(rp,allow_pickle=False) as z:old_array=z['outputs']
                    if np.array_equal(array,old_array) and read(output_json(arm,video))==read(old/'outputs'/prior/(video+'.json')):
                        reuse=(prior,mp);break
                if reuse:
                    prior,mp=reuse;oldmetric=read(mp);details=FULL/'metrics'/arm/(video+'.details.json.gz');details.parent.mkdir(parents=True,exist_ok=True)
                    source=old/'metrics'/prior/(video+'.details.json.gz');assert sha(source)==oldmetric['details_sha256'];shutil.copyfile(source,details)
                    write(metric,dict(oldmetric,arm=arm,reused_from=str(mp.relative_to(ROOT)),output_sha256=sha(output_json(arm,video))),new=True)
                else:launch('eval_'+arm,video,'full_precision_eval.py',[video,arm])
            done.append(arm)
        artifacts={}
        for arm in p['parameters']:
            for path in [output_json(arm,video),FULL/'metrics'/arm/(video+'.json'),FULL/'metrics'/arm/(video+'.details.json.gz')]:artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h
        verify_phase('full_precision_execution_phase.json')
        write(completed,dict(passed=True,video=video,seconds=time.perf_counter()-started,artifacts=artifacts,
            protocol_sha256=sha(FULL/'protocol.json'),selection_sha256=p['selection_sha256']),new=True)
        release_scratch(video)
        if index==0:
            available=(datetime(2026,11,9,7,40,tzinfo=MSK)-datetime.now(MSK)).total_seconds()
            write(FULL/'first_video_price.json',dict(seconds=time.perf_counter()-started,conservative_remaining_seconds=(time.perf_counter()-started)*2*19+300,available_seconds=available),new=True)
        status('full_native_precision_video_complete',video=video,complete=index+1,total=len(p['videos']),elapsed_seconds=time.perf_counter()-began)
    write(FULL/'rollouts_complete.json',dict(passed=True,videos=len(p['videos']),arms=list(p['parameters']),all_causal=True,
        baseline_reused=True,core_weights_unchanged=True,execution='sequential guarded workers, native precision before resizing',seconds=time.perf_counter()-began),new=True)


if __name__=='__main__':main()
