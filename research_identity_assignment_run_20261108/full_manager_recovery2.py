"""Light controller: sequential separately guarded stages, no resident neural model."""
import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL
from research_identity_assignment_run_20261108.full_master import output_json,raw_path,metric_copy,cleanup


def verify_receipt(arm,video):
    r=read(FULL/'rollouts'/arm/(video+'.receipt.json'));assert r['passed']
    for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
    for n,h in r['feature_files'].items():assert sha(ROOT/n)==h


def launch(role,video,script,args):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,9,7,39,30,tzinfo=MSK),'Author deadline reached'
    folder=FULL/'isolated_children';folder.mkdir(parents=True,exist_ok=True);name=role+'_'+video
    assert not (folder/(name+'.log')).exists(),'A failed stage needs an explicit versioned recovery.'
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS if os.name=='nt' else 0
        proc=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/'+script,*args],cwd=ROOT,stdout=out,stderr=err,creationflags=flags)
        write(folder/(name+'.started.json'),dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),role=role,video=video),new=True)
        code=proc.wait()
    lines=(folder/(name+'.log')).read_text(encoding='utf-8').splitlines();reports=[line for line in lines if line.startswith('SYSTEM_MEMORY_GUARD ')]
    assert reports,(role,video,'No completed guardian receipt',code)
    report=json.loads(reports[-1].split(' ',1)[1]);owned_ended=True
    for pid,birth in report['owned'].items():
        try:process=psutil.Process(int(pid));owned_ended &= abs(process.create_time()-birth)>.01
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=owned_ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and owned_ended,(role,video,code,report)


def main():
    verify_phase('full_execution_recovery2_phase.json');p=read(FULL/'protocol.json');names=list(p['videos']);began=time.perf_counter()
    assert read(FULL/'zero_parity.json')['passed'] and read(FULL/'late_cpu_completion_audit.json')['passed']
    for index,video in enumerate(names):
        completed=FULL/'completed'/(video+'.json')
        if completed.exists():
            r=read(completed)
            for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
            cleanup(video);continue
        started=time.perf_counter();entry=p['videos'][video]
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
        if not output_json('B0',video).exists():write(output_json('B0',video),read(ROOT/entry['baseline']),new=True)
        if not (FULL/'rollouts/G_R1'/(video+'.receipt.json')).exists():launch('cpu',video,'full_cpu_guarded.py',[video])
        for arm in ['G','G_R1']:verify_receipt(arm,video)
        for arm,source in p['aliases'].items():
            assert p['parameters'][arm]['beta']==0 and p['parameters'][arm]['alpha']==p['parameters'][source]['alpha']
            if not output_json(arm,video).exists():write(output_json(arm,video),read(output_json(source,video)),new=True)
        if not (FULL/'rollouts/G_PERM'/(video+'.receipt.json')).exists():launch('gpu',video,'full_gpu_guarded.py',[video])
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
            if same is None:launch('eval_'+arm,video,'full_eval_guarded.py',[video,arm])
            else:metric_copy(same,arm,video)
            done.append(arm)
        artifacts={}
        for arm in p['parameters']:
            for path in [output_json(arm,video),FULL/'metrics'/arm/(video+'.json'),FULL/'metrics'/arm/(video+'.details.json.gz')]:artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
        write(completed,dict(passed=True,video=video,seconds=time.perf_counter()-started,artifacts=artifacts,
            protocol_sha256=sha(FULL/'protocol.json'),selection_sha256=p['selection_sha256']),new=True)
        cleanup(video)
        if index==0 and not (FULL/'first_video_price.json').exists():
            remaining=(datetime(2026,11,9,7,40,tzinfo=MSK)-datetime.now(MSK)).total_seconds();estimate=(time.perf_counter()-started)*2*(len(names)-1)+300
            write(FULL/'first_video_price.json',dict(seconds=time.perf_counter()-started,conservative_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate<remaining),new=True)
            assert estimate<remaining
        status('full_video_complete',video=video,complete=index+1,total=len(names),elapsed_seconds=time.perf_counter()-began)
    verify_phase('full_execution_recovery2_phase.json')
    write(FULL/'rollouts_complete.json',dict(passed=True,videos=len(names),arms=list(p['parameters']),all_causal=True,baseline_reused=True,
        core_weights_unchanged=True,execution='isolated sequential externally guarded workers',seconds=time.perf_counter()-began),new=True)


if __name__=='__main__':main()
