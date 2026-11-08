"""One guarded parent, sequential CPU children, and one persistent frozen CUDA core."""
import os,sys,time,subprocess,shutil,gc
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def child(stage,video,script,args,obs):
    import psutil
    folder=FULL/'children';folder.mkdir(parents=True,exist_ok=True);name=stage+'_'+video
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        argv=[sys.executable,'-u','research_identity_assignment_run_20261108/'+script,*args,'--parent-report',obs.read()['supervisor_report']]
        flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS if os.name=='nt' else 0
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=out,stderr=err,creationflags=flags)
        write(folder/(name+'.started.json'),dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),argv=argv,parent_guard=obs.read()['supervisor_report']),new=True)
        result=proc.wait()
    write(folder/(name+'.ended.json'),dict(returncode=result,log_sha256=sha(folder/(name+'.log')),err_sha256=sha(folder/(name+'.err'))),new=True)
    assert result==0,(stage,video,result)


def output_json(arm,video):return FULL/'outputs'/arm/(video+'.json')
def raw_path(p,arm,video):
    if arm=='B0':return ROOT/p['videos'][video]['raw_array']
    arm=p['aliases'].get(arm,arm)
    return FULL/'rollouts'/arm/(video+'.npz')


def metric_copy(source,target,video):
    folder=FULL/'metrics'/target;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(video+'.json');details=folder/(video+'.details.json.gz')
    if marker.exists():assert read(marker)['details_sha256']==sha(details);return
    old=read(FULL/'metrics'/source/(video+'.json'))
    assert read(output_json(source,video))==read(output_json(target,video))
    shutil.copyfile(FULL/'metrics'/source/(video+'.details.json.gz'),details)
    result=dict(old);result.update(arm=target,reused_from=source,output_sha256=sha(output_json(target,video)))
    write(marker,result,new=True)


def cleanup(video):
    path=FULL/'processed'/(video+'.float32');marker=FULL/'processed'/(video+'.json')
    receipt=FULL/'processed'/(video+'.released.json')
    if receipt.exists():assert not path.exists();return
    assert (FULL/'completed'/(video+'.json')).exists()
    assert path.resolve().parent==(FULL/'processed').resolve() and path.name==video+'.float32'
    assert sha(path)==read(marker)['sha256'];gc.collect();path.unlink()
    write(receipt,dict(released=True,bytes=read(marker)['bytes'],used_sha256=read(marker)['sha256'],
        reason='All causal arm outputs, metrics, used retinal views and feature arrays sealed. Only derived full-frame scratch released.'),new=True)


def main(obs):
    verify_phase('full_execution_recovery1_phase.json');p=read(FULL/'protocol.json');assert read(FULL/'technical_gate.json')['passed']
    names=list(p['videos']);first=names[0];began=time.perf_counter();backbone=None;weights=None;gpu_lock=None
    assert read(RUN/'coverage_followup_recovery1.json')['passed']
    zero=FULL/'rollouts/B0'/(first+'.receipt.json')
    if not zero.exists():child('zero',first,'full_cpu.py',[first,'--zero'],obs)
    assert read(zero)['passed'] and read(zero)['zero']
    if not (FULL/'zero_parity.json').exists():
        write(FULL/'zero_parity.json',dict(passed=True,video=first,frames=p['videos'][first]['frames'],
            raw_arrays_exact=True,scalabel_json_exact=True,receipt_sha256=sha(zero)),new=True)
    else:assert read(FULL/'zero_parity.json')['receipt_sha256']==sha(zero)
    from research_identity_assignment_run_20261108.full_run import rollout
    try:
        for index,video in enumerate(names):
            completed=FULL/'completed'/(video+'.json')
            if completed.exists():
                r=read(completed)
                for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
                cleanup(video);continue
            started=time.perf_counter();entry=p['videos'][video]
            for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
            if not output_json('B0',video).exists():write(output_json('B0',video),read(ROOT/entry['baseline']),new=True)
            if not (FULL/'rollouts/G_R1'/(video+'.receipt.json')).exists():child('cpu',video,'full_cpu.py',[video],obs)
            for arm,source in p['aliases'].items():
                assert p['parameters'][arm]['beta']==0 and p['parameters'][arm]['alpha']==p['parameters'][source]['alpha']
                if not output_json(arm,video).exists():write(output_json(arm,video),read(output_json(source,video)),new=True)
            if backbone is None:
                import torch
                from tracking.gpu_lock import GpuLock
                gpu_lock=GpuLock('flyvis_fovea_gpu');gpu_lock.__enter__()
                torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
                from scripts.z13_target_model import build_arm
                with torch.no_grad():_,_,backbone=build_arm('C',torch,None)
                weights={k:v.detach().clone() for k,v in backbone.net.named_parameters()}
            for arm in ['G_FLYVIS','G_PERM']:rollout(video,arm,obs,backbone=backbone)
            import torch
            assert all(torch.equal(v,weights[k]) for k,v in backbone.net.named_parameters())
            # Recover missing components; reuse every already-identical metric calculation.
            done=[];arrays={}
            for arm in p['parameters']:
                metric=FULL/'metrics'/arm/(video+'.json')
                if metric.exists():done.append(arm);continue
                path=raw_path(p,arm,video)
                with np.load(path,allow_pickle=False) as z:array=z['outputs']
                same=None
                for prior in done:
                    if prior not in arrays:
                        with np.load(raw_path(p,prior,video),allow_pickle=False) as z:arrays[prior]=z['outputs']
                    if np.array_equal(array,arrays[prior]):same=prior;break
                arrays[arm]=array
                if same is not None:metric_copy(same,arm,video)
                else:child('eval_'+arm,video,'full_evaluate.py',[video,arm],obs)
                done.append(arm)
            artifacts={}
            for arm in p['parameters']:
                for path in [output_json(arm,video),FULL/'metrics'/arm/(video+'.json'),FULL/'metrics'/arm/(video+'.details.json.gz')]:artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
            for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
            write(completed,dict(passed=True,video=video,seconds=time.perf_counter()-started,artifacts=artifacts,
                protocol_sha256=sha(FULL/'protocol.json'),selection_sha256=p['selection_sha256']),new=True)
            cleanup(video)
            if index==0:
                remaining=(datetime(2026,11,9,7,40,tzinfo=MSK)-datetime.now(MSK)).total_seconds()
                estimate=(time.perf_counter()-started)*2*(len(names)-1)+300
                write(FULL/'first_video_price.json',dict(seconds=time.perf_counter()-started,conservative_remaining_seconds=estimate,
                    available_seconds=remaining,proceed=estimate<remaining),new=True)
                assert estimate<remaining,'Pilot cost exceeds the authorized deadline.'
            status('full_video_complete',video=video,complete=index+1,total=len(names),elapsed_seconds=time.perf_counter()-began,guard=obs.read())
    finally:
        if gpu_lock is not None:gpu_lock.__exit__(None,None,None)
    verify_phase('full_execution_recovery1_phase.json')
    write(FULL/'rollouts_complete.json',dict(passed=True,videos=len(names),arms=list(p['parameters']),
        all_causal=True,baseline_reused=True,core_weights_unchanged=True,seconds=time.perf_counter()-began),new=True)


if __name__=='__main__':main(guard())
