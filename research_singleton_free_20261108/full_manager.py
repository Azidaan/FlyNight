"""Sequential first-problem pilot, with component-wise exact result reuse."""
import os,sys,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_singleton_free_20261108.common import *
from research_singleton_free_20261108.full_paths import FULL,source_arm,raw_path,output_path


def launch(name,script,args):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,59,30,tzinfo=MSK)
    folder=FULL/'children';folder.mkdir(parents=True,exist_ok=True);assert not (folder/(name+'.log')).exists()
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/'+script,*args],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time()),new=True);code=proc.wait()
    reports=[v for v in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if v.startswith('SYSTEM_MEMORY_GUARD ')]
    assert reports,(name,code,'No finished guardian receipt')
    report=json.loads(reports[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            process=psutil.Process(int(pid))
            if abs(process.create_time()-birth)<.01:
                try:process.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(name,code,report)


def reuse_metric(source_folder,source_arm_name,arm,video):
    old=read(source_folder/'metrics'/source_arm_name/(video+'.json'));source=source_folder/'metrics'/source_arm_name/(video+'.details.json.gz')
    assert old['passed'] and old['details_sha256']==sha(source)
    assert read(source_folder/'outputs'/source_arm_name/(video+'.json'))==read(output_path(arm,video))
    folder=FULL/'metrics'/arm;folder.mkdir(parents=True,exist_ok=True);details=folder/(video+'.details.json.gz');shutil.copyfile(source,details)
    result=dict(old,arm=arm,output_sha256=sha(output_path(arm,video)),reused_metric_source=str((source_folder/'metrics'/source_arm_name/(video+'.json')).relative_to(ROOT)),
        identical_output_arrays_verified=True)
    write(folder/(video+'.json'),result,new=True)


def main():
    # Do not dispatch a heavy worker until the parent has attached this tree
    # to the independent14:00MSK shutdown watch.
    registered=FULL/'deadline_registered.json';began_registration=time.monotonic()
    while not registered.exists() and time.monotonic()-began_registration<10:time.sleep(.1)
    assert registered.exists(),'Author-window registration did not finish; no heavy child launched.'
    phase('full_phase.json');p=read(FULL/'protocol.json');assert read(RUN/'independent_audit.json')['passed']
    if not (FULL/'technical.json').exists():launch('technical','full_technical.py',[])
    assert read(FULL/'technical.json')['passed'];first=next(iter(p['videos']))
    if not (FULL/'zero_parity.json').exists():
        launch('zero_'+first,'full_worker.py',['zero',first]);r=read(FULL/'rollouts/ZERO'/(first+'.receipt.json'));assert r['passed'] and r['zero']
        write(FULL/'zero_parity.json',dict(passed=True,video=first,raw_outputs_exact=True,annotation_output_exact=True,receipt_sha256=sha(FULL/'rollouts/ZERO'/(first+'.receipt.json'))),new=True)
    old=BASE/'full_precision';oldp=read(old/'protocol.json');began=time.perf_counter()
    for index,(video,entry) in enumerate(p['videos'].items()):
        completed=FULL/'completed'/(video+'.json')
        if completed.exists():
            for name,h in read(completed)['artifacts'].items():assert sha(ROOT/name)==h
            continue
        for role,arms in [('cpu',p['cpu_arms']),('gpu',p['gpu_arms'])]:
            if any(not (FULL/'rollouts'/arm/(video+'.receipt.json')).exists() for arm in arms):launch(role+'_'+video,'full_worker.py',[role,video])
        for arm in p['cpu_arms']+p['gpu_arms']:
            r=read(FULL/'rollouts'/arm/(video+'.receipt.json'));assert r['passed']
            for name,h in {**r['artifacts'],**r['feature_files']}.items():assert sha(ROOT/name)==h
        for arm in p['aliases']:
            if not output_path(arm,video).exists():
                source=source_arm(p,arm);dest=output_path(arm,video);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(output_path(source,video),dest)
        done=['B0'];arrays={}
        for arm in p['parameters']:
            metric=FULL/'metrics'/arm/(video+'.json')
            if metric.exists():
                if arm not in done:done.append(arm)
                continue
            with np.load(raw_path(p,arm,video),allow_pickle=False) as z:array=z['outputs']
            same=None
            for prior in done:
                if prior not in arrays:
                    with np.load(raw_path(p,prior,video),allow_pickle=False) as z:arrays[prior]=z['outputs']
                if np.array_equal(array,arrays[prior]):same=prior;break
            arrays[arm]=array
            if same is not None:reuse_metric(FULL,same,arm,video)
            else:
                previous=None
                for prior in oldp['parameters']:
                    source=oldp['aliases'].get(prior,prior)
                    oldraw=ROOT/entry['raw_array'] if source=='B0' else old/'rollouts'/source/(video+'.npz')
                    with np.load(oldraw,allow_pickle=False) as z:oldarray=z['outputs']
                    if np.array_equal(array,oldarray) and read(old/'outputs'/prior/(video+'.json'))==read(output_path(arm,video)):
                        previous=prior;break
                if previous is not None:reuse_metric(old,previous,arm,video)
                else:launch('eval_'+arm+'_'+video,'full_evaluate.py',[video,arm])
            done.append(arm)
        artifacts={}
        for arm in p['parameters']:
            for path in [output_path(arm,video),FULL/'metrics'/arm/(video+'.json'),FULL/'metrics'/arm/(video+'.details.json.gz')]:artifacts[path.relative_to(ROOT).as_posix()]=sha(path)
        for name,h in entry['sources'].items():assert sha(ROOT/name)==h
        phase('full_phase.json');write(completed,dict(passed=True,video=video,artifacts=artifacts,protocol_sha256=sha(FULL/'protocol.json')),new=True)
        scratch=FULL/'processed'/(video+'.float64')
        if scratch.exists():
            assert scratch.resolve().parent==(FULL/'processed').resolve();meta=read(scratch.with_suffix('.json'));assert sha(scratch)==meta['sha256'];scratch.unlink()
            write(FULL/'processed'/(video+'.released.json'),dict(released=True,sha256=meta['sha256'],bytes=meta['bytes'],reason='Disposable decoded input; source movie, resume states, all used retinal views and feature maps retained.'),new=True)
        status('full_first_problem_video_complete',video=video,complete=index+1,total=len(p['videos']),elapsed_seconds=time.perf_counter()-began)
    write(FULL/'rollouts_complete.json',dict(passed=True,videos=len(p['videos']),arms=list(p['parameters']),seconds=time.perf_counter()-began),new=True)
    launch('additional','full_additional.py',[]);launch('statistics','full_statistics.py',[])
    phase('full_phase.json');write(FULL/'analyses_complete.json',dict(passed=True,comparison_sha256=sha(FULL/'comparison_report.json'),additional_sha256=sha(FULL/'additional_report.json')),new=True)
    status('full_first_problem_analysis_complete',report=str(FULL/'comparison_report.json'))


if __name__=='__main__':main()
