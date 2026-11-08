"""Verify late CPU completion and isolate future stages under independent guardians."""
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main():
    import psutil
    phase=verify_phase('full_execution_recovery1_phase.json')
    guard_path=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/3b163ce6adfd4dad9cba65aa0865cd78/memory_guard.json'
    result=read(guard_path);assert result['reason']=='system_memory_reserve' and result['returncode']!=0
    started=read(FULL/'children/cpu_b1d10d08-743fd86c.started.json')
    for pid,birth in {**result['owned'],str(started['pid']):started['created']}.items():
        try:process=psutil.Process(int(pid));assert abs(process.create_time()-birth)>.01,(pid,'Still alive')
        except psutil.NoSuchProcess:pass
    live=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            argv=process.info['cmdline'] or []
            if any(str(x).replace('\\','/').endswith('/full_cpu.py') for x in argv) and 'b1d10d08-743fd86c' in argv:live.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    assert not live,live
    video='b1d10d08-743fd86c';receipts={}
    for arm in ['G','G_R1']:
        path=FULL/'rollouts'/arm/(video+'.receipt.json');r=read(path);assert r['passed'] and r['frames']==1209 and not r['core_imported']
        for n,h in {**r['artifacts'],**r['feature_files']}.items():assert sha(ROOT/n)==h,n
        receipts[arm]=sha(path)
    processed=FULL/'processed'/(video+'.float32');assert sha(processed)==read(processed.with_suffix('.json'))['sha256']
    write(FULL/'late_cpu_completion_audit.json',dict(passed=True,all_known_owned_processes_ended=True,extra_matching_live_processes=live,
        parent_guard_sha256=sha(guard_path),parent_returncode=result['returncode'],reason_preserved=result['reason'],child_receipts=receipts,
        verified_launch_mistake='The helper child argv had parent-report but no unique audit-memory-worker marker. The supervisor intentionally owns only processes carrying that marker; assuming whole-tree control was incorrect.',
        recovery='Use separately guarded CPU, CUDA and evaluation stages; no nested compute children or persistent CUDA parent. Reuse verified late-completed outputs.'),new=True)
    folder=ROOT/'research_identity_assignment_run_20261108';old=(folder/'full_retina.py').read_text(encoding='utf-8')
    line="        if frame%100==0:status('full_retinal_preparation',video=video,frame=frame,total=shape[0],guard=obs.read())"
    assert old.count(line)==1
    new=old.replace(line,"        if frame%100==0:\n            data.flush()\n            status('full_retinal_preparation',video=video,frame=frame,total=shape[0],guard=obs.read())")
    with (folder/'full_retina_recovery1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(new)
    names=['research_identity_assignment_run_20261108/'+n for n in ['prepare_full_recovery2.py','full_cpu_guarded.py',
        'full_gpu_guarded.py','full_eval_guarded.py','full_manager_recovery2.py','full_retina_recovery1.py']]
    names+=['runs/identity_assignment_20261108/full/late_cpu_completion_audit.json']
    phase['sources'].update({n:sha(ROOT/n) for n in names});phase['registered']=datetime.now(MSK).isoformat()
    phase['recovery2']=dict(parent_resource_failure_preserved=True,scientific_settings_changed=False,
        completed_CPU_arms_reused=True,execution='Light controller runs one independently externally guarded stage at a time. GPU checkpoint reloaded per video; no scientific model change.',
        input_memory_change='Flush processed-frame memmap every100frames. Pixel values and filtering order unchanged; only writeback timing changes.',
        previous_phase_sha256=sha(RUN/'full_execution_recovery1_phase.json'))
    write(RUN/'full_execution_recovery2_phase.json',phase,new=True)


if __name__=='__main__':main()
