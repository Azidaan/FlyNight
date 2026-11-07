"""Строгий полный аудит20, с отдельным сохранением завершённых видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import service,RUN
read,write,sha,now=service.read,service.write,service.sha,service.now

def audit_source():
    import scripts.audit_b1_tracker_20261028 as original
    code=Path(original.__file__).read_text(encoding='utf-8')
    old="        v = source_row['video']; expected = read(ROOT / source_row['baseline'])"
    new="""        v = source_row['video']
        fingerprint=input_manifest(ROOT,source,v,variants)
        checkpoint=folder/'completed'/(v+'.json')
        if checkpoint.exists():
            cached=validate_checkpoint(read(checkpoint),fingerprint)
            all_metrics[v]=cached['metrics'];checked+=cached['events_checked']
            write(folder/'status.json',dict(state='running',updated=now(),video=v,stage='Готовый независимый аудит сохранён без повторения',events_checked=checked),required=False)
            continue
        checked_before=checked
        expected = read(ROOT / source_row['baseline'])"""
    assert code.count(old)==1;code=code.replace(old,new)
    old="        assert read(source / 'appearance' / (v + '.json')) == read(source / 'zero' / (v + '.json'))"
    new=old+"\n        write(checkpoint,dict(passed=True,video=v,input_manifest=fingerprint,metrics=all_metrics[v],events_checked=checked-checked_before,strict_cpu_replay=True,updated=now()))"
    assert code.count(old)==1;code=code.replace(old,new)
    return code

def main():
    import numpy as np,hashlib
    from tracking.b1_audit_checkpoints import input_manifest,validate_checkpoint
    source=RUN/'b1_tracker';folder=RUN/'b1_tracker_audit'
    p=read(folder/'protocol.json')
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    with np.load(source/'geometry.npz') as a,np.load(source/'geometry_seed.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in ['points','eye_size'])
    seed=read(source/'resume_seed.json');v=seed['completed_videos'][0]
    proof_path=ROOT/'runs/recovery_research_20261027/b1_tracker/recovery2/independent_audit.json'
    proof=read(proof_path);assert proof['passed'] and proof['videos']==1
    # Первое видео целиком и независимо проверено ранее, его21вход точно тотже.
    manifest=input_manifest(ROOT,source,v,['baseline','appearance','zero','C','D','R1','B'])
    for rel,digest in manifest.items():
        path=ROOT/rel
        if path.parent.name in ['completed'] or path.name in ['protocol.json','geometry.npz']:continue
        assert seed['output_sha256'][rel]==digest
    checkpoint=folder/'completed'/(v+'.json')
    if not checkpoint.exists():
        write(checkpoint,dict(passed=True,video=v,input_manifest=manifest,metrics={a:seed['per_video'][a][v] for a in ['baseline','appearance','zero','C','D','R1','B']},
            events_checked=proof['events_checked'],strict_cpu_replay=True,reused_completed_audit=True,old_audit_sha256=sha(proof_path),updated=now()))
    import scripts.audit_b1_tracker_20261028 as original
    code=audit_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_audit_sha256']
    ns={**vars(original),'__name__':'b1_checkpointed_audit','input_manifest':input_manifest,'validate_checkpoint':validate_checkpoint}
    exec(compile(code,str(Path(original.__file__)),'exec'),ns);ns['run']()

if __name__=='__main__':main()
