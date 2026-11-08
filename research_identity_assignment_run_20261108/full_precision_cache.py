"""Reuse an old feature only after every retinal input in its causal sequence is exact."""
import gzip,shutil
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_precision_io import FULL,Receptors
from research_identity_assignment_run_20261108.export_states import sequence_key

OLD=RUN/'full'


def recover(video):
    marker=FULL/'cache_reuse'/(video+'.json')
    if marker.exists():return read(marker)
    sequences={};files={};per_arm={}
    for arm in ['G_R1','G_FLYVIS','G_PERM']:
        receipt=OLD/'rollouts'/arm/(video+'.receipt.json')
        if not receipt.exists():continue
        r=read(receipt);assert r['passed'];per_arm[arm]=r
        for n,h in {**r['artifacts'],**r['feature_files']}.items():assert sha(ROOT/n)==h
        files.update(r['feature_files'])
        event_file=OLD/'rollouts'/arm/(video+'.events.json.gz')
        with gzip.open(event_file,'rt',encoding='utf-8') as f:events=json.load(f)
        for event in events:
            rows=event['rows'];plan=event['plan']
            for j,col in plan['columns'].items():
                if not col['geometry_valid']:continue
                seeds=[i for i,jj in plan['votes'] if jj==int(j)]
                observer=min(seeds,key=lambda i:(float(np.trace(np.array(rows[i]['covariance'])[:2,:2])),rows[i]['start_frame'],rows[i]['tid']))
                current=rows[observer];seq=[[h['frame'],h['crop']] for h in current['history']]+current['search']
                sequences[sequence_key(video,seq)]=seq
                for i in col['rows']:
                    seq=[[h['frame'],h['crop']] for h in rows[i]['history']];key=sequence_key(video,seq)
                    if key in sequences:assert sequences[key]==seq
                    sequences[key]=seq
    renderer=Receptors(video);views={};differences={}
    for seq in sequences.values():
        for frame,crop in seq:
            key=hashlib.sha256(json.dumps([frame,crop],separators=(',',':')).encode()).hexdigest()[:32]
            if key in views:continue
            old=OLD/'retinal'/video/(key+'.npy');assert old.exists()
            a=np.load(old,allow_pickle=False);b=renderer.get(frame,crop,renderer.shape[0]-1)
            exact=bool(np.array_equal(a,b));views[key]=exact
            if not exact:differences[key]=dict(frame=frame,crop=crop,max_error=float(np.max(np.abs(a.astype(float)-b.astype(float)))),different_cells=int(np.count_nonzero(a!=b)))
    reused={};recomputed=[]
    for name,h in files.items():
        old=ROOT/name;key,kind,_=old.name.split('.');assert key in sequences,(name,'Sequence provenance missing')
        seq=sequences[key]
        keys=[hashlib.sha256(json.dumps([frame,crop],separators=(',',':')).encode()).hexdigest()[:32] for frame,crop in seq]
        if all(views[k] for k in keys):
            dest=FULL/'features'/video/old.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(old,dest)
            assert sha(dest)==h;reused[dest.relative_to(ROOT).as_posix()]=dict(sha256=h,sequence=seq,source=name)
        else:recomputed.append(dict(key=key,kind=kind,sequence=seq,source=name))
    report=dict(passed=True,video=video,old_features=len(files),old_views=len(views),exact_views=sum(views.values()),
        changed_views=differences,features_reused=reused,features_needing_new_input=recomputed,
        cause='Old full pilot stored float32 before resizing; original Retina/BoxEye path resizes native float64 from frame1 onward.',
        new_core_calls=0,policy='Only features with exact full causal input sequences are reused. No old mismatched feature is copied.')
    write(marker,report,new=True);return report
