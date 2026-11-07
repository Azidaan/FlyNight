"""Independently score saved candidate vectors and compare video-level outcomes."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('runs/flyvis_atlas_core_20261107')
PRE=Path('runs/flyvis_exclusive_t4t5_20261107')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def audit():
    read=lambda p:json.loads(p.read_bytes())
    p=read(BASE/'protocol.json');finished=read(BASE/'inference_completed.json')
    assert finished['protocol_sha256']==digest(BASE/'protocol.json') and len(finished['raw_sha256'])==len(p['cases'])==172
    with gzip.open(PRE/'choices.csv.gz','rt') as f:
        truth={int(r['uid']):(r['video'],int(r['target_side'])) for r in csv.DictReader(f)
               if r['split']=='evaluate26' and r['exactly2']=='1' and r['label_reason']=='ok' and r['exclusive_choice']}
    assert {c['uid'] for c in p['cases']}==set(truth)
    sets=['native','identity',*[scale+'/'+t for scale in ['0.90','1.10'] for t in p['types']]]
    results=[]; margin=[];clips=[]; identities=[]; stored=[]
    for c in p['cases']:
        uid=c['uid'];path=BASE/'raw'/f'{uid}.npz'
        assert digest(path)==finished['raw_sha256'][path.name]
        v,side=truth[uid];assert v==c['video']
        with np.load(path,allow_pickle=False) as z:
            assert int(z['uid'])==uid and z['means'].shape==(len(sets),2,34)
            # Evaluate in channel order using signed sums, independently of worker matrix multiplication.
            diff=np.sum((z['means'][:,0,:]-z['means'][:,1,:])*z['weights'][None,:],axis=1)*side
            ref=np.sum((z['reference'][0]-z['reference'][1])*z['weights'])*side
            identities.append(float(z['identity_map_max']));stored.append(float(z['archive_map_max']))
        vote=np.where(np.abs(diff)<=1e-10,.5,(diff>0).astype(float))
        assert np.isfinite(diff).all() and abs(diff[0]-ref)<=2*p['gates']['archive_score_max_abs'] and vote[0]==vote[1]
        results.append(vote);margin.append(diff);clips.append(v)
    results=np.array(results);margin=np.array(margin)
    videos=sorted(set(clips));clips=np.array(clips)
    by_video=np.stack([(results[clips==v]-results[clips==v,:1]).sum(0) for v in videos]);counts=np.array([(clips==v).sum() for v in videos])
    draw=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),size=20000)
    bootstrap=100*(draw@by_video)/(draw@counts)[:,None]
    report=[]
    for i,name in enumerate(sets):
        delta=margin[:,i]-margin[:,0];change=results[:,i]-results[:,0]
        report.append(dict(name=name,correct=float(results[:,i].sum()),choice_changes=int((change!=0).sum()),
                           gains=int((change>0).sum()),losses=int((change<0).sum()),ties=int((results[:,i]==.5).sum()),
                           paired_delta_pp=float(change.mean()*100),median_abs_margin_change=float(np.median(np.abs(delta))),
                           max_abs_margin_change=float(np.max(np.abs(delta))),
                           ci95_pp=np.quantile(bootstrap[:,i],[.025,.975]).tolist(),
                           ci_family65_pp=np.quantile(bootstrap[:,i],[.025/65,1-.025/65]).tolist()))
    ans=dict(passed=True,units=len(p['types']),pairs=len(results),videos=len(videos),intervention='penultimate recurrent state pulse; read after one real frame',
             archive_max_abs_map=max(stored),identity_max_abs_map=max(identities),conditions=report,
             independent_evaluator_sha256=digest(Path(__file__)),raw_manifest_sha256=digest(BASE/'inference_completed.json'))
    with (BASE/'independent_audit.json').open('x',encoding='utf-8') as f:json.dump(ans,f,indent=2,allow_nan=False)
    print('Audited',ans['units'],ans['pairs'],ans['videos'],'baseline',report[0]['correct'])


if __name__=='__main__':audit()
