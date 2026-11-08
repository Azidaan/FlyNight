"""Fixed descriptive statistics; no operating threshold or deployment selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
def register():
    paths=[Path(__file__),OUT/'results.json',OUT/'comparison_protocol.json',OUT/'comparison_read_set.json',OUT/'external_inventory.json',ROOT/'runs/clipped_anchor_readout_20261109/summary.json',ROOT/'runs/clipped_anchor_readout_20261109/strong_input_control.json',ROOT/'runs/bbox_normalized_signals_20261108/full/comparison_report.json']
    write('analysis_protocol.json',dict(seed=42,replicates=5000,metric='Pooled own-vs-foreign AUROC with half ties; higher is better. No per-event accept/reject threshold. Separate six other-video new-reader cases from five ready clipped development cases; never pool them.',bootstrap='Paired whole-video resampling with replacement. Report number of resamples with both classes and discarded absent-class draws. With one foreign case or one own case, narrow empirical intervals do NOT establish generalization.',criteria='Apply original three criteria. Criterion2 requires strict gain on new six-case other-video group, not on reused development cases. The absent independent clipped group is a coverage limitation. Criterion3 is descriptive point superiority over both matched34controls, with uncertainty explicitly reported.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
def auc(rows,name,arm,field='maximum'):
    own=[r['variants'][name][arm][field] for r in rows if r['kind']=='own'];foreign=[r['variants'][name][arm][field] for r in rows if r['kind']=='foreign']
    if not own or not foreign or any(v is None for v in own+foreign):return None
    return float(np.mean([[float(a>b)+.5*float(a==b) for b in foreign] for a in own]))
def run():
    p=read(OUT/'analysis_protocol.json');verify(p);verify(read(OUT/'comparison_read_set.json'));d=read(OUT/'results.json');summary={};rng=np.random.default_rng(42);draws={}
    for group,rows in [('other_video_no_clip',d['new_records']),('development_ready_clip',d['historical_clipped_records'])]:
        videos=sorted({r['video'] for r in rows});names=list(rows[0]['variants']);draw=rng.integers(0,len(videos),(5000,len(videos)));draws[group]=draw;reps=[];valid=0
        for sample in draw:
            chosen=[r for i in sample for r in rows if r['video']==videos[int(i)]]
            if not any(r['kind']=='own' for r in chosen) or not any(r['kind']=='foreign' for r in chosen):continue
            reps.append({n:{a:auc(chosen,n,a) for a in ['old','normalization','edge','joint']} for n in names});valid+=1
        table={n:{a:auc(rows,n,a) for a in ['old','normalization','edge','joint']} for n in names};fixed={n:{a:auc(rows,n,a,'fixed') for a in ['old','normalization','edge','joint']} for n in names};comparisons={}
        for label,n1,a1,n0,a0 in [('C34_joint-old','C34','joint','C34','old'),('C8_joint-old','FlyVis','joint','FlyVis','old'),('C34_joint-input_temporal','C34','joint','R134_temporal','joint'),('C34_joint-input_spatial','C34','joint','R134_spatial','joint'),('C34_joint-input_temporal_original','C34','joint','R134_temporal','old')]:
            delta=table[n1][a1]-table[n0][a0];boot=[r[n1][a1]-r[n0][a0] for r in reps];comparisons[label]=dict(delta=delta,ci95=[float(x) for x in np.percentile(boot,[2.5,97.5])],relative_percent=100*delta/table[n0][a0] if table[n0][a0] else None)
        own=[r for r in rows if r['kind']=='own'];foreign=[r for r in rows if r['kind']=='foreign'];rankings=[]
        for a in own:
            for b in foreign:
                rankings.append(dict(own=[a['video'],a['frame'],a['old'],a['new']],foreign=[b['video'],b['frame'],b['old'],b['new']],same_frame=a['video']==b['video'] and a['frame']==b['frame'],same_candidate=a['video']==b['video'] and a['frame']==b['frame'] and a['new']==b['new'],scores={n:{arm:dict(own=a['variants'][n][arm]['maximum'],foreign=b['variants'][n][arm]['maximum'],margin=a['variants'][n][arm]['maximum']-b['variants'][n][arm]['maximum']) for arm in ['old','normalization','edge','joint']} for n in names}))
        summary[group]=dict(events=len(rows),own=len(own),foreign=len(foreign),videos=len(videos),auroc=table,fixed_auroc=fixed,comparisons=comparisons,rankings=rankings,bootstrap_valid=valid,bootstrap_absent_class=5000-valid,bootstrap_limitation='One foreign case in new group; one own case in reused clip group. Degenerate empirical intervals reflect observed support, not established certainty for new identities.')
    assert summary['other_video_no_clip']['auroc']['C34']['old']==1 and summary['other_video_no_clip']['auroc']['C34']['joint']==1
    write('summary.json',dict(groups=summary,independent_clip_group=None,independent_clip_reason='No both-direct clipped singleton found in517 saved max_age6 singleton events on46other videos:506 lacked annotation of both exact frames,9were not uniquely matched,2direct own nonclip remain inventory-only.',acceptance=[True,False,summary['other_video_no_clip']['auroc']['C34']['joint']>max(summary['other_video_no_clip']['auroc'][n]['joint'] for n in ['R134_temporal','R134_spatial'])],state='completed_negative',interpretation='Strict new-case gain criterion failed at the observed C34 baseline ceiling. Matched control advantage is a sparse point observation. Clipping improvement is reused development evidence, not new validation. Do not deploy or claim real-ID gain.'))
    with (OUT/'bootstrap_video_draws.npz').open('xb') as f:np.savez_compressed(f,**draws)
    verify(p);verify(read(OUT/'comparison_read_set.json'));print(json.dumps({k:dict(auroc=v['auroc'],comparisons=v['comparisons'],valid=v['bootstrap_valid']) for k,v in summary.items()},ensure_ascii=False),flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
