"""Сопоставимые вмешательства в сохранённый выход; это диагностика, не модель."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import read,write,sha
RUN=ROOT/'runs/recovery_research_20261027/error_budget'

def interventions(base,truth,threshold=.5):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from tracking.recovery_detection_support import box,iou
    from evaluation.hota_eval import VALID
    gt=[g for g in truth if g['category'] in VALID and not g.get('attributes',{}).get('crowd',False)]
    scores=np.array([[iou(box(b),box(g)) if b['category']==g['category'] else 0. for g in gt] for b in base]).reshape(len(base),len(gt))
    pairs={}
    if base and gt:
        rows,cols=linear_sum_assignment(-scores)
        pairs={int(r):int(c) for r,c in zip(rows,cols) if scores[r,c]>=threshold}
    identified=copy.deepcopy(base);geometry=copy.deepcopy(base)
    for j,b in enumerate(identified):b['id']='G_'+str(gt[pairs[j]]['category'])+'_'+str(gt[pairs[j]]['id']) if j in pairs else 'P_'+str(b['id'])
    for j,k in pairs.items():geometry[j]['box2d']=copy.deepcopy(gt[k]['box2d'])
    both=copy.deepcopy(identified)
    for j,k in pairs.items():both[j]['box2d']=copy.deepcopy(gt[k]['box2d'])
    missing=copy.deepcopy(identified)
    for k,g in enumerate(gt):
        if k not in pairs.values():missing.append(dict(id='G_'+g['category']+'_'+str(g['id']),category=g['category'],score=1.,box2d=copy.deepcopy(g['box2d'])))
    return dict(identity=identified,geometry=geometry,remove_unmatched=[copy.deepcopy(b) for j,b in enumerate(base) if j in pairs],identity_geometry=both,identity_missing=missing)

def run():
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic()
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    arms=p['arms'];per={a:{} for a in arms};per['baseline']={};timing={};proof={}
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Error budget stage deadline')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    for i,item in enumerate(p['sources']):
        start=time.monotonic();v=item['video'];baseline=read(ROOT/item['baseline']);truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json')
        assert len(baseline)==len(truth);outputs={a:[] for a in arms}
        for b,g in zip(baseline,truth):
            assert b['name']==g['name'];variants=interventions(b['labels'],g['labels'],p['matching_iou'])
            for arm,labels in variants.items():
                row=copy.deepcopy(b);row['labels']=labels
                assert len({x['id'] for x in labels})==len(labels)
                if arm=='identity':
                    assert len(labels)==len(b['labels'])
                    assert all({k:v for k,v in x.items() if k!='id'}=={k:v for k,v in y.items() if k!='id'} for x,y in zip(labels,b['labels']))
                if arm=='geometry':
                    assert len(labels)==len(b['labels'])
                    assert all({k:v for k,v in x.items() if k!='box2d'}=={k:v for k,v in y.items() if k!='box2d'} for x,y in zip(labels,b['labels']))
                outputs[arm].append(row)
        # Базу не прогоняем через детектор/трекер/оценщик повторно: её индивидуальные метрики сохранены.
        per['baseline'][v]=item['baseline_metrics'];proof[v]=dict(baseline_sha256=sha(ROOT/item['baseline']),rows=len(baseline),identity_changes_only_ids=True,geometry_changes_only_boxes=True)
        for arm,rows in outputs.items():
            status(video=v,index=i,total=len(p['sources']),arm=arm)
            write(RUN/arm/(v+'.json'),rows);per[arm][v]=evaluate_hota(RUN/arm,[v],split='val')
        timing[v]=time.monotonic()-start;write(RUN/'per_video.json',per)
        if i==0:
            cost=2*timing[v]*(len(p['sources'])-1)+120;available=end-time.time()
            write(RUN/'price.json',dict(first_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    comparisons={}
    for group,vids in p['groups'].items():
        comparisons[group]={arm:paired({v:per[arm][v]['HOTA'] for v in vids},{v:per['baseline'][v]['HOTA'] for v in vids}) for arm in arms}
        comparisons[group]['missing_given_identity']=paired({v:per['identity_missing'][v]['HOTA'] for v in vids},{v:per['identity'][v]['HOTA'] for v in vids})
        comparisons[group]['geometry_given_identity']=paired({v:per['identity_geometry'][v]['HOTA'] for v in vids},{v:per['identity'][v]['HOTA'] for v in vids})
    write(RUN/'report.json',dict(complete=True,diagnostic_only=True,maximum_or_upper_bound=False,real_model_improvement=False,
        per_video=per,paired=comparisons,timing=timing,proof=proof,seconds=time.monotonic()-started,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
