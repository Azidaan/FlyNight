"""Дешёвая калибровка тех же координат по train/dev, без изображений и новых нейросетей."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now
RUN=BASE/'pixel_box_numeric_control'

def main():
    import numpy as np
    from tracking.pixel_box_refinement import metadata,coordinates,apply_offsets
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');assert read(BASE/'pixel_box_validation/independent_audit.json')['passed'];end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();start=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Numeric control original budget')
        write(RUN/'status.json',dict(state='running',updated=now(),**kw),required=False)
    xs=[];ys=[]
    for item in read(BASE/'pixel_box_dataset/report.json')['videos']:
        if item['split']!='train':continue
        z=np.load(BASE/'pixel_box_dataset/train'/item['video']/'targets.npz');xs.append(z['M'][z['known']]);ys.append(z['Y'][z['known']])
    x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(np.float64);mean=x.mean(0);std=np.maximum(x.std(0),1e-4)
    xn=np.clip((x-mean)/std,-10,10);design=np.c_[np.ones(len(xn)),xn];reg=np.eye(design.shape[1]);reg[0,0]=0
    beta=np.linalg.solve(design.T@design+reg,design.T@y)
    medians=np.zeros((8,4));counts=[]
    for cls in range(8):
        take=x[:,4+cls]>.5;counts.append(int(take.sum()))
        if take.any():medians[cls]=np.median(y[take],axis=0)
    np.savez(RUN/'calibration.npz',mean=mean,std=std,beta=beta,class_medians=medians,class_counts=np.array(counts))
    def predict(m,method):
        if method=='class_median':return medians[np.argmax(m[:,4:],axis=1)].astype(np.float32)
        return np.clip(np.c_[np.ones(len(m)),np.clip((m-mean)/std,-10,10)]@beta,-.25,.25).astype(np.float32)
    perdev={'baseline':{}};variants={f'{method}_a{alpha}':dict(method=method,alpha=alpha) for method in ['class_median','ridge_meta'] for alpha in [0.,.5,1.]}
    perdev.update({name:{} for name in variants});devsrc=read(BASE/'pixel_box_dev/protocol.json')['sources']
    for i,s in enumerate(devsrc):
        video_start=time.monotonic();v=s['video'];base=read(ROOT/s['baseline']);records=read(BASE/'pixel_box_dev/offsets'/(v+'_records.json'));lookup=[{str(l['id']):l for l in row['labels']} for row in base]
        m=np.array([metadata(lookup[r['reference']][str(r['id'])],r['past_frame'] is not None,0 if r['past_frame'] is None else r['frame']-r['past_frame']) for r in records]).reshape(-1,12)
        perdev['baseline'][v]=read(BASE/'pixel_box_dev/report.json')['per_video']['baseline'][v]
        for name,spec in variants.items():
            status(phase='dev',video=v,index=i,variant=name);out=copy.deepcopy(base)
            if spec['alpha']:
                target=[{str(l['id']):l for l in row['labels']} for row in out]
                for r,delta in zip(records,predict(m,spec['method'])):
                    if np.any(delta):target[r['reference']][str(r['id'])]['box2d']={k:float(value) for k,value in zip(['x1','y1','x2','y2'],apply_offsets(r['box'],delta,spec['alpha']))}
            write(RUN/'dev'/name/(v+'.json'),out);perdev[name][v]=perdev['baseline'][v] if not spec['alpha'] else evaluate_hota(RUN/'dev'/name,[v],split='train')
        if i==0:
            seconds=time.monotonic()-video_start;cost=2*seconds*(len(devsrc)-1+len(p['sources']))+120;available=end-time.time()
            write(RUN/'price.json',dict(first_full_dev_video_seconds=seconds,conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    scores={name:float(np.mean([m['HOTA'] for m in values.values()])) for name,values in perdev.items()};chosen=max(variants,key=lambda n:(scores[n],-variants[n]['alpha'],n=='class_median_a0.0'))
    frozen=dict(name=chosen,**variants[chosen],dev_mean_hota=scores[chosen],calibration_sha256=sha(RUN/'calibration.npz'),selected_on='dev only',created=now())
    write(RUN/'frozen_before_validation.json',frozen);write(RUN/'dev_report.json',dict(per_video=perdev,mean_per_video=scores,chosen=frozen))
    original=read(BASE/'pixel_box_validation/report.json');per={'baseline':{},'numeric_control':{}};timing={};proof={}
    for i,s in enumerate(p['sources']):
        began=time.monotonic();v=s['video'];base=read(ROOT/s['baseline']);records=read(BASE/'pixel_box_validation/offsets'/(v+'_records.json'));lookup=[{str(l['id']):l for l in row['labels']} for row in base]
        m=np.array([metadata(lookup[r['reference']][str(r['id'])],r['past_frame'] is not None,0 if r['past_frame'] is None else r['frame']-r['past_frame']) for r in records]).reshape(-1,12)
        d=predict(m,frozen['method']);np.save(RUN/(v+'_offsets.npy'),d);out=copy.deepcopy(base)
        if frozen['alpha']:
            target=[{str(l['id']):l for l in row['labels']} for row in out]
            for r,delta in zip(records,d):
                if np.any(delta):target[r['reference']][str(r['id'])]['box2d']={k:float(value) for k,value in zip(['x1','y1','x2','y2'],apply_offsets(r['box'],delta,frozen['alpha']))}
        for a,b in zip(out,base):assert len(a['labels'])==len(b['labels']) and all({k:v for k,v in aa.items() if k!='box2d'}=={k:v for k,v in bb.items() if k!='box2d'} for aa,bb in zip(a['labels'],b['labels']))
        status(phase='val',video=v,index=i,total=len(p['sources']));write(RUN/'numeric_control'/(v+'.json'),out);write(RUN/'baseline'/(v+'.json'),base)
        per['baseline'][v]=s['baseline_metrics'];per['numeric_control'][v]=per['baseline'][v] if not frozen['alpha'] else evaluate_hota(RUN/'numeric_control',[v],split='val')
        timing[v]=time.monotonic()-began;proof[v]=dict(baseline_exact=True,only_boxes_changed=True,offsets_sha256=sha(RUN/(v+'_offsets.npy')));write(RUN/'per_video.json',per)
    groups=dict(p['groups']);groups['all40']=[s['video'] for s in p['sources']];contrasts={}
    for group,vs in groups.items():
        contrasts[group]={'numeric_vs_baseline':paired({v:per['numeric_control'][v]['HOTA'] for v in vs},{v:per['baseline'][v]['HOTA'] for v in vs})}
        for model,values in original['per_video'].items():
            if model=='baseline':continue
            contrasts[group][model+'_vs_numeric']=paired({v:values[v]['HOTA'] for v in vs},{v:per['numeric_control'][v]['HOTA'] for v in vs})
    pooled={name:evaluate_hota(RUN/name,groups['all40'],split='val') for name in per}
    write(RUN/'report.json',dict(complete=True,frozen=frozen,per_video=per,pooled=pooled,paired=contrasts,timing=timing,proof=proof,images_used=False,ground_truth_used_at_inference=False,
        val_used_for_selection=False,seconds=time.monotonic()-start,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':main()
