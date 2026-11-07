"""Причинное уточнение только рамок: выбор силы на dev и фиксированная проверка val."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now

def run(folder):
    import cv2,numpy as np,torch
    from tracking.pixel_box_refinement import coordinates,region,metadata,make_model,apply_offsets
    from tracking.gpu_lock import GpuLock
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    run=BASE/folder;p=read(run/'protocol.json');phase=p['phase'];is_dev=phase=='dev';split='train' if is_dev else 'val'
    previous=read(BASE/p['depends_on']/'controller_result.json');assert previous['exit_code']==0 and previous['all_owned_ended']
    if not is_dev:assert read(BASE/'pixel_box_dev_audit/report.json')['passed']
    for path,h in p['sha256'].items():assert sha(ROOT/path)==h,path
    end=datetime.fromisoformat(read(run/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic();torch.set_num_threads(2)
    def status(**kw):
        if time.time()>end-5 or (run/'stop.request').exists():raise TimeoutError('Pixel geometry evaluation budget')
        write(run/'status.json',dict(state='running',updated=now(),**kw),required=False)
    learning=read(BASE/'pixel_box_learning/report.json');assert learning['complete'];models={};specs={};variants={};per={'baseline':{}};timing={};proof={}
    selected=None if is_dev else read(BASE/'pixel_box_frozen_for_validation.json')
    with GpuLock('Local\\DissertationSingleGpu'):
        for spec in learning['models']:
            path=BASE/spec['checkpoint'];assert sha(path)==spec['sha256'];model=make_model().to('cuda');saved=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(saved['state_dict'],strict=True);model.eval();models[spec['name']]=model;specs[spec['name']]=spec
            for alpha in ([0.,.5,1.] if is_dev else [next(x['alpha'] for x in selected['models'] if x['name']==spec['name'])]):
                name=spec['name']+'_a'+str(alpha);variants[name]=dict(model=spec['name'],alpha=alpha);per[name]={}
        for index,source in enumerate(p['sources']):
            began=time.monotonic();v=source['video'];base=read(ROOT/source['baseline']);align=read(ROOT/source['alignment'])['matches'];assert len(base)==len(align)
            anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};assert len(anchors)==len(align)
            cap=cv2.VideoCapture(str(ROOT/source['video_path']));assert cap.isOpened();images=[];metas=[];records=[];past={}
            for frame in range(max(anchors)+1):
                ok,image=cap.read();assert ok
                if frame not in anchors:continue
                ref=anchors[frame];labels=base[ref]['labels'];patches={l['id']:region(image,coordinates(l),96,1.5) for l in labels}
                for label in labels:
                    old=past.get(label['id']);cur=patches[label['id']];images.append(np.stack([cur if old is None else old[1],cur]))
                    metas.append(metadata(label,old is not None,0 if old is None else frame-old[0]));records.append(dict(reference=ref,id=label['id'],frame=frame,past_frame=None if old is None else old[0],box=coordinates(label).tolist()))
                past={l['id']:(frame,patches[l['id']]) for l in labels}
                if ref%25==0:status(stage='Реальные области до текущего запроса',video=v,index=index,total=len(p['sources']),reference=ref)
            cap.release();n=len(records);a=np.asarray(images,np.uint8).reshape(n,2,96,96,3);m=np.asarray(metas,np.float32).reshape(n,12)
            offsets={name:[] for name in models}
            with torch.inference_mode():
                for start in range(0,n,128):
                    status(stage='Поправки по фиксированным моделям',video=v,index=index,total=len(p['sources']),boxes_done=start,boxes_total=n)
                    x=torch.from_numpy(a[start:start+128]).to('cuda',dtype=torch.float32).permute(0,1,4,2,3);x=(x/255-.5)/.25
                    meta=torch.from_numpy(m[start:start+128]).to('cuda')
                    for name,model in models.items():
                        delta=model(x,meta,specs[name]['arm']=='temporal');assert torch.isfinite(delta).all() and delta.abs().max()<=.250001;offsets[name].append(delta.cpu().numpy())
            offsets={name:np.concatenate(d) if d else np.zeros((0,4),np.float32) for name,d in offsets.items()}
            features=run/'offsets';features.mkdir(exist_ok=True);np.savez_compressed(features/(v+'.npz'),**offsets);write(features/(v+'_records.json'),records)
            picks=np.unique(np.array([0,n//2,n-1])) if n else np.array([],int);np.savez_compressed(features/(v+'_audit_samples.npz'),images=a[picks],M=m[picks],indices=picks)
            write(run/'baseline'/(v+'.json'),base)
            per['baseline'][v]=evaluate_hota(run/'baseline',[v],split=split) if is_dev else source['baseline_metrics']
            for name,var in variants.items():
                status(stage='Полная HOTA поправки',video=v,index=index,variant=name);out=copy.deepcopy(base)
                if var['alpha']!=0:
                    lookup=[{str(l['id']):l for l in row['labels']} for row in out]
                    for r,delta in zip(records,offsets[var['model']]):
                        if not np.any(delta):continue
                        label=lookup[r['reference']][str(r['id'])];b=apply_offsets(r['box'],delta,var['alpha'])
                        label['box2d']={k:float(x) for k,x in zip(['x1','y1','x2','y2'],b)}
                for arow,brow in zip(out,base):
                    assert {k:x for k,x in arow.items() if k!='labels'}=={k:x for k,x in brow.items() if k!='labels'}
                    assert len(arow['labels'])==len(brow['labels']) and all({k:x for k,x in a.items() if k!='box2d'}=={k:x for k,x in b.items() if k!='box2d'} for a,b in zip(arow['labels'],brow['labels']))
                write(run/name/(v+'.json'),out);per[name][v]=per['baseline'][v] if var['alpha']==0 else evaluate_hota(run/name,[v],split=split)
            timing[v]=time.monotonic()-began;proof[v]=dict(baseline_sha256=sha(ROOT/source['baseline']),offsets_sha256=sha(features/(v+'.npz')),only_boxes_changed=True,strict_past=True,boxes=n)
            write(run/'completed_videos.json',dict(timing=timing,proof=proof));write(run/'per_video.json',per)
            if index==0:
                cost=2*timing[v]*(len(p['sources'])-1)+120;available=end-time.time();write(run/'price.json',dict(first_full_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
        if is_dev:
            mean={name:float(np.mean([x['HOTA'] for x in values.values()])) for name,values in per.items()};chosen=[]
            for name,spec in specs.items():
                best=max([0.,.5,1.],key=lambda alpha:(mean[name+'_a'+str(alpha)],-alpha));chosen.append(dict(**spec,alpha=best,dev_mean_hota=mean[name+'_a'+str(best)]))
            write(BASE/'pixel_box_frozen_for_validation.json',dict(models=chosen,selected_on='dev only',val_used=False,created=now()))
            report=dict(complete=True,chosen=chosen,mean_per_video=mean,per_video=per,timing=timing,proof=proof,seconds=time.monotonic()-started,val_used=False)
        else:
            groups=dict(p['groups']);groups['all40']=[s['video'] for s in p['sources']];comparisons={};pooled={}
            for group,vs in groups.items():
                comparisons[group]={name+'_vs_baseline':paired({v:per[name][v]['HOTA'] for v in vs},{v:per['baseline'][v]['HOTA'] for v in vs}) for name in variants}
                for seed in [42,43,44]:
                    current=next(k for k in variants if variants[k]['model']=='current_seed'+str(seed));temporal=next(k for k in variants if variants[k]['model']=='temporal_seed'+str(seed))
                    comparisons[group]['temporal_vs_current_seed'+str(seed)]=paired({v:per[temporal][v]['HOTA'] for v in vs},{v:per[current][v]['HOTA'] for v in vs})
            for name in per:status(stage='Общая HOTA',variant=name);pooled[name]=evaluate_hota(run/name,groups['all40'],split='val')
            report=dict(complete=True,per_video=per,paired=comparisons,pooled=pooled,fixed_models=selected['models'],timing=timing,proof=proof,seconds=time.monotonic()-started,
                ground_truth_used_by_predictor=False,weights_or_alpha_changed_on_val=False,flyvis_core_used=False,limitation=p['limitation'])
        write(run/'report.json',report);write(run/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':
    folder=Path(__file__).name # overwritten only by thin, frozen stage wrappers
    raise SystemExit('Use registered dev/val wrappers')
