"""Реальное причинное сопоставление внутри трекера на неизменных детекциях."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now

def run(folder):
    import cv2,numpy as np,torch
    from tracking.busca_runtime import load_model
    from tracking.gpu_lock import GpuLock
    from tracking.lazy_appearance_association import AppearanceServer,LazyAppearanceXYWH
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    run=BASE/folder;p=read(run/'protocol.json');is_dev=p['phase']=='dev';split='train' if is_dev else 'val';start=time.monotonic()
    for path,h in p['sha256'].items():assert sha(ROOT/path)==h,path
    end=datetime.fromisoformat(read(run/'execution_deadline.json')['deadline']).timestamp();torch.set_num_threads(2)
    def status(**kw):
        if time.time()>end-10 or (run/'stop.request').exists():raise TimeoutError('Lazy association stage original deadline')
        write(run/'status.json',dict(state='running',updated=now(),**kw),required=False)
    chosen=None if is_dev else read(BASE/'lazy_appearance_frozen.json');per={};proof={};timing={};events={}
    with GpuLock('Local\\DissertationSingleGpu'):
        status(stage='Строгая загрузка неизменного извлекателя');model,coverage=load_model('cuda');encoder=model.reid_encoder;encoder.use_domain_adaptation=False;encoder.model.eval()
        buffers={k:v.clone() for k,v in encoder.model.named_buffers()}
        variants={'baseline':dict(weight=0.,null_seed=None)}
        if is_dev:
            for w in [.1,.3]:variants['appearance_w'+str(w)]=dict(weight=w,null_seed=None)
        else:
            variants['appearance']=dict(weight=chosen['weight'],null_seed=None)
            for seed in [42,43,44]:variants['null_seed'+str(seed)]=dict(weight=chosen['weight'],null_seed=seed)
        per={name:{} for name in variants}
        for index,source in enumerate(p['sources']):
            began=time.monotonic();v=source['video'];expected=read(ROOT/source['baseline']);align=read(ROOT/source['alignment'])['matches'];assert len(align)==len(expected)
            anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};assert len(anchors)==len(align)
            cache=load_detection_cache(ROOT/source['detections']);server=AppearanceServer(encoder)
            trackers={name:LazyAppearanceXYWH(SimpleNamespace(**p['config']),server,spec['weight'],spec['null_seed']) for name,spec in variants.items()}
            cap=cv2.VideoCapture(str(ROOT/source['video_path']));assert cap.isOpened();rows={name:[] for name in variants};rawbase=[]
            for frame in range(int(cache['frame_count'])):
                if frame%50==0:status(stage='Реальные назначения и обновление состояния',video=v,index=index,total=len(p['sources']),frame=frame,total_frames=int(cache['frame_count']),embedded=server.embedded)
                ok,image=cap.read();assert ok;server.start_frame(frame,image);xyxy,conf,classes=detections_for_frame(cache,frame);dets=_DetSlice(xyxy,conf,classes)
                for name,tracker in trackers.items():
                    out=np.asarray(tracker.update(dets),np.float32).reshape(-1,8)
                    if name=='baseline' and len(out):rawbase.append(np.column_stack([np.full(len(out),frame),out[:,:7]]).astype(np.float32))
                    if frame in anchors:
                        ref=anchors[frame];template={k:expected[ref][k] for k in ['name','frameIndex']};row=scalabel_frame(v,template,out)
                        if name=='baseline':assert row==expected[ref],(v,ref,'disabled path differs from native')
                        rows[name].append(row)
            cap.release()
            if 'raw_array' in source:
                original=np.load(ROOT/source['raw_array'])[source['raw_array_key']];actual=np.concatenate(rawbase) if rawbase else np.zeros((0,8),np.float32)
                assert np.array_equal(original,actual),(v,'native raw arrays differ')
            for name,values in rows.items():
                status(stage='Полная HOTA внутри трекера',video=v,index=index,variant=name)
                write(run/name/(v+'.json'),values);per[name][v]=source['baseline_metrics'] if name=='baseline' else evaluate_hota(run/name,[v],split=split)
                write(run/'events'/name/(v+'.json'),trackers[name].events)
            assert all(torch.equal(buffers[k],value) for k,value in encoder.model.named_buffers())
            proof[v]=dict(disabled_native_json_exact=True,disabled_raw_exact=('raw_array' in source),embedded=server.embedded,
                changed_cost_frames={n:t.changed_cost_frames for n,t in trackers.items()},
                output_differs_from_baseline={n:rows[n]!=rows['baseline'] for n in variants if n!='baseline'},
                all_event_history_strictly_past=all(all(f<e['frame'] for f in e['past_frames']) for t in trackers.values() for e in t.events))
            timing[v]=time.monotonic()-began;write(run/'per_video.json',per);write(run/'completed_videos.json',dict(timing=timing,proof=proof))
            if index==0:
                cost=2*timing[v]*(len(p['sources'])-1)+180;available=end-time.time();write(run/'price.json',dict(first_full_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
        if is_dev:
            scores={n:float(np.mean([m['HOTA'] for m in values.values()])) for n,values in per.items()}
            selected=max(variants,key=lambda n:(scores[n],-variants[n]['weight']));frozen=dict(weight=variants[selected]['weight'],dev_mean_hota=scores[selected],selected=selected,selected_on='dev only',checkpoint_sha256=coverage['checkpoint_sha256'],created=now())
            write(BASE/'lazy_appearance_frozen.json',frozen)
            result=dict(complete=True,per_video=per,mean_per_video=scores,chosen=frozen,proof=proof,timing=timing,seconds=time.monotonic()-start,val_used=False,coverage=coverage)
        else:
            groups=dict(p['groups']);groups['all40']=[s['video'] for s in p['sources']];comparisons={}
            for group,vs in groups.items():
                comparisons[group]={'appearance_vs_baseline':paired({v:per['appearance'][v]['HOTA'] for v in vs},{v:per['baseline'][v]['HOTA'] for v in vs})}
                for seed in [42,43,44]:comparisons[group]['appearance_vs_null_seed'+str(seed)]=paired({v:per['appearance'][v]['HOTA'] for v in vs},{v:per['null_seed'+str(seed)][v]['HOTA'] for v in vs})
            pooled={}
            for name in per:status(stage='Общие метрики сорока видео',variant=name);pooled[name]=evaluate_hota(run/name,groups['all40'],split='val')
            result=dict(complete=True,per_video=per,pooled=pooled,paired=comparisons,proof=proof,timing=timing,frozen=chosen,
                batchnorm_buffers_unchanged=True,ground_truth_used_by_predictor=False,seconds=time.monotonic()-start,limitation=p['limitation'])
        write(run/'report.json',result);write(run/'status.json',dict(state='completed',updated=now()))
