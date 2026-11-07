"""Один заранее ограниченный обучающий пилот Б2 на104видео, не вся проверка Б2."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
from datetime import datetime
import sys,time,copy,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
FOLDER=RUN/'b2_sparse_training_pilot'


def main():
    p=service.read(FOLDER/'protocol.json')
    for f,h in p['sha256'].items(): assert service.sha(ROOT/f)==h,f
    c=service.read(RUN/'b2_causal960_record/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    end=datetime.fromisoformat(service.read(FOLDER/'execution_deadline.json')['deadline']).timestamp()
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    from tracking.b2_causal_detector import CausalP3Branch,detection_targets,CrowdAwareBCE
    from tracking.b2_feature_temporal import detector_features,detection_training_predictions
    from tracking.b1_learning import ReceptorHistory
    from tracking.z13_readout import FlyBackbone,make_modules
    from tracking.b1_fast_stimulus import fast_init
    from tracking.b1_index_copy_stimulus import index_copy_step
    from scripts.z13_target_model import build_arm
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2)
    FlyBackbone.init=fast_init;FlyBackbone.step=index_copy_step
    faulthandler.dump_traceback_later(300,exit=True)

    def progress(**kw):
        if time.time()>end-45: raise TimeoutError('Registered B2 sparse pilot deadline')
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(300,exit=True)
        service.write(FOLDER/'status.json',dict(updated=service.now(),**kw))

    class TimedR1(ReceptorHistory):
        dt=.02;steps=10
        def step(self,state,previous,current):
            value=current[:,None]
            alpha=1-(1-self.alpha)**(self.dt*self.steps/.2)
            state=state+(value-state)*alpha
            return state,torch.cat((state,value-state),1)

    with GpuLock('Local\\DissertationSingleGpu'):
        progress(stage='loading frozen detector and isolated six branches')
        yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
        first=service.read(ROOT/p['entries'][0]['manifest'])
        cap=cv2.VideoCapture(str(ROOT/first['video_path']));ok,frame=cap.read();cap.release();assert ok
        for _ in range(2):yolo.predict(frame,imgsz=960,conf=.01,iou=.7,device='0',verbose=False)
        predictor=yolo.predictor;detector=predictor.model.model
        detector.eval().float().requires_grad_(False);detector.args=get_cfg(overrides=detector.args)
        detector_before={k:v.detach().clone() for k,v in detector.named_parameters()}
        head=detector.model[-1]
        models={}
        _,geom,stock_c=build_arm('C',torch,None)
        _,geom_d,stock_d=build_arm('C',torch,None)
        bb_c=FlyBackbone(stock_c.est,trainable=False)
        bb_d=FlyBackbone(stock_d.est,trainable=True)
        assert all(torch.equal(x,y) for x,y in zip(bb_c.net.parameters(),bb_d.net.parameters()))
        assert {id(x) for x in bb_c.net.parameters()}.isdisjoint(id(x) for x in bb_d.net.parameters())
        core_c_before=[x.detach().clone() for x in bb_c.net.parameters()]
        assert sum(x.numel() for x in bb_d.params)==734
        _,Ordinary=make_modules(torch)
        torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
        ordinary=Ordinary(geom.neighbors).cuda();ordinary.dev=torch.device('cuda')
        for name in p['arms']:
            torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
            branch=CausalP3Branch(128,geom.points,geom.eye_size).cuda()
            bb={'C':bb_c,'D':bb_d}.get(name)
            if name=='R1':bb=TimedR1('cuda')
            if name in ('B','last','mean'):bb=copy.deepcopy(ordinary)
            groups=[dict(params=list(branch.parameters()),lr=.001)]
            if name=='D':groups+=bb.param_groups()
            if name in ('B','last','mean'):groups.append(dict(params=list(bb.parameters()),lr=.001))
            models[name]=dict(branch=branch,bb=bb,optimizer=torch.optim.Adam(groups),groups=groups,steps=0,nonzero_core_gradient=False)
        init={name:{k:v.detach().cpu().clone() for k,v in x['branch'].state_dict().items()} for name,x in models.items()}
        assert all(all(torch.equal(v,init['C'][k]) for k,v in s.items()) for s in init.values())
        torch.save(dict(adapters=init,ordinary=ordinary.state_dict(),stock_core=bb_c.net.state_dict(),protocol_sha256=service.sha(FOLDER/'protocol.json')),FOLDER/'initial.pt')

        def checkpoint(label):
            state={}
            for name,x in models.items():
                bb=x['bb']
                state[name]=dict(branch=x['branch'].state_dict(),backend=bb.net.state_dict() if name in ('C','D') else bb.state_dict() if hasattr(bb,'state_dict') else None,
                                 optimizer=x['optimizer'].state_dict(),steps=x['steps'])
            path=FOLDER/'checkpoints'/f'{label}.pt';path.parent.mkdir(exist_ok=True)
            torch.save(dict(models=state,protocol_sha256=service.sha(FOLDER/'protocol.json')),path)
            return dict(path=path.relative_to(ROOT).as_posix(),sha256=service.sha(path))

        def video(entry,training,phase):
            tick=time.monotonic();m=service.read(ROOT/entry['manifest'])
            assert service.sha(ROOT/entry['manifest'])==entry['manifest_sha256']
            for f,h in entry['input_sha256'].items(): assert service.sha(ROOT/f)==h,f
            clips=[m['clips'][i] for i in entry['clip_indices']]
            wanted={f for clip in clips for f in clip['input_frames']};target={c['input_frames'][-1]:c for c in clips}
            truth=service.read(ROOT/m['gt_only_detection_targets'])
            cache=np.load(ROOT/f"runs/train_cache/{m['video']}/detections/raw_video.npz")
            capture=cv2.VideoCapture(str(ROOT/m['video_path']));fps=capture.get(cv2.CAP_PROP_FPS)
            saved={};index=0;losses={name:[] for name in models};counts=0;peak_bytes=0
            while True:
                ok,frame=capture.read()
                if not ok:break
                if index in wanted:
                    progress(stage=phase,video=m['video'],frame=index,selected_clips=counts,total_clips=len(clips))
                    image=predictor.preprocess([frame]);features=detector_features(detector,image)
                    with torch.no_grad():
                        native=head(features)
                        found=predictor.postprocess(native,image,[frame])[0].boxes
                        mask=cache['frame_idx']==index
                        assert np.array_equal(found.xyxy.cpu().numpy().astype(np.float32),cache['xyxy'][mask]),('native boxes',m['video'],index)
                        assert np.array_equal(found.conf.cpu().numpy().astype(np.float32),cache['conf'][mask]),('native scores',m['video'],index)
                        assert np.array_equal(found.cls.cpu().numpy().astype(np.int16),cache['cls'][mask])
                    saved[index]=features[0].detach()
                    if index in target:
                        clip=target[index];sequence=[saved[f] for f in clip['input_frames']]
                        gt=truth[clip['target_reference_index']];assert gt['name']==clip['target_reference_name']
                        t=detection_targets(gt,detector.names,frame.shape,image.shape)
                        batch=dict(img=image,batch_idx=torch.zeros(len(t['cls']),device='cuda',dtype=torch.long),
                                   cls=torch.as_tensor(t['cls'][:,None],device='cuda',dtype=torch.float32),bboxes=torch.as_tensor(t['bboxes'],device='cuda'))
                        for name,x in models.items():
                            x['optimizer'].zero_grad(set_to_none=True)
                            with torch.set_grad_enabled(training):
                                mod,_=x['branch'](sequence,x['bb'],clip['input_frames'],fps,name if name in ('last','mean') else 'history')
                                pred=detection_training_predictions(head,[mod,*features[1:]])
                                if training and x['steps']==0:
                                    assert all(torch.equal(pred[k],native[1]['one2one'][k]) for k in ('boxes','scores'))
                                criterion=v8DetectionLoss(detector,tal_topk=1)
                                points,strides=make_anchors(pred['feats'],head.stride,.5)
                                criterion.bce=CrowdAwareBCE(points*strides,torch.as_tensor(t['crowd_xyxy'],device='cuda'),torch.as_tensor(t['crowd_cls'],device='cuda'),head.nc)
                                detection,_=criterion(pred,batch);loss=detection.sum()
                                objective=loss+x['bb'].penalty() if name=='D' else loss
                                assert torch.isfinite(objective)
                                if training:
                                    objective.backward()
                                    parameters=[v for g in x['groups'] for v in g['params']]
                                    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in parameters)
                                    if name=='D' and all(v.grad is not None and v.grad.norm()>0 for v in x['bb'].params):
                                        x['nonzero_core_gradient']=True
                                    torch.nn.utils.clip_grad_norm_(parameters,1.)
                                    x['optimizer'].step();x['steps']+=1
                                    if name=='D':x['bb'].clamp()
                            losses[name].append(float(loss.detach()))
                            del mod,pred,criterion,loss,objective
                        counts+=1
                    # Retain only entries still needed by a future selected clip.
                    future={f for c in clips if c['input_frames'][-1]>index for f in c['input_frames']}
                    saved={f:value for f,value in saved.items() if f in future}
                    peak_bytes=max(peak_bytes,sum(v.numel()*v.element_size() for v in saved.values()))
                index+=1
            capture.release();assert counts==len(clips)
            result=dict(video=m['video'],phase=phase,training=training,clips=counts,losses=losses,
                        seconds=time.monotonic()-tick,peak_retained_p3_bytes=peak_bytes,native_cache_exact=True,
                        manifest_sha256=entry['manifest_sha256'],updated=service.now())
            service.write(FOLDER/'completed'/f'{phase}_{m["video"]}.json',result)
            return result

        fit=[e for e in p['entries'] if e['split']=='fit84'];dev=[e for e in p['entries'] if e['split']=='fit20']
        first_train=video(fit[0],True,'fit');first_checkpoint=checkpoint('fit_000')
        first_dev=video(dev[0],False,'price_dev_probe')
        remaining_fit=sum(len(e['clip_indices']) for e in fit[1:]);remaining_dev=sum(len(e['clip_indices']) for e in dev)
        conservative=2*(first_train['seconds']/first_train['clips']*remaining_fit+first_dev['seconds']/first_dev['clips']*remaining_dev)+180
        price=dict(proceed=conservative<end-time.time()-30,conservative_remaining_seconds=conservative,
                   available_seconds=end-time.time()-30,first_train=first_train['seconds'],first_dev=first_dev['seconds'],
                   checkpoint=first_checkpoint,updated=service.now())
        service.write(FOLDER/'price.json',price)
        completed=[first_train]
        if price['proceed']:
            for i,e in enumerate(fit[1:],1):
                completed.append(video(e,True,'fit'));checkpoint(f'fit_{i:03d}')
                override=RUN/'history_tracker_author_continuation1/author_override.json'
                if override.exists() and service.read(override).get('accepted'):
                    service.write(FOLDER/'priority_stop.json',dict(reason='new explicit author matrix priority;checkpoint saved',updated=service.now()))
                    break
            if len(completed)==len(fit):
                final=checkpoint('final')
                completed += [video(e,False,'dev') for e in dev]
        assert all(torch.equal(detector_before[k],v) for k,v in detector.named_parameters())
        assert all(torch.equal(a,b) for a,b in zip(core_c_before,bb_c.net.parameters()))
        complete=len(completed)==104
        if complete: assert models['D']['nonzero_core_gradient']
        report=dict(complete=complete,scientific_verdict=None,training_only=True,hota_measured=False,
                    detector_accuracy_measured=False,price=price,completed=completed,
                    optimizer_steps={k:x['steps'] for k,x in models.items()},
                    adapter_parameters=4609,backend_trainable_parameters=dict(C=0,D=734,R1=0,B=sum(x.numel() for x in ordinary.parameters()),last=sum(x.numel() for x in ordinary.parameters()),mean=sum(x.numel() for x in ordinary.parameters())),
                    d_nonzero_core_gradient=models['D']['nonzero_core_gradient'],frozen_core_C_unchanged=True,
                    final=final if complete else None,detector_unchanged=True,updated=service.now())
        service.write(FOLDER/'report.json',report)
    faulthandler.cancel_dump_traceback_later()
    script='scripts/record_b2_sparse_pilot_20261029.py'
    folder=RUN/'b2_sparse_training_record'
    service.write(RUN/'ready_jobs/116a_b2_sparse_training_record.json',dict(name='b2_sparse_training_record',
                  folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))


if __name__=='__main__':main()
