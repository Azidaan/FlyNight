"""Независимые CPU-потери всех200devцелей и источники шести ветвей Б2."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_sparse_training_audit';source=RUN/'b2_sparse_training_pilot'
    p=service.read(folder/'protocol.json');tp=service.read(source/'protocol.json')
    for name in ('b2_sparse_training_pilot','b2_sparse_training_record'):
        c=service.read(RUN/name/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    report=service.read(source/'report.json')
    if not report['complete']:
        service.write(folder/'report.json',dict(passed=False,complete=False,scientific_verdict=None,reason='Incomplete training; no promotion',updated=service.now()));return
    for f,h in tp['sha256'].items():assert service.sha(ROOT/f)==h,f
    assert all(x==840 for x in report['optimizer_steps'].values())
    assert report['d_nonzero_core_gradient'] and report['frozen_core_C_unchanged'] and report['detector_unchanged']
    final=ROOT/report['final']['path'];assert service.sha(final)==report['final']['sha256']
    snapshot={f:service.sha(ROOT/f) for f in [final.relative_to(ROOT).as_posix(),(source/'report.json').relative_to(ROOT).as_posix(),(source/'protocol.json').relative_to(ROOT).as_posix(),(source/'initial.pt').relative_to(ROOT).as_posix()]}
    service.write(folder/'input_manifest.json',snapshot)
    import cv2,numpy as np,torch
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.set_default_device('cpu')
    test=torch.tensor(2.,device='cpu',requires_grad=True);(test*test).backward();assert test.grad==4 and not torch.cuda.is_initialized()
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    from tracking.b2_pilot_runtime import load_branches,modified_features
    from tracking.b2_feature_temporal import detector_features,detection_training_predictions
    from tracking.b2_causal_detector import detection_targets,CrowdAwareBCE
    from tracking.z13_readout import FREE
    state=torch.load(final,map_location='cpu',weights_only=False)
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False)
    assert state['protocol_sha256']==initial['protocol_sha256']==service.sha(source/'protocol.json')
    for name,x in state['models'].items():
        assert x['steps']==840
        assert all(torch.equal(value,initial['adapters']['C'][key]) for key,value in initial['adapters'][name].items())
        assert all(torch.isfinite(v).all() for v in x['branch'].values())
        for opt in x['optimizer']['state'].values():assert int(opt['step'])==840
    for key,value in state['models']['C']['backend'].items():assert torch.equal(value,initial['stock_core'][key]),key
    for key,value in state['models']['D']['backend'].items():
        if key not in FREE: assert torch.equal(value,initial['stock_core'][key]),key
    assert any(not torch.equal(state['models']['D']['backend'][key],initial['stock_core'][key]) for key in FREE)
    models=load_branches(state,'cpu')
    detector=YOLO(str(ROOT/'weights/baseline_seed42/best.pt')).model.cpu().float().eval()
    detector.fuse(verbose=False);detector.requires_grad_(False);detector.args=get_cfg(overrides=detector.args)
    head=detector.model[-1];letter=LetterBox(new_shape=(960,960),auto=True,stride=32)
    end=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp()
    completed=[];maximum=0.;compared=0
    expected={x['video']:x for x in report['completed'] if x['phase']=='dev'}
    entries=[e for e in tp['entries'] if e['split']=='fit20'];assert len(expected)==len(entries)==20
    for vi,e in enumerate(entries):
        marker=folder/'completed'/(e['video']+'.json')
        key=dict(snapshot=snapshot,entry=e,reference=expected[e['video']])
        if marker.exists():
            old=service.read(marker);assert old['inputs']==key
            completed.append(old);maximum=max(maximum,old['maximum_loss_difference']);compared+=old['losses_checked'];continue
        began=time.monotonic();m=service.read(ROOT/e['manifest'])
        assert service.sha(ROOT/e['manifest'])==e['manifest_sha256']
        for f,h in e['input_sha256'].items():assert service.sha(ROOT/f)==h,f
        truth=service.read(ROOT/m['gt_only_detection_targets'])
        clips=[m['clips'][i] for i in e['clip_indices']];target={c['input_frames'][-1]:c for c in clips}
        wanted={f for c in clips for f in c['input_frames']};saved={};losses={n:[] for n in models}
        capture=cv2.VideoCapture(str(ROOT/m['video_path']));fps=capture.get(cv2.CAP_PROP_FPS);frame=0
        while True:
            if time.time()>end-30:raise TimeoutError('B2 training audit original deadline')
            ok,image=capture.read()
            if not ok:break
            if frame in wanted:
                value=letter(image=image)
                tensor=torch.from_numpy(np.ascontiguousarray(value[:,:,::-1].transpose(2,0,1))[None]).to(device='cpu',dtype=torch.float32)/255
                features=detector_features(detector,tensor);saved[frame]=features[0]
                if frame in target:
                    clip=target[frame];sequence=[saved[f] for f in clip['input_frames']]
                    gt=truth[clip['target_reference_index']];assert gt['name']==clip['target_reference_name']
                    t=detection_targets(gt,detector.names,image.shape,tensor.shape)
                    batch=dict(img=tensor,batch_idx=torch.zeros(len(t['cls']),dtype=torch.long,device='cpu'),
                               cls=torch.tensor(t['cls'][:,None],dtype=torch.float32,device='cpu'),bboxes=torch.tensor(t['bboxes'],device='cpu'))
                    with torch.no_grad():
                        for name,model in models.items():
                            modified,_=modified_features(model,sequence,clip['input_frames'],fps,name)
                            pred=detection_training_predictions(head,[modified,*features[1:]])
                            criterion=v8DetectionLoss(detector,tal_topk=1);points,strides=make_anchors(pred['feats'],head.stride,.5)
                            criterion.bce=CrowdAwareBCE(points*strides,torch.tensor(t['crowd_xyxy'],device='cpu'),torch.tensor(t['crowd_cls'],device='cpu'),head.nc)
                            loss,_=criterion(pred,batch);value=float(loss.sum())
                            ref=expected[e['video']]['losses'][name][len(losses[name])]
                            diff=abs(value-ref);maximum=max(maximum,diff);compared+=1
                            assert diff<=p['loss_absolute_tolerance'],(e['video'],frame,name,value,ref,diff)
                            losses[name].append(value)
                future={f for c in clips if c['input_frames'][-1]>frame for f in c['input_frames']}
                saved={f:v for f,v in saved.items() if f in future}
                service.write(folder/'status.json',dict(stage='Independent CPUdevloss',video=e['video'],index=vi,total=20,frame=frame,losses_checked=compared,max_difference=maximum,updated=service.now()))
            frame+=1
        capture.release();assert all(len(v)==10 for v in losses.values())
        item=dict(video=e['video'],inputs=key,losses=losses,losses_checked=60,maximum_loss_difference=max(abs(a-b) for n in losses for a,b in zip(losses[n],expected[e['video']]['losses'][n])),seconds=time.monotonic()-began,passed=True)
        service.write(marker,item);completed.append(item)
        if vi==0:
            need=2*item['seconds']*19+90;available=end-time.time()
            service.write(folder/'price.json',dict(proceed=need<available,conservative_remaining_seconds=need,available_seconds=available,first_complete_video_seconds=item['seconds']))
            if need>=available:
                service.write(folder/'report.json',dict(passed=False,complete=False,stopped_on_cost=True,completed=completed,scientific_verdict=None,updated=service.now()));return
    assert not torch.cuda.is_initialized() and compared==1200
    service.write(folder/'report.json',dict(passed=True,complete=True,dev_videos=20,dev_targets=200,losses_checked=compared,max_cpu_gpu_loss_difference=maximum,
                  initializations_sources_optimizers_checked=True,frozen_YOLO_and_C=True,only_D734core_updated=True,
                  no_cuda_initialized=True,completed=completed,scientific_verdict=None,updated=service.now()))
    service.write(source/'independent_audit.json',service.read(folder/'report.json'))


if __name__=='__main__':main()
