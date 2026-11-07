"""Б2: реальный путь потерь YOLO через временную ветвь; только проверка реализации."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/b2_gradient'
MSK=timezone(timedelta(hours=3))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.next')
    temp.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode());temp.replace(p)


def prepare():
    assert not RUN.exists();RUN.mkdir()
    video=read(ROOT/'runs/b1_real_inputs_20261026/protocol.json')['fit84'][0]
    images=[f'data/bdd100k/images/track/train/{video}/{video}-{i:07d}.jpg' for i in range(1,6)]
    labels=f'data/bdd100k/labels/box_track_20/train/{video}.json'
    files=['scripts/b2_gradient_profile_20261027.py','tracking/b2_feature_temporal.py',
           'tracking/b1_learning.py','tracking/z13_readout.py','weights/baseline_seed42/best.pt',labels,*images]
    write(RUN/'protocol.json',dict(id='b2_predetection_gradient_v1',created=datetime.now(MSK).isoformat(),
        timezone='Europe/Moscow',author_stop='2026-10-27T07:20:00+03:00',budget_seconds=300,device='cuda:0',
        video=video,images=images,labels=labels,arms=['C','D','R1','B'],seed=26,
        action='Insert additive temporal feature correction in first YOLO detection-head input P3, before boxes/scores are produced. Same input/output adapters across arms; YOLO weights frozen.',
        history='Five consecutive 5Hz annotated images; no future relative to supervised last frame; FlyVis integration0.02s,10steps per0.2s interval. Images are train inputs; no detector accuracy on train is evaluated.',
        training='Three scratch Adam updates for common adapters and for free core parameters in D; objective is frozen one-to-one YOLO detection loss on the last frame. The final head forward_head removes only input detach, retaining fixed head values and weights. Scratch weights are not promoted.',
        acceptance=['At zero output all raw one-to-one detector boxes/scores match the same frozen detector exactly',
                    'After output warm-up, real detection loss produces finite nonzero core gradients in D and adapter gradients in all arms',
                    'YOLO parameters unchanged; record actual steps, memory, time, per-arm parameters; no HOTA claim'],
        limitations='Implementation and cost on one real clip, not training convergence, detector quality, or scientific B2 acceptance. Ordinary recurrent control parameter count differs and is reported.',
        sha256={f:sha(ROOT/f) for f in files}))
    write(RUN/'status.json',dict(state='ready',stage='registered before implementation run',updated=datetime.now(MSK).isoformat()))


def run():
    p=read(RUN/'protocol.json')
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    import torch,cv2
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from tracking.b2_feature_temporal import TemporalDetectionBranch,detector_features,detection_training_predictions
    from tracking.b1_learning import ReceptorHistory
    from tracking.z13_readout import FlyBackbone,make_modules
    from scripts.z13_target_model import build_arm
    from tracking.gpu_lock import GpuLock
    began=time.monotonic();records={}
    with GpuLock('flyvis_fovea_gpu'):
        detector=YOLO(str(ROOT/'weights/baseline_seed42/best.pt')).model.cuda().eval().float()
        detector.requires_grad_(False);detector.args=get_cfg(detector.args)
        before={n:x.detach().clone() for n,x in detector.named_parameters()}
        head=detector.model[-1];loss_fn=v8DetectionLoss(detector,tal_topk=1)
        letter=LetterBox(new_shape=(640,640),auto=True,stride=32)
        sequence=[];current=None
        for image in p['images']:
            frame=cv2.imread(str(ROOT/image));assert frame is not None
            value=letter(image=frame)
            tensor=torch.as_tensor(np.ascontiguousarray(value[:,:,::-1].transpose(2,0,1))[None],device='cuda',dtype=torch.float32)/255
            current=detector_features(detector,tensor);sequence.append(current[0])
        labels=read(ROOT/p['labels']);name=Path(p['images'][-1]).name
        entry=next(x for x in labels if x['name']==name)
        height,width=frame.shape[:2];ih,iw=tensor.shape[-2:];gain=min(640/height,640/width)
        dx=(iw-round(width*gain))/2;dy=(ih-round(height*gain))/2
        classes=[];boxes=[];mapping={v:k for k,v in detector.names.items()}
        for obj in entry['labels']:
            if obj['category'] not in mapping or 'box2d' not in obj:continue
            b=obj['box2d'];x1=b['x1']*gain+dx;x2=b['x2']*gain+dx;y1=b['y1']*gain+dy;y2=b['y2']*gain+dy
            boxes.append([(x1+x2)/2/iw,(y1+y2)/2/ih,(x2-x1)/iw,(y2-y1)/ih]);classes.append(mapping[obj['category']])
        batch=dict(img=tensor,batch_idx=torch.zeros(len(boxes),dtype=torch.long,device='cuda'),
                   cls=torch.tensor(classes,device='cuda',dtype=torch.float32)[:,None],
                   bboxes=torch.tensor(boxes,device='cuda',dtype=torch.float32).reshape(-1,4))
        with torch.no_grad():baseline=detection_training_predictions(head,current)
        for arm in p['arms']:
            write(RUN/'status.json',dict(state='running',stage='real detector loss backward',arm=arm,updated=datetime.now(MSK).isoformat()))
            if arm=='R1':
                bb=ReceptorHistory('cuda')
                _,geom,_=build_arm('R1',torch,None)
                points=geom.points;eye_size=geom.eye_size
            else:
                _,geom,stock=build_arm('C',torch,None);points=geom.points;eye_size=geom.eye_size
                if arm=='B':
                    _,cls=make_modules(torch);torch.manual_seed(26);bb=cls(geom.neighbors).cuda();bb.dev=torch.device('cuda')
                else:
                    bb=FlyBackbone(stock.est,trainable=arm=='D');bb.steps=10;bb.dt=.02
            torch.manual_seed(26);torch.cuda.manual_seed_all(26)
            branch=TemporalDetectionBranch(sequence[-1].shape[1],points,eye_size).cuda()
            groups=[dict(params=list(branch.parameters()),lr=.001)]
            if arm=='D':groups+=bb.param_groups()
            if arm=='B':groups+=[dict(params=list(bb.parameters()),lr=.001)]
            optimizer=torch.optim.Adam(groups)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
            losses=[];norms=[];adapter=[]
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                modified,correction=branch(sequence,bb)
                features=[modified,*current[1:]];pred=detection_training_predictions(head,features)
                if step==0:
                    exact=all(torch.equal(pred[k],baseline[k]) for k in ('boxes','scores'));assert exact
                loss,_=loss_fn(pred,batch);loss=loss.sum()
                if arm=='D':loss=loss+bb.penalty()
                assert torch.isfinite(loss);loss.backward()
                losses.append(float(loss.detach()))
                adapter.append(float(branch.input.weight.grad.norm()))
                if arm=='D':norms.append([float(x.grad.norm()) if x.grad is not None else 0 for x in bb.params])
                torch.nn.utils.clip_grad_norm_([x for g in groups for x in g['params']],1.)
                optimizer.step()
                if arm=='D':bb.clamp()
            torch.cuda.synchronize()
            assert adapter[-1]>0 and np.isfinite(adapter).all()
            if arm=='D':assert all(x>0 for x in norms[-1]) and np.isfinite(norms).all()
            records[arm]=dict(zero_detector_exact=exact,steps=3,losses=losses,adapter_gradient_norms=adapter,
                core_gradient_norms=norms,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                adapter_parameters=sum(x.numel() for x in branch.parameters()),
                backend_trainable_parameters=sum(x.numel() for g in groups[1:] for x in g['params']))
            del bb,branch,optimizer
        unchanged=all(torch.equal(before[n],x) for n,x in detector.named_parameters());assert unchanged
    write(RUN/'report.json',dict(passed=True,stage='real predetection path implementation only',arms=records,
        detector_weights_unchanged=unchanged,total_seconds=time.monotonic()-began,detector_accuracy_measured=False,
        hota_measured=False,scientific_hypothesis_tested=False,scratch_weights_promoted=False,limitation=p['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='predetection implementation verified, scientific pilot pending',updated=datetime.now(MSK).isoformat()))


if __name__=='__main__':
    try:prepare() if '--prepare' in sys.argv else run()
    except Exception as e:
        if RUN.exists():write(RUN/'failure.json',dict(error=repr(e),updated=datetime.now(MSK).isoformat()))
        raise
