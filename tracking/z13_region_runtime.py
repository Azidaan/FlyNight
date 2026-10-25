"""Annotation-free runtime adapter for the learned regional FlyVis readout.

This module is inactive unless explicitly installed by a separate benchmark.
Activation checks the frozen readout gate and model/calibration hashes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from tracking.z13_ridge_localizer import CenterRidgeLocalizer,receptor_step,ridge_response,map_center
from tracking.z13_region_assignment import regions,pool_activity,apply_policy
from tracking.z13_region_choice import categorical_rows

ROOT=Path(__file__).resolve().parents[1]


def verified_file(relative,digest):
    path=(ROOT/relative).resolve();path.relative_to(ROOT.resolve())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Runtime input hash changed: '+relative)
    return path


def region_cost(torch,model,activity,target,geom,crop,prediction,detections,track_class,original,tau):
    """Same ROI, logits, gate and hard action as the frozen one-step experiment."""
    source=np.asarray(original)
    if tau>1:return source.copy(),dict(action='unchanged',probability=None)
    boxes=np.asarray([d.xyxy for d in detections],float).reshape(-1,4)
    confidence=np.asarray([d.score for d in detections],float)
    classes=np.asarray([int(d.cls) for d in detections])
    choices,masks,geo=regions(geom,crop,prediction,boxes,confidence,classes,track_class)
    x=pool_activity(activity,masks)
    with torch.no_grad():
        logits=model(torch.as_tensor(x[None],dtype=torch.float32,device='cpu'),
            torch.as_tensor(np.asarray(target)[None],dtype=torch.float32,device='cpu'),
            torch.as_tensor(geo[None],dtype=torch.float32,device='cpu'),
            torch.ones((1,len(x)),dtype=torch.bool,device='cpu'))[0].cpu().numpy()
    index,probability=apply_policy(logits,tau)
    if index<0:return source.copy(),dict(action='unchanged',probability=probability)
    valid=np.zeros(len(detections),bool);valid[[c for c in choices if c>=0]]=True
    rows,indices=categorical_rows(source,confidence,valid)
    assert indices[1:]==choices
    return rows[index+1].astype(source.dtype,copy=False),dict(action='unmatched' if choices[index]<0 else 'region',detection=choices[index],probability=probability)


def make_region_tracker(base_class,localizer_getter):
    """Change only costs returned inside the original first association call."""
    class RegionalTracker(base_class):
        _regional_first=False

        def _first_association(self,*args,**kwargs):
            previous=self._regional_first;self._regional_first=True
            try:return super()._first_association(*args,**kwargs)
            finally:self._regional_first=previous

        def get_dists(self,tracks,detections):
            original=super().get_dists(tracks,detections)
            localizer=localizer_getter()
            return localizer.costs(tracks,detections,original) if self._regional_first and localizer is not None else original
    return RegionalTracker


class RegionalLocalizer(CenterRidgeLocalizer):
    def __init__(self,config_path,bank,frame_size=(1280,720)):
        import torch
        from tracking.z13_filter_descriptor import make_ranker
        config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        if config.get('format')!='z13-region-assignment-v1':raise ValueError('Explicit regional readout configuration required')
        gate_path=verified_file(config['gate_file'],config['gate_sha256'])
        if not json.loads(gate_path.read_text(encoding='utf-8'))['primary_pass']:raise ValueError('Learned readout gate did not pass')
        model_path=verified_file(config['model_file'],config['model_sha256'])
        calibration_path=verified_file(config['calibration_file'],config['calibration_sha256'])
        calibration=json.loads(calibration_path.read_text(encoding='utf-8'))
        if not calibration.get('frozen_before_evaluation') or calibration['tau']!=config['tau']:raise ValueError('Calibration mismatch')
        report_path=verified_file(config['model_report_file'],config['model_report_sha256'])
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if (report['variant'],report['seed'],report['tau'])!=(config['variant'],config['seed'],config['tau']):
            raise ValueError('Model variant, seed or calibrated policy differs from runtime configuration')
        engine_path=verified_file(config['engine_config_file'],config['engine_config_sha256'])
        super().__init__(engine_path,bank,frame_size)
        self.variant=config['variant'];self.tau=float(config['tau']);self.max_batch=8
        if self.variant not in ('D','C','R1','geometry'):raise ValueError('Unknown variant')
        if self.arm!=('R1' if self.variant=='geometry' else self.variant):raise ValueError('Engine representation differs from readout')
        self.model=make_ranker(torch,activity=self.variant!='geometry').cpu()
        self.model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True));self.model.eval()
        self.query={};self.stats.update(region_choices=0,unmatched_choices=0)

    def advance(self,eyes,current):
        torch=self.torch;cur=current.half().float()
        if self.arm=='R1':
            state,f=receptor_step(np.stack([e['state'] for e in eyes]),cur.cpu().numpy());packed=[x.copy() for x in state]
        else:
            state=self._unpack(torch.stack([e['state'] for e in eyes]));last=torch.stack([e['last'] for e in eyes])
            state,f=self.bb.step(state,last,cur);f=f.double().cpu().numpy();packed=[x.clone() for x in self._pack(state)]
        normalized=self.features(f);answers=[]
        for i,e in enumerate(eyes):
            e.update(state=packed[i],last=cur[i].clone(),query_features=normalized[i])
            answers.append(map_center(ridge_response(e['ridge'],normalized[i]),self.geom))
        return answers

    def before_associate(self,pool,eye_frame,tracked_state):
        from tracking.foveated_eyes import crop_box
        self.query={};self.frame_size=(eye_frame.shape[1],eye_frame.shape[0])
        if self.tau>1:return
        lost=[t for t in pool if t.state!=tracked_state and int(t.track_id) in self.hist]
        alive={int(t.track_id) for t in lost}
        for tid in list(self.eyes):
            if tid not in alive:self.eyes.pop(tid)
        with self.torch.no_grad():
            if self.variant!='geometry':
                new=[t for t in lost if int(t.track_id) not in self.eyes]
                for left in range(0,len(new),self.max_batch):self._new(new[left:left+self.max_batch])
            for left in range(0,len(lost),self.max_batch):
                group=lost[left:left+self.max_batch];crops=[crop_box(tuple(float(v) for v in t.mean[:4]),self.frame_size) for t in group]
                if self.variant!='geometry':
                    current=self.bank._render(eye_frame,crops)[:,0];self.advance([self.eyes[int(t.track_id)] for t in group],current)
                for t,crop in zip(group,crops):
                    tid=int(t.track_id)
                    activity=np.zeros((721,68)) if self.variant=='geometry' else self.eyes[tid]['query_features']
                    target=np.zeros(68) if self.variant=='geometry' else self.eyes[tid]['ridge'][1]
                    self.query[tid]=dict(activity=activity,target=target,crop=crop,prediction=t.mean[:2].copy(),track_class=int(t.cls))
                    self.stats['steps']+=1

    def costs(self,tracks,detections,original):
        answer=np.asarray(original).copy()
        for i,t in enumerate(tracks):
            q=self.query.get(int(t.track_id))
            if q is None:continue
            if not np.array_equal(q['prediction'],t.mean[:2]):raise ValueError('Tracker prediction changed after regional observation')
            row,decision=region_cost(self.torch,self.model,q['activity'],q['target'],self.geom,q['crop'],q['prediction'],detections,q['track_class'],answer[i],self.tau)
            answer[i]=row
            if decision['action']=='unchanged':self.stats['refusals']+=1
            else:
                self.stats['interventions']+=1
                self.stats['region_choices' if decision['action']=='region' else 'unmatched_choices']+=1
        return answer
