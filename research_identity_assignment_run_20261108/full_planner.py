"""Causal singleton proposals and relative track scores, with no annotation interface."""
from collections import defaultdict,deque
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.export_states import sequence_key
from research_identity_assignment_run_20261108.full_retina import FULL,Receptors
from tracking.foveated_eyes import crop_box
from tracking.spatial_identity_assignment import geometric_match,frame_points,centered_similarity
from tracking.z13_readout import HexGeometry
from tracking.z13_region_assignment import regions


_DIS_TRI=None
def dis_domain(points):
    # Same Delaunay hull and boundary convention as the saved legacy control.
    global _DIS_TRI
    if _DIS_TRI is None:
        from scipy.spatial import Delaunay
        xx,yy=np.meshgrid(np.arange(4,320,8)*4,np.arange(4,180,8)*4)
        _DIS_TRI=Delaunay(np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32))
    return _DIS_TRI.find_simplex(points)>=0


class Representations:
    def __init__(self,video,kind,backbone=None,*,archive=None):
        self.video=video;self.kind=kind;self.backbone=backbone;self.archive=archive;self.cache={};self.files={}
        self.renderer=None if archive else Receptors(video)
        self.folder=FULL/'features'/video;self.folder.mkdir(parents=True,exist_ok=True)
        norm=read(ROOT/f'runs/z13_target_center_dev_20261018/{kind}/normalization.json')
        self.mean=np.array(norm['mean'])[:34];self.sd=np.array(norm['sd'])[:34]

    def get(self,sequence,now):
        assert all(0<=f<=now for f,_ in sequence)
        key=sequence_key(self.video,sequence)
        if key in self.cache:return self.cache[key]
        if self.archive:
            with np.load(self.archive/(key+'.npz'),allow_pickle=False) as z:value=z[self.kind]
        else:
            path=self.folder/(key+'.'+self.kind+'.npy')
            if path.exists():value=np.load(path,allow_pickle=False)
            else:
                hx=np.stack([self.renderer.get(f,c,now) for f,c in sequence]).astype(np.float32)
                if self.kind=='R1':
                    from scripts.z13_target_model import receptor_features_rich
                    raw=receptor_features_rich(hx[None].astype(np.float64))[-1][0].T
                    value=((raw-self.mean)/self.sd).astype(np.float32)
                else:
                    assert self.kind=='C' and self.backbone is not None
                    import torch
                    with torch.no_grad():
                        x=torch.from_numpy(hx[None]).to(self.backbone.dev);state=self.backbone.init(x[:,:1])
                        for j in range(1,len(hx)):state,feature=self.backbone.step(state,x[:,j-1],x[:,j])
                        value=(((feature[0].cpu().numpy().T.astype(float)-self.mean)/self.sd)[:,4:12]).astype(np.float32)
                    del x,state,feature
                assert np.isfinite(value).all()
                with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            self.files[path.relative_to(ROOT).as_posix()]=sha(path)
        if len(self.cache)>=500:self.cache.clear()
        self.cache[key]=value;return value


class Planner:
    def __init__(self,video,arm,representations=None):
        self.video=video;self.arm=arm;self.representations=representations
        self.g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
        self.geom=HexGeometry(np.array(self.g['points'],np.float32),self.g['eye_size'])
        self.histories={};self.searches={};self.frame=-1;self.last=None

    def before(self,tracker,pool):
        self.frame=tracker.frame_id-1
        for t in pool:
            tid=int(t.track_id)
            if int(t.state)==2:self.searches.setdefault(tid,[]).append([self.frame,list(crop_box(tuple(t.mean[:4]),(1280,720)))])
            else:self.searches[tid]=[]

    def rows(self,pool):
        rows=[]
        for t in pool:
            tid=int(t.track_id);h=list(self.histories.get(tid,[]));h=[h[0]]*max(0,4-len(h))+h[-4:] if h else []
            rows.append(dict(tid=tid,state=int(t.state),cls=int(t.cls),start_frame=int(t.start_frame),mean=t.mean.tolist(),
                covariance=t.covariance.tolist(),history=h,current_crop=list(crop_box(tuple(t.mean[:4]),(1280,720))),search=list(self.searches.get(tid,[]))))
        return rows

    def build(self,rows,boxes,scores,classes,cost,frame,*,score=True):
        votes=[];seeds=defaultdict(list)
        for i,row in enumerate(rows):
            if row['state']!=2 or not row['history']:continue
            choices,_,_=regions(self.geom,row['current_crop'],np.array(row['mean'][:2]),boxes,scores,classes,row['cls'])
            choices=choices[1:]
            if len(choices)!=2:continue
            eligible=[j for j in choices if geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],boxes[j],self.g)['valid']]
            if len(eligible)==1:votes.append([i,eligible[0]]);seeds[eligible[0]].append(i)
        columns={}
        if not score:return dict(votes=votes,columns=columns)
        for j,ss in seeds.items():
            rivals=sorted(set(np.flatnonzero(cost[:,j]<=.8).tolist()+ss))
            observer=min(ss,key=lambda i:(float(np.trace(np.array(rows[i]['covariance'])[:2,:2])),rows[i]['start_frame'],rows[i]['tid']))
            view=rows[observer];record=dict(rows=rivals,geometry_valid=True,scores={},support=[])
            columns[str(j)]=record
            if len(rivals)<2 or any(rows[i]['cls']!=int(classes[j]) or not rows[i]['history'] for i in rivals):record['geometry_valid']=False;continue
            current_sequence=[[h['frame'],h['crop']] for h in view['history']]+view['search']
            current_points=frame_points(self.g['points'],view['current_crop'],self.g['eye_size']);current_valid=dis_domain(current_points)
            items=[]
            for i in rivals:
                row=rows[i];g=geometric_match(row['mean'],row['covariance'],row['history'],view['current_crop'],boxes[j],self.g)
                if not g['valid'] or len(set(g.get('q',[])))<3:record['geometry_valid']=False;break
                reference_points=frame_points(self.g['points'],row['history'][-1]['crop'],self.g['eye_size']);ref_valid=dis_domain(reference_points)
                pp=np.array(g['p'],int);qq=np.array(g['q'],int);valid=current_valid[pp]&ref_valid[qq];pp=pp[valid];qq=qq[valid]
                unique,inverse=np.unique(qq,return_inverse=True);record['support'].append(len(unique))
                if len(unique)<3:record['geometry_valid']=False;break
                items.append((i,g,pp,unique,inverse,ref_valid))
            if not record['geometry_valid']:continue
            current=self.representations.get(current_sequence,frame).astype(float);values=[]
            for i,g,pp,unique,inverse,ref_valid in items:
                row=rows[i];refseq=[[h['frame'],h['crop']] for h in row['history']]
                reference=self.representations.get(refseq,frame).astype(float)
                x=np.zeros((len(unique),current.shape[1]));np.add.at(x,inverse,current[pp]);x/=np.bincount(inverse)[:,None]
                if self.arm=='G_PERM':
                    ids=np.array(g['reference_ids'],int);ids=ids[ref_valid[ids]];vv=[]
                    for rep in range(32):
                        seed=int.from_bytes(hashlib.sha256(f"{self.video}|{row['tid']}|{row['history'][-1]['frame']}|{rep}|42".encode()).digest()[:8],'little')
                        y=reference.copy();y[ids]=y[np.random.default_rng(seed).permutation(ids)]
                        vv.append(centered_similarity(x,y[unique]))
                    values.append(float(np.mean(vv)) if all(v is not None for v in vv) else None)
                else:values.append(centered_similarity(x,reference[unique]))
            record['scores'][self.arm]=values
        return dict(votes=votes,columns=columns)

    def provide(self,tracker,pool,detections,cost):
        rows=self.rows(pool);assert len(detections)==len(self.high)
        boxes=self.high.xyxy;scores=self.high.conf;classes=self.high.cls
        plan=self.build(rows,boxes,scores,classes,cost,self.frame,score=tracker.identity_beta!=0)
        self.last=dict(frame=self.frame,rows=rows,plan=plan,cost=cost.copy())
        return plan

    def after(self,tracker):
        for t in tracker.tracked_stracks:
            if t.frame_id!=tracker.frame_id:continue
            tid=int(t.track_id)
            self.histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=self.frame,box=np.asarray(t.xyxy,np.float64).tolist(),
                crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))))
            self.searches[tid]=[]
