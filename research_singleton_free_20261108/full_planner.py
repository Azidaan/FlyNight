"""Same historical geometry/maps; only free-edge application is new."""
from collections import defaultdict
import numpy as np
from research_singleton_free_20261108.common import *
from research_identity_assignment_run_20261108.full_planner import Planner as Parent,dis_domain
from tracking.spatial_identity_assignment import geometric_match,frame_points,centered_similarity,flow_similarity


class Planner(Parent):
    def __init__(self,video,arm,representations=None):
        mapped=arm.replace('U_','G_',1)
        if arm in ['U_GEO','U_THRESHOLD','B0','ZERO']:mapped='G'
        super().__init__(video,mapped,representations)

    def build(self,rows,boxes,scores,classes,cost,frame,*,score=True):
        plan=super().build(rows,boxes,scores,classes,cost,frame,score=False)
        if not score:return plan
        seeds=defaultdict(list)
        for i,j in plan['votes']:seeds[j].append(i)
        columns={};plan['columns']=columns
        for j,ss in seeds.items():
            rivals=sorted(set(np.flatnonzero(cost[:,j]<=.8).tolist()+ss))
            observer=min(ss,key=lambda i:(float(np.trace(np.array(rows[i]['covariance'])[:2,:2])),rows[i]['start_frame'],rows[i]['tid']))
            view=rows[observer];record=dict(rows=rivals,geometry_valid=True,scores={},support=[]);columns[str(j)]=record
            if len(rivals)<2 or any(rows[i]['cls']!=int(classes[j]) or not rows[i]['history'] for i in rivals):record['geometry_valid']=False;continue
            seq=[[h['frame'],h['crop']] for h in view['history']]+view['search']
            cv=dis_domain(frame_points(self.g['points'],view['current_crop'],self.g['eye_size']));items=[]
            for i in rivals:
                row=rows[i];g=geometric_match(row['mean'],row['covariance'],row['history'],view['current_crop'],boxes[j],self.g)
                if not g['valid'] or len(set(g.get('q',[])))<3:record['geometry_valid']=False;break
                rv=dis_domain(frame_points(self.g['points'],row['history'][-1]['crop'],self.g['eye_size']))
                pp=np.array(g['p'],int);qq=np.array(g['q'],int);keep=cv[pp]&rv[qq];pp=pp[keep];qq=qq[keep]
                unique,inverse=np.unique(qq,return_inverse=True);record['support'].append(len(unique))
                if len(unique)<3:record['geometry_valid']=False;break
                items.append((i,g,pp,unique,inverse,rv))
            if not record['geometry_valid']:continue
            if self.arm=='G_SHIFT':continue
            current=self.representations.get(seq,frame).astype(float);values=[]
            for i,g,pp,unique,inverse,rv in items:
                row=rows[i];reference=self.representations.get([[h['frame'],h['crop']] for h in row['history']],frame).astype(float)
                x=np.zeros((len(unique),current.shape[1]));np.add.at(x,inverse,current[pp]);x/=np.bincount(inverse)[:,None]
                if self.arm=='G_PERM':
                    ids=np.array(g['reference_ids'],int);ids=ids[rv[ids]];vv=[]
                    for rep in range(32):
                        seed=int.from_bytes(hashlib.sha256(f"{self.video}|{row['tid']}|{row['history'][-1]['frame']}|{rep}|42".encode()).digest()[:8],'little')
                        y=reference.copy();y[ids]=y[np.random.default_rng(seed).permutation(ids)]
                        vv.append(centered_similarity(x,y[unique]))
                    values.append(float(np.mean(vv)) if all(v is not None for v in vv) else None)
                elif self.arm=='G_DIS':values.append(flow_similarity(x,reference[unique],g['scale']))
                else:values.append(centered_similarity(x,reference[unique]))
            record['scores'][self.arm]=values
        return plan
