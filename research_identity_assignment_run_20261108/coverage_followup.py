"""New quantitative coverage audit; reuse old clipping flags and all saved maps."""
import os,sys,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial import ConvexHull
from research_identity_assignment_run_20261108.common import *


def area(poly):
    p=np.asarray(poly,float)
    return abs(float(np.dot(p[:,0],np.roll(p[:,1],1))-np.dot(p[:,1],np.roll(p[:,0],1))))/2 if len(p)>=3 else 0.


def polygon_box(box,equations):
    x1,y1,x2,y2=box;poly=[np.array(p,float) for p in [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]
    for e in equations:
        if not poly:break
        result=[];prev=poly[-1];dp=float(prev@e[:2]+e[2])
        for cur in poly:
            dc=float(cur@e[:2]+e[2]);a=dp<=0;b=dc<=0
            if a!=b:result.append(prev+(cur-prev)*(dp/(dp-dc)))
            if b:result.append(cur)
            prev=cur;dp=dc
        poly=result
    return area(poly)


class Coverage:
    def __init__(self):
        self.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],float)
        self.hull=ConvexHull(self.points);self.mask=np.zeros((391,391),bool)
        for x,y in self.points.astype(int):self.mask[max(0,y-6):min(391,y+7),max(0,x-6):min(391,x+7)]=True
        self.integral=np.pad(self.mask.astype(float).cumsum(0).cumsum(1),((1,0),(1,0)))
    def integral_to(self,x,y):
        x=float(np.clip(x,0,391));y=float(np.clip(y,0,391));i=min(int(x),390);j=min(int(y),390);fx=x-i;fy=y-j;S=self.integral
        return S[j,i]+fx*(S[j,i+1]-S[j,i])+fy*(S[j+1,i]-S[j,i])+fx*fy*self.mask[j,i]
    def support_area(self,b):
        x1,y1,x2,y2=b;F=self.integral_to
        return max(0,F(x2,y2)-F(x1,y2)-F(x2,y1)+F(x1,y1))
    def measure(self,box,crop):
        b=np.asarray(box,float);c=np.asarray(crop,float);den=max(0,b[2]-b[0])*max(0,b[3]-b[1])
        assert den>0
        visible=np.r_[np.maximum(b[:2],[0,0]),np.minimum(b[2:],[1280,720])]
        visible_area=max(0,visible[2]-visible[0])*max(0,visible[3]-visible[1])
        if visible_area<=0:return dict(image_visible_fraction=0,direct_square_fraction=None,center_hull_fraction=None,receptor_support_fraction=None)
        low=np.maximum(visible[:2],c[:2]);high=np.minimum(visible[2:],c[2:]);square=np.prod(np.maximum(0,high-low))/visible_area
        eye=np.r_[(visible[:2]-c[:2])*391/(c[2:]-c[:2]),(visible[2:]-c[:2])*391/(c[2:]-c[:2])]
        scale=np.prod(c[2:]-c[:2])/391**2
        hull=polygon_box(eye,self.hull.equations)*scale/visible_area
        support=self.support_area(eye)*scale/visible_area
        assert all(-1e-9<=v<=1+1e-9 for v in [square,hull,support])
        return dict(image_visible_fraction=min(1,visible_area/den),direct_square_fraction=float(np.clip(square,0,1)),
            center_hull_fraction=float(np.clip(hull,0,1)),receptor_support_fraction=float(np.clip(support,0,1)))


def summarize(rows):
    answer={'N':len(rows)}
    for key in ['image_visible_fraction','direct_square_fraction','center_hull_fraction','receptor_support_fraction']:
        values=[r[key] for r in rows if r[key] is not None]
        answer[key]=dict(known=len(values),minimum=min(values) if values else None,mean=float(np.mean(values)) if values else None,
            median=float(np.median(values)) if values else None,p10=float(np.percentile(values,10)) if values else None,
            p90=float(np.percentile(values,90)) if values else None,fully_covered=sum(x>=1-1e-8 for x in values),
            zero_coverage=sum(x<=1e-8 for x in values))
    return answer


def main(obs):
    import cv2
    from research_identity_assignment_run_20261108.evaluator import Labels
    verify_phase('coverage_phase.json');cov=Coverage();rng=np.random.default_rng(42);support_error=0;hull_error=0
    hull=cov.points[cov.hull.vertices].astype(np.float32)
    for _ in range(100):
        lows=rng.uniform(-60,360,2);highs=lows+rng.uniform(2,180,2);b=np.r_[lows,highs]
        xx=np.arange(391);wx=np.maximum(0,np.minimum(xx+1,b[2])-np.maximum(xx,b[0]));wy=np.maximum(0,np.minimum(xx+1,b[3])-np.maximum(xx,b[1]))
        brute=float(wy@cov.mask.astype(float)@wx);error=abs(brute-cov.support_area(b));support_error=max(support_error,error);assert error<1e-8
        x1,y1,x2,y2=b;box=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)
        reference,_=cv2.intersectConvexConvex(hull,box);error=abs(reference-polygon_box(b,cov.hull.equations))/391**2
        hull_error=max(hull_error,error);assert error<1e-6
    old=read(ROOT/'runs/spatial_identity_maps_20261107/bottleneck_rows.json');old={r['uid']:r for r in old}
    strata={r['uid']:r for r in read(ROOT/'research_identity_assignment_protocol_20261108/evaluation_strata.json')['rows']}
    outcomes={a:{r['uid']:r for r in read(RUN/'outcomes'/('evaluate26_'+a+'.json'))['rows']} for a in ['B0','G','G_FLYVIS']}
    groups=defaultdict(list);rows=[];old_flags=defaultdict(Counter);gt_missing=Counter()
    exports=read(RUN/'export_complete.json')['videos']
    for number,(video,entry) in enumerate(exports.items()):
        data=read(ROOT/entry['path']);lab=None
        for frame,snap in data['snapshots'].items():
            for q in snap['queries']:
                if q['uid'] not in strata:continue
                if lab is None:lab=Labels(video,data)
                row=next(r for r in snap['rows'] if r['tid']==q['tid']);s=strata[q['uid']];l=lab.supervision[q['uid']]
                r=dict(uid=q['uid'],video=video,split=q['split'],primary_known=s['primary_known'],reason=s['reason'],candidates=[])
                if row['history']:
                    h=row['history'][-1];r['reference']=cov.measure(h['box'],h['crop']);groups[q['split']+'|reference'].append(r['reference'])
                for k,j in enumerate(q['candidate_high_ids']):
                    identity=l['current_identity'][k];role='unknown' if identity is None or l['target_identity'] is None else ('true' if identity==l['target_identity'] else 'foreign')
                    value=cov.measure(snap['boxes'][j],row['current_crop']);r['candidates'].append(dict(role=role,**value))
                    groups[q['split']+'|'+('primary|' if s['primary_known'] else 'shadow|')+role].append(value)
                target_boxes=[b for identity,category,b in lab.truth.get(q['frame'],[]) if identity==l['target_identity']]
                if len(target_boxes)==1:
                    r['current_GT_target']=cov.measure(target_boxes[0],row['current_crop'])
                    groups[q['split']+'|'+('primary' if s['primary_known'] else 'shadow')+'|GT_target'].append(r['current_GT_target'])
                else:r['current_GT_target']=None;gt_missing[q['split']]+=1
                if q['uid'] in old:
                    previous=old[q['uid']];idx=previous['true_candidate'];flags=old_flags[q['split']]
                    flags['pairs']+=1;flags['reference_clipped']+=int(previous['reference_field_clipped']);flags['any_candidate_clipped']+=int(any(previous['current_field_clipped']))
                    flags['true_candidate_clipped']+=int(previous['current_field_clipped'][idx]);flags['foreign_candidate_clipped']+=int(previous['current_field_clipped'][1-idx])
                if q['uid'] in outcomes['G']:
                    r['frozen_outcomes']={a:outcomes[a][q['uid']]['outcome'] for a in outcomes}
                rows.append(r)
        status('fourth_problem_coverage_audit',complete=number+1,total=len(exports),rows=len(rows),guard=obs.read())
    assert len(rows)==1991
    write(RUN/'coverage_rows.json',rows,new=True)
    report=dict(passed=True,queries=len(rows),old_clipping_flags_reused={k:dict(v) for k,v in old_flags.items()},
        summaries={k:summarize(v) for k,v in groups.items()},GT_target_bbox_unavailable=dict(gt_missing),
        independent_geometry_checks=dict(random_boxes=100,receptor_integral_max_absolute_error=support_error,convex_area_max_eye_square_fraction_error=hull_error),
        measurements=dict(direct_square='Visible bbox area inside the actually supplied square crop.',
            center_hull='Visible bbox area inside the convex hull of721 column centers; matches the historical clipping concept.',
            receptor_support='Visible bbox area inside the union of the13x13 input-filter pixel supports. This differs from the hull of centers.'),
        causal_limit='Geometric coverage is not recognition quality. Whole-frame lamina preprocessing can transmit spatial context; recurrent receptive fields and partial visibility also prevent equating missing center coverage with zero information. No larger crop or new core response is tested here.',
        runtime_leakage='GT boxes appear only in this evaluator child, never in tracker candidates, alignment, input windows or parameter selection.',
        rows_sha256=sha(RUN/'coverage_rows.json'))
    verify_phase('coverage_phase.json');write(RUN/'coverage_followup.json',report,new=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent-report',required=True);args=parser.parse_args()
    from research_identity_assignment_run_20261108.full_cpu import Observer
    main(Observer(args.parent_report))
