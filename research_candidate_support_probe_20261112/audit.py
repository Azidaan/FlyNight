from common import *
import math,numpy as np
from scipy.interpolate import LinearNDInterpolator
from tracking.z13_readout import HexGeometry
from scipy.spatial import Delaunay
p=verify();s=read(OUT/'scores.json');assert s['passed'] and sha(OUT/'patches.npz')==s['patches_sha256'];geo=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.array(geo['points']);mesh=Delaunay(points);nb=HexGeometry(points,geo['eye_size']).neighbors;boxes=read(ROOT/'research_image_correspondence_probe_20261112/detector_box_audit.json')['records'];max_score=max_sample=0.;verified=0;summaries=[]
def independent(a,b,g,arm,head):
    if len(a)<3:return None
    active=np.ptp(a,axis=0)!=0;other=np.ptp(b,axis=0)!=0
    if not active.any() or not other.any():return None
    ma=a.mean(0);mb=b.mean(0);va=np.mean((a-ma)**2,0);vb=np.mean((b-mb)**2,0);va[~active]=0;vb[~other]=0
    if head=='agreement':
        den=va+vb+(ma-mb)**2;q=np.divide(den-np.mean((a-b)**2,0),den,out=np.zeros(34),where=den>0)
    else:
        cov=np.array([np.cov(a[:,j],b[:,j],ddof=0)[0,1] if active[j] and other[j] else 0. for j in range(34)]);den=np.sqrt(va*vb);q=np.divide(cov,den,out=np.zeros(34),where=den>0)
    q[~active]=0;q*=34/active.sum();h=s['heads'][arm][head];x=np.r_[(np.array(g)-h['geometry']['mean'])/h['geometry']['scale'],(q-h['signal_mean'])/h['signal_scale']];return math.fsum(x*np.array(h['theta'][:-1]))+h['theta'][-1]
with np.load(OUT/'patches.npz',allow_pickle=False) as patches:
    for m in s['meta']:
        ci=m['case'];arm=m['representation'];r=p['cases'][ci]['case'];key=f'{ci}_{arm}';a=patches[key+'_a'];b=patches[key+'_b'];cells=patches[key+'_cells'];query=patches[key+'_query'];world=patches[key+'_world'];cc=np.array(r['current_crop']);assert np.array_equal(world,query*(cc[2:]-cc[:2])/geo['eye_size']+cc[:2])
        # Reconstruct saved map samples independently; no reuse of the producer's mesh interpolation.
        for ei,expected in [(0,a),(1,b)]:
            ep=p['cases'][ci]['endpoints'][ei]
            if arm=='C34':raw=np.load(ROOT/ep['paths']['C34'],allow_pickle=False).astype(float)
            else:
                rr=np.load(ROOT/ep['paths']['R1'],allow_pickle=False).astype(float);raw=rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)]
            actual=raw[cells] if ei==0 else LinearNDInterpolator(mesh,raw)(query);max_sample=max(max_sample,float(abs(actual-expected).max()));assert max_sample<1e-9
        det=r['current_box'];gt=next(x for x in boxes if x['old_id']==r['old_id'])['annotated_candidate_box']
        mask1=np.array([det[0]<=x<=det[2] and det[1]<=y<=det[3] for x,y in world]);mask2=np.array([gt[0]<=x<=gt[2] and gt[1]<=y<=gt[3] for x,y in world])&mask1
        assert np.array_equal(mask1,patches[key+'_detector']) and np.array_equal(mask2,patches[key+'_detector_and_annotation'])
        for row in [z for z in s['records'] if z['old_id']==r['old_id'] and z['representation']==arm]:
            mask=np.ones(len(a),bool) if row['mode'] in ['full','identity'] else mask1 if row['mode']=='detector' else mask2
            z=independent(a[mask],b[mask],r['geometry'],arm,row['head']);assert (z is None)==(row['score'] is None)
            if z is not None:max_score=max(max_score,abs(z-row['score']));assert max_score<1e-9 and (z>0)==row['accepted']
            if row['mode']=='full':assert abs(row['score']-row['expected'])<1e-9
            if row['mode']=='identity':assert row['score']==next(x['score'] for x in s['records'] if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==row['head'] and x['mode']=='full')
            verified+=1
        for ctrl in [z for z in s['controls'] if z['old_id']==r['old_id'] and z['representation']==arm]:
            N=ctrl['total'];k=ctrl['count'];assert N==len(a) and k==int((mask1 if ctrl['mode']=='detector' else mask2).sum())
            for start,old in enumerate(ctrl['scores']):
                indices=list(range(start,N))+list(range(start));indices=indices[:k];z=independent(a[indices],b[indices],r['geometry'],arm,ctrl['head']);assert (z is None)==(old is None)
                if z is not None:max_score=max(max_score,abs(z-old));assert max_score<1e-9
                verified+=1
            values=np.array([v for v in ctrl['scores'] if v is not None]);actual=next(x['score'] for x in s['records'] if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==ctrl['head'] and x['mode']==ctrl['mode'])
            summaries.append(dict(old_id=r['old_id'],representation=arm,head=ctrl['head'],mode=ctrl['mode'],count=k,total=N,actual_score=actual,control_min=float(values.min()) if len(values) else None,control_median=float(np.median(values)) if len(values) else None,control_max=float(values.max()) if len(values) else None,control_reject_count=int((values<=0).sum()),defined_controls=len(values),controls_score_at_or_below_actual=int((values<=actual).sum()) if actual is not None else None))
def get(i,h,m):return next(x for x in s['records'] if x['old_id']==i and x['head']==h and x['mode']==m and x['representation']=='C34')
gates={}
for mode in ['detector','detector_and_annotation']:
    gates[mode]=all(get(574,h,mode)['accepted'] is False and all(get(i,h,mode)['accepted'] is True for i in [70,1102]) and all(get(i,h,mode)['accepted'] is not None and (not get(i,h,'full')['accepted'] or get(i,h,mode)['accepted']) for i in [232,832]) for h in ['native','agreement'])
verify();save('audit.json',dict(passed=True,verified_scores=verified,max_score_error=max_score,max_sample_error=max_sample,mask_reconstruction_exact=True,identity_exact=True,primary_gates=gates,primary_any=any(gates.values()),control_summaries=summaries,independence='SciPy interpolation, explicit box inequalities, np.cov Pearson and MSE concordance, math.fsum scores, independent cyclic index construction. Shared saved maps, head coefficients, geometry, annotations; no independent physical pixel labels.'))
for row in s['records']:
    if row['representation']=='C34':print(row['old_id'],row['head'],row['mode'],row['count'],None if row['score'] is None else round(row['score'],6))
print('PRIMARY',gates)
