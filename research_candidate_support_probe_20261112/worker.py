import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-12T07:40:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
from tracking.z13_readout import HexGeometry
p=verify();save('execution_start.json',dict(created=now(),supervisor=observation.read(),cpu_only=True))
geo=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geo['points'],geo['eye_size'],[1280,720]);nb=HexGeometry(np.array(geo['points']),geo['eye_size']).neighbors
pred=read(ROOT/'research_agreement_readout_20261112/predictions.json');native=read(ROOT/'runs/domain_matched_native_20261109/deployment.json');native_old=read(ROOT/'research_identity_training_stability_20261109/comparison.json')['cases'];boxes=read(ROOT/'research_image_correspondence_probe_20261112/detector_box_audit.json')['records']
patches={};heads={};rows=[];controls=[];meta=[]
for arm in ARMS:
    fit=next(f for f in pred['fits'] if f['video']=='FINAL' and f['model']==arm+'__agreement')
    heads[arm]=dict(native=dict(theta=native['models'][arm]['theta'],geometry=native['geometry_normalization'],signal_mean=[0.]*34,signal_scale=[1.]*34),agreement=dict(theta=fit['fit']['theta'],geometry=fit['normalization'],signal_mean=fit['signal_mean'],signal_scale=fit['signal_scale']))
def evaluate(a,b,geometry,arm,head):
    if len(a)<3:return None
    if head=='agreement':q=agreement_channels(a,b,mean_penalty=True)
    else:
        x,y,an,bn,active,active_b=_center_pair(a,b)
        if not active.any() or not active_b.any():return None
        q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
    if q is None:return None
    h=heads[arm][head];g=(np.array(geometry)-h['geometry']['mean'])/h['geometry']['scale'];x=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta'])
    return dict(score=float(x@theta[:-1]+theta[-1]),channels=q)
for ci,case in enumerate(p['cases']):
    r=case['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];cc=np.array(r['current_crop']);gt=np.array(next(x for x in boxes if x['old_id']==r['old_id'])['annotated_candidate_box']);det=np.array(r['current_box'])
    for arm in ARMS:
        maps=[]
        for ep in case['endpoints']:
            if arm=='C34':m=np.load(ROOT/ep['paths']['C34'],allow_pickle=False).astype(float)
            else:
                rr=np.load(ROOT/ep['paths']['R1'],allow_pickle=False).astype(float);m=rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)]
            maps.append(m)
        query=plan['query'][:,r['old_variants'][arm]['winner']];world=query*(cc[2:]-cc[:2])/reader.eye_size+cc[:2];inside=lambda bb:((world>=bb[:2])&(world<=bb[2:])).all(1)
        key=f'{ci}_{arm}';a=maps[0][cells];b=sample_linear(reader.mesh,maps[1],query)
        values=dict(a=a,b=b,cells=cells,query=query,world=world,detector=inside(det),detector_and_annotation=inside(det)&inside(gt))
        for n,v in values.items():patches[key+'_'+n]=v
        meta.append(dict(case=ci,old_id=r['old_id'],representation=arm,total=len(cells),detector=int(values['detector'].sum()),detector_and_annotation=int(values['detector_and_annotation'].sum()),geometry=r['geometry']))
        for head in p['heads']:
            result=evaluate(a,b,r['geometry'],arm,head);assert result is not None
            expected=(next(x for x in native_old if x['old_id']==r['old_id'])['baseline'][arm] if head=='native' else next(x for x in pred['screen'] if x['old_id']==r['old_id'])['scores'][arm+'__agreement'])
            assert abs(result['score']-expected)<1e-9,(r['old_id'],arm,head,result['score'],expected)
            rows.append(dict(old_id=r['old_id'],representation=arm,head=head,mode='full',count=len(a),label=r['label'],accepted=result['score']>0,expected=expected,**result))
save('baseline_gate.json',dict(passed=True,records=rows.copy(),all30_original_scores_before_masks=True))
for m in meta:
    ci=m['case'];arm=m['representation'];r=p['cases'][ci]['case'];key=f'{ci}_{arm}';a=patches[key+'_a'];b=patches[key+'_b'];N=len(a)
    (OUT/'status.json').write_text(json.dumps(dict(stage='support_masks',old_id=r['old_id'],representation=arm,updated=now())),encoding='utf8')
    for mode in ['identity','detector','detector_and_annotation']:
        mask=np.ones(N,bool) if mode=='identity' else patches[key+'_'+mode];k=int(mask.sum())
        for head in p['heads']:
            result=evaluate(a[mask].copy(),b[mask].copy(),r['geometry'],arm,head)
            if mode=='identity':assert result['score']==next(x['score'] for x in rows if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==head and x['mode']=='full')
            rows.append(dict(old_id=r['old_id'],representation=arm,head=head,mode=mode,count=k,label=r['label'],accepted=None if result is None else result['score']>0,**(dict(score=None,channels=None) if result is None else result)))
            if mode!='identity':
                values=[]
                for start in range(N):
                    ix=(np.arange(k)+start)%N;z=evaluate(a[ix],b[ix],r['geometry'],arm,head);values.append(None if z is None else z['score'])
                controls.append(dict(old_id=r['old_id'],representation=arm,head=head,mode=mode,count=k,total=N,scores=values,rule='k consecutive indices modulo N, all N starts; order is stored template cell order.'))
    print('MASKS',r['old_id'],arm,flush=True)
verify()
with (OUT/'patches.npz').open('xb') as f:np.savez_compressed(f,**patches)
save('scores.json',dict(passed=True,created=now(),records=rows,controls=controls,meta=meta,heads=heads,patches_sha256=sha(OUT/'patches.npz'),new_fits=0,new_maps=0,core_calls=0,tracker_calls=0))
(OUT/'status.json').write_text(json.dumps(dict(stage='complete',updated=now())),encoding='utf8')
