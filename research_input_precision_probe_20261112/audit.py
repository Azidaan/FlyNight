import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import math,numpy as np
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
from tracking.z13_readout import HexGeometry
p=verify();rep=read(OUT/'replay.json');assert rep['passed'] and sha(OUT/'maps.npz')==rep['maps_sha256'];assert sha(OUT/'expanded_inputs.npz')==rep['expanded_inputs_sha256']
source=read(ROOT/'research_candidate_support_probe_20261112/scores.json');heads=source['heads'];geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors;rows=[];moments=[];max_interp=max_feature=max_score=0.
with np.load(OUT/'maps.npz') as maps,np.load(OUT/'expanded_inputs.npz') as inputs:
 for ci,c in enumerate(p['cases']):
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];original=np.stack([np.load(ROOT/n) for n in c['views']]).astype(np.float32)
  assert np.array_equal(inputs[f'index_{ci}'],original) and np.array_equal(inputs[f'identity_{ci}'],original) and np.array_equal(inputs[f'precision32_{ci}'].astype(np.float16),original.astype(np.float16))
  for mode in p['modes']:
   hx=inputs[f'{mode}_{ci}'];assert hx.shape==original.shape and np.isfinite(hx).all()
   for arm in ARMS:
    mm=[]
    for ei in [0,1]:
     rr=maps[f'{mode}_{ci}_{ei}_R1'].astype(float);mm.append(maps[f'{mode}_{ci}_{ei}_C34'].astype(float) if arm=='C34' else rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)])
    query=plan['query'][:,r['old_variants'][arm]['winner']];a=mm[0][cells];b=sample_linear(reader.mesh,mm[1],query);other=LinearNDInterpolator(reader.mesh,mm[1])(query);max_interp=max(max_interp,float(abs(other-b).max()));assert max_interp<1e-9
    ia=hx[c['endpoints'][0]['length']-1][cells].astype(float);ib=sample_linear(reader.mesh,hx[c['endpoints'][1]['length']-1,:,None].astype(float),query)[:,0];im,ibm=ia.mean(),ib.mean();moments.append(dict(old_id=r['old_id'],mode=mode,representation=arm,mean_reference=float(im),mean_candidate=float(ibm),variance_reference=float(np.var(ia)),variance_candidate=float(np.var(ib)),covariance=float(np.mean((ia-im)*(ib-ibm)))))
    va=np.var(a,0);vb=np.var(b,0);ma=a.mean(0);mb=b.mean(0);active=np.ptp(a,0)!=0;activeb=np.ptp(b,0)!=0;cov=np.mean((a-ma)*(other-other.mean(0)),0)
    for head in ['native','agreement']:
     if head=='agreement':q=agreement_channels(a,b,mean_penalty=True);den=va+vb+(ma-mb)**2;ind=np.divide(den-np.mean((a-other)**2,0),den,out=np.zeros(34),where=den>0);ind[~active|~activeb]=0;ind*=34/active.sum()
     else:
      x,y,an,bn,aa,bb=_center_pair(a,b);q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/aa.sum();den=np.sqrt(va*vb);ind=np.divide(cov,den,out=np.zeros(34),where=(den>0)&active&activeb)*34/active.sum()
     max_feature=max(max_feature,float(abs(ind-q).max()));assert max_feature<1e-9
     h=heads[arm][head];g=(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];theta=np.array(h['theta']);features=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];score=float(features@theta[:-1]+theta[-1]);indscore=math.fsum(np.r_[g,(ind-h['signal_mean'])/h['signal_scale']]*theta[:-1])+theta[-1];max_score=max(max_score,abs(score-indscore));assert max_score<1e-9
     if mode=='index':
      old=next(x['score'] for x in source['records'] if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==head and x['mode']=='full');assert abs(score-old)<1e-3,(r['old_id'],arm,head,score,old)
     if mode=='identity':assert score==next(x['score'] for x in rows if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==head and x['mode']=='index')
     rows.append(dict(old_id=r['old_id'],label=r['label'],mode=mode,representation=arm,head=head,score=score,accepted=bool(score>0),channels=q))
gates={}
for arm in ARMS:
 for head in ['native','agreement']:
  d={x['old_id']:x for x in rows if x['mode']=='precision32' and x['representation']==arm and x['head']==head};gates[arm+'|'+head]=all(d[i]['accepted'] for i in [70,1102]) and any(d[i]['accepted'] for i in [232,832]) and not d[574]['accepted']
verify();save('scores.json',dict(passed=True,records=rows,input_moments=moments,gates=gates));save('audit.json',dict(passed=True,records=len(rows),max_interpolation_error=max_interp,max_feature_error=max_feature,max_score_error=max_score,identity_exact=True,inputs_and_weights_unchanged=True,primary_gate=all(gates['C34|'+h] for h in ['native','agreement']),independence='SciPy interpolation, direct covariance/MSE channels and scalar sum; shared maps, normalizations and heads. Core dynamics and renderer not independently reimplemented.'))
print(json.dumps([{k:v for k,v in x.items() if k!='channels'} for x in rows if x['representation']=='C34']))
