from common import *
import numpy as np,math
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.z13_readout import HexGeometry
p=verify();frozen_protocol=read(OUT/'factorial_addendum.json');assert sha(__file__)==frozen_protocol['script_sha256'];d=read(OUT/'terminal_addendum.json')
for n,h in d['inputs'].items():assert sha(ROOT/n)==h,n
folder=ROOT/'research_terminal_response_probe_20261112';old=read(folder/'protocol.json');rep=read(folder/'replay.json');assert rep['passed'] and sha(folder/'maps.npz')==rep['maps_sha256'] and sha(folder/'expanded_inputs.npz')==rep['expanded_inputs_sha256'];assert read(folder/'audit.json')['passed']
assert [c['case']['old_id'] for c in old['cases']]==[c['case']['old_id'] for c in p['cases']]
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;boxes=read(ROOT/'research_image_correspondence_probe_20261112/detector_box_audit.json')['records'];base=read(OUT/'result.json')['records'];rows=[];headrows=[];maxerr=0.;baselineerr=0.;heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads']
def compare(a,b):
 global maxerr
 ma=a.mean(0);mb=b.mean(0);x=a-ma;y=b-mb;va=(x*x).mean(0);vb=(y*y).mean(0);cov=(x*y).mean(0);den=va+vb+(ma-mb)**2;q=np.divide(2*cov,den,out=np.zeros_like(cov),where=den>0)
 ind=[]
 for j in range(a.shape[1]):
  A=a[:,j];B=b[:,j];M=math.fsum(A)/len(A);N=math.fsum(B)/len(B);D=math.fsum((float(z)-M)**2 for z in A)/len(A)+math.fsum((float(z)-N)**2 for z in B)/len(B)+(M-N)**2;C=math.fsum((float(z)-M)*(float(w)-N) for z,w in zip(A,B))/len(A);ind.append(2*C/D if D else 0.)
 maxerr=max(maxerr,float(abs(q-ind).max()));assert maxerr<1e-9;return float(q.mean())
with np.load(folder/'maps.npz') as maps,np.load(folder/'expanded_inputs.npz') as inputs:
 for ci,c in enumerate(p['cases']):
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cc=np.array(r['current_crop']);det=np.array(r['current_box']);gt=np.array(next(x for x in boxes if x['old_id']==r['old_id'])['annotated_candidate_box'])
  for ei in [0,1]:
   assert np.array_equal(inputs[f'terminal_hold_{ci}_{ei}'][-1],inputs[f'index_{ci}_{ei}'][-1]);assert np.all(inputs[f'terminal_hold_{ci}_{ei}']==inputs[f'terminal_hold_{ci}_{ei}'][-1])
  for query_arm in ['C34','R134_temporal']:
   query=plan['query'][:,r['old_variants'][query_arm]['winner']];world=query/reader.eye_size*(cc[2:]-cc[:2])+cc[:2];mask=((world>=det[:2])&(world<=det[2:])&(world>=gt[:2])&(world<=gt[2:])).all(1)
   for support,M in [('full',np.ones(len(query),bool)),('detector_and_annotation',mask)]:
    for pair in old['pairs']:
     for arm in ARMS:
      arr=[]
      for ei,mode in enumerate(pair):
       m=maps[f'{mode}_{ci}_{ei}_C34'].astype(float) if arm=='C34' else maps[f'{mode}_{ci}_{ei}_R1'].astype(float)
       if arm=='R134_spatial':m=np.c_[m[:,:17],m[nb[:,1:],:17].mean(1)]
       arr.append(m)
      a=arr[0][plan['cells']][M];b=LinearNDInterpolator(reader.mesh,arr[1])(query)[M];value=compare(a,b)
      from tracking.channel_agreement_correspondence import agreement_channels
      from tracking.channel_balanced_correspondence import _center_pair
      for head in ['native','agreement']:
       if head=='agreement':q=agreement_channels(a,b,mean_penalty=True)
       else:
        x,y,an,bn,active,activeb=_center_pair(a,b);q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
       h=heads[arm][head];gg=(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];t=np.array(h['theta']);features=np.r_[gg,(q-h['signal_mean'])/h['signal_scale']];score=float(features@t[:-1]+t[-1]);ind=math.fsum(features*t[:-1])+t[-1];assert abs(score-ind)<1e-10
       headrows.append(dict(old_id=r['old_id'],label=r['label'],query=query_arm,support=support,pair=pair,representation=arm,head=head,score=score,accepted=bool(score>0)))
      if pair==['index','index']:
       z=next(x for x in base if x['old_id']==r['old_id'] and x['stage']==arm and x['query']==query_arm and x['support']==support);baselineerr=max(baselineerr,abs(z['concordance']-value));assert baselineerr<1e-4
      rows.append(dict(old_id=r['old_id'],query=query_arm,support=support,pair=pair,representation=arm,concordance=value))
gaps=[]
for query in ['C34','R134_temporal']:
 for support in ['full','detector_and_annotation']:
  for pair in old['pairs']:
   for arm in ARMS:
    rr={x['old_id']:x for x in rows if x['query']==query and x['support']==support and x['pair']==pair and x['representation']==arm}
    for tid in [232,832]:gaps.append(dict(query=query,support=support,pair=pair,representation=arm,own_id=tid,foreign=rr[574]['concordance'],own=rr[tid]['concordance'],gap=rr[574]['concordance']-rr[tid]['concordance']))
verify();save('frozen_head_results.json',dict(passed=True,records=headrows,max_channel_formula_error=maxerr,weights_and_thresholds_fixed=True,posthoc=True,new_core_calls=0))
print('FROZEN_FACTORIAL',len(headrows))
