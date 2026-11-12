from common import *
import numpy as np,math
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.z13_readout import HexGeometry
p=verify();expected=read(OUT/'frozen_head_results.json')['records'];heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors;boxes=read(ROOT/'research_image_correspondence_probe_20261112/detector_box_audit.json')['records'];maximum=0.;checked=0
folder=ROOT/'research_terminal_response_probe_20261112';old=read(folder/'protocol.json')
with np.load(folder/'maps.npz') as maps,np.load(folder/'expanded_inputs.npz') as histories,np.load(ROOT/'research_input_precision_probe_20261112/expanded_inputs.npz') as original:
 for ci,c in enumerate(p['cases']):
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cc=np.array(r['current_crop']);det=np.array(r['current_box']);gt=np.array(next(x for x in boxes if x['old_id']==r['old_id'])['annotated_candidate_box'])
  for ei,ep in enumerate(c['endpoints']):
   base=original[f'index_{ci}'][:ep['length']];assert np.array_equal(base,histories[f'index_{ci}_{ei}']);assert np.array_equal(base,histories[f'identity_{ci}_{ei}']);assert np.array_equal(np.vstack([base,np.tile(base[-1:],(30,1))]),histories[f'history_hold_{ci}_{ei}']);assert np.array_equal(np.tile(base[-1:],(len(base)+30,1)),histories[f'terminal_hold_{ci}_{ei}'])
  for query_arm in ['C34','R134_temporal']:
   query=plan['query'][:,r['old_variants'][query_arm]['winner']];world=query/reader.eye_size*(cc[2:]-cc[:2])+cc[:2];mask=((world>=det[:2])&(world<=det[2:])&(world>=gt[:2])&(world<=gt[2:])).all(1)
   for support,M in [('full',np.ones(len(query),bool)),('detector_and_annotation',mask)]:
    for pair in old['pairs']:
     for arm in ARMS:
      arr=[]
      for ei,mode in enumerate(pair):
       z=maps[f'{mode}_{ci}_{ei}_C34'].astype(float) if arm=='C34' else maps[f'{mode}_{ci}_{ei}_R1'].astype(float)
       if arm=='R134_spatial':z=np.c_[z[:,:17],z[nb[:,1:],:17].mean(1)]
       arr.append(z)
      A=arr[0][plan['cells']][M];B=LinearNDInterpolator(reader.mesh,arr[1])(query)[M];ma=A.mean(0);mb=B.mean(0);a=A-ma;b=B-mb;va=np.mean(a*a,0);vb=np.mean(b*b,0);cov=np.mean(a*b,0);active=np.ptp(A,0)!=0;activeb=np.ptp(B,0)!=0
      for head in ['native','agreement']:
       den=np.sqrt(va*vb) if head=='native' else (va+vb+(ma-mb)**2)/2;q=np.divide(cov,den,out=np.zeros(34),where=(den>0)&active&activeb)*34/active.sum();h=heads[arm][head];g=(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];features=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta']);score=math.fsum(features*theta[:-1])+theta[-1];target=next(x for x in expected if x['old_id']==r['old_id'] and x['query']==query_arm and x['support']==support and x['pair']==pair and x['representation']==arm and x['head']==head);maximum=max(maximum,abs(score-target['score']));assert (score>0)==target['accepted'];checked+=1
assert maximum<1e-9;verify();save('causal_audit.json',dict(passed=True,scores=checked,max_error=maximum,original_histories_match=True,identity_exact=True,history_hold_only_appends_identical_terminal_image=True,terminal_hold_only_same_image=True,frozen_geometry_heads_and_thresholds=True,new_core_calls=0,method='Independent direct covariance channels and scalar weighted sum, compared to helper-based scoring. Exact array checks for all counterfactual histories. Shared stored core maps disclosed.',script_sha256=sha(__file__)))
print('CAUSAL_AUDIT',checked,maximum)
