from common import *
import math,numpy as np
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
import psutil,shutil
p=verify();s=read(OUT/'scores.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);maximum=0.
with np.load(OUT/'expanded_inputs.npz') as inputs:
 for ci,c in enumerate(p['cases']):
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets'])
  for mode in p['modes']:
   hx=inputs[f'{mode}_{ci}']
   for arm in ARMS:
    a=hx[c['endpoints'][0]['length']-1][plan['cells']].astype(float);query=plan['query'][:,r['old_variants'][arm]['winner']];b=LinearNDInterpolator(reader.mesh,hx[c['endpoints'][1]['length']-1].astype(float))(query);ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b)
    vals=dict(mean_reference=ma,mean_candidate=mb,variance_reference=math.fsum((float(x)-ma)**2 for x in a)/len(a),variance_candidate=math.fsum((float(x)-mb)**2 for x in b)/len(b),covariance=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a,b))/len(a))
    old=next(x for x in s['input_moments'] if x['old_id']==r['old_id'] and x['mode']==mode and x['representation']==arm)
    maximum=max(maximum,max(abs(v-old[k]) for k,v in vals.items()))
assert maximum<1e-12;save('input_moments_audit.json',dict(passed=True,records=len(s['input_moments']),max_error=maximum,method='Independent SciPy interpolator and scalar fsum moments; shared query plan and inputs.',script_sha256=sha(__file__)))
reports=[]
for phase,token in [('render_failed','5cf0963261204adb8389f433f0f5a46d'),('render_v2','551e89d1f50248d0bc0e799237d04e0e'),('replay','11c389f15f24421bbeba96dedde109bf')]:
 folder=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'/token;rep=read(folder/'memory_guard.json');ended=[]
 for pid,birth in rep['owned'].items():
  try:proc=psutil.Process(int(pid));gone=proc.create_time()!=birth
  except psutil.NoSuchProcess:gone=True
  assert gone,(phase,pid,'old worker still live');ended.append(int(pid))
 shutil.copy2(folder/'memory_guard.worker.log',OUT/(phase+'_worker.log'));reports.append(dict(phase=phase,report=rep,all_owned_ended=True,ended=ended))
assert reports[0]['report']['returncode']==1 and all(x['report']['returncode']==0 and x['report']['reason'] is None for x in reports[1:]);verify();save('execution_receipt.json',dict(passed=True,all_owned_ended=True,reports=reports,archive_inputs_unchanged=True,weights_unchanged=read(OUT/'replay.json')['weights_unchanged']))
print('MOMENTS_AND_EXECUTION_VERIFIED',maximum)
