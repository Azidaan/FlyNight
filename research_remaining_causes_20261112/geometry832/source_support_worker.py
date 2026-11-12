"""Separately preregistered source-support probe; no model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads(p.read_text(encoding='utf8'))
def cast(x):
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cast(v) for v in x]
 return x.tolist() if hasattr(x,'tolist') else x
p=read(OUT/'source_support_protocol.json');assert read(OUT/'source_support_GO.json')['approved'] is True
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
stage=read(ROOT/'research_stage_localization_20261112/protocol.json');old=read(OUT/'result.json');meta={r['old_id']:r for r in old['geometry']};g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];rows=[];metadata=[];samples={};started=time.monotonic();witnesserr=0.
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as maps,np.load(ROOT/'research_input_precision_probe_20261112/expanded_inputs.npz') as inputs:
 for ci,c in enumerate(stage['cases']):
  assert time.monotonic()-started<180
  r=c['case'];m=meta[r['old_id']];rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);a=np.array(m['reference']['box']);b=np.array(m['current']['box']);rb=np.array(r['reference_box']);cb=np.array(r['current_box']);x=reader.points*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];det=cb[:2]+(x-rb[:2])*(cb[2:]-cb[:2])/(rb[2:]-rb[:2]);offset=np.array(r['offsets'][r['old_variants']['C34']['winner']]);worlds=dict(detector_affine_saved_offset=det+offset,gt_center_zero_offset=x+(b[:2]+b[2:]-a[:2]-a[2:])/2,gt_affine_zero_offset=b[:2]+(x-a[:2])*(b[2:]-b[:2])/(a[2:]-a[:2]));queries={k:(v-cc[:2])*reader.eye_size/(cc[2:]-cc[:2]) for k,v in worlds.items()};valid=np.ones(len(x),bool)
  # Preserve old all-shift eye-valid support before adding alternative mappings.
  allqueries=(det[:,None]+np.array(r['offsets'])-cc[:2])*reader.eye_size/(cc[2:]-cc[:2]);valid&=(reader.mesh.find_simplex(allqueries)>=0).all(1)
  for q in queries.values():valid&=reader.mesh.find_simplex(q)>=0
  detmask=((x>=rb[:2])&(x<=rb[2:])).all(1);gtmask=((x>=a[:2])&(x<=a[2:])).all(1);supports=dict(reference_detector=detmask,reference_detector_and_gt=detmask&gtmask,reference_gt=gtmask)
  metadata.append(dict(old_id=r['old_id'],counts={k:int((v&valid).sum()) for k,v in supports.items()},gt_added=int((gtmask&~detmask&valid).sum()),gt_removed=int((detmask&~gtmask&valid).sum())))
  for arm in ['C34','R134_temporal','retina32']:
   arr=[inputs[f'precision32_{ci}'][ep['length']-1,:,None].astype(float) if arm=='retina32' else maps[f'index_{ci}_{ei}_'+('C34' if arm=='C34' else 'R1')].astype(float) for ei,ep in enumerate(c['endpoints'])]
   for ei in [0,1]:samples[f'{ci}_{arm}_map{ei}']=arr[ei]
   for variant,q in queries.items():
    for support,mask in supports.items():
     cells=np.flatnonzero(mask&valid);assert len(cells)>=3;A=arr[0][cells];B=sample_linear(reader.mesh,arr[1],q[cells]);key=f'{ci}_{arm}_{variant}_{support}';samples[key+'_a']=A;samples[key+'_b']=B;samples[key+'_q']=q[cells];samples[key+'_cells']=cells
     for head in ['native','agreement']:
      if head=='agreement':ch=agreement_channels(A,B,mean_penalty=True)
      else:
       aa,bb,an,bn,active,_=_center_pair(A,B);ch=np.divide((aa*bb).sum(0),an*bn,out=np.zeros(A.shape[1]),where=an*bn>0)*A.shape[1]/active.sum()
      score=None
      if arm!='retina32':
       h=heads[arm][head];features=np.r_[(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'],(ch-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta']);score=float(features@theta[:-1]+theta[-1])
       if support=='reference_detector':
        prior=next(z for z in old['records'] if z['old_id']==r['old_id'] and z['representation']==arm and z['query']=='C34' and z['variant']==variant and z['support']=='common' and z['head']==head)
        if len(cells)==prior['count']:
         err=max(abs(score-prior['score']),float(abs(ch-prior['channels']).max()));witnesserr=max(witnesserr,err);assert err<1e-9
      rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,variant=variant,support=support,count=len(cells),head=head,channels=ch,mean_similarity=float(ch.mean()),score=score,accepted=None if score is None else bool(score>0),patch_key=key))
with (OUT/'source_support_samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
out=dict(passed=True,records=rows,geometry=old['geometry'],metadata=metadata,witness_error=witnesserr,elapsed_seconds=time.monotonic()-started,samples_sha256=hashlib.sha256((OUT/'source_support_samples.npz').read_bytes()).hexdigest(),core_calls=0,new_fits=0,tracker_calls=0)
with (OUT/'source_support_result.json').open('x',encoding='utf8') as f:json.dump(cast(out),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print('SOURCE SUPPORT COMPLETE',len(rows))
