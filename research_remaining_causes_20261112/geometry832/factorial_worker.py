"""Cached GT-coordinate by held-state factorial; no forward or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.hex_linear_interpolation import sample_linear
from scipy.spatial import Delaunay
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads(p.read_text(encoding='utf8'))
def cast(x):
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cast(v) for v in x]
 return x.tolist() if hasattr(x,'tolist') else x
p=read(OUT/'factorial_protocol.json');assert read(OUT/'factorial_GO.json')['approved'] is True
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
original=read(OUT/'result.json');stage=read(ROOT/'research_stage_localization_20261112/protocol.json');heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(np.array(g['points']));rows=[];samples={};err=0.;started=time.monotonic()
with np.load(OUT/'paired_samples.npz') as base,np.load(ROOT/'research_terminal_response_probe_20261112/maps.npz') as held,np.load(ROOT/'research_terminal_response_probe_20261112/expanded_inputs.npz') as terminalinputs:
 for ci,c in enumerate(stage['cases']):
  assert time.monotonic()-started<180
  r=c['case'];cells=base[f'{ci}_cells'];common=base[f'{ci}_common'];samples[f'{ci}_cells']=cells;samples[f'{ci}_common']=common
  for arm in ['C34','R134_temporal','retina32']:
   bank={}
   for ei,ep in enumerate(c['endpoints']):
    for mode in ['index','terminal_hold']:
     if mode=='index':m=base[f'{ci}_{arm}_map{ei}']
     elif arm=='retina32':m=base[f'{ci}_{arm}_map{ei}'];assert np.array_equal(terminalinputs[f'terminal_hold_{ci}_{ei}'][-1],terminalinputs[f'index_{ci}_{ei}'][-1])
     else:m=held[f'terminal_hold_{ci}_{ei}_'+('C34' if arm=='C34' else 'R1')].astype(float)
     bank[(ei,mode)]=m;samples[f'{ci}_{arm}_{ei}_{mode}']=m
   for qa in ['C34','R134_temporal']:
    for variant in p['coordinate_variants']:
     q=base[f'{ci}_{arm}_{qa}_{variant}_common_q']
     for pair in p['state_pairs']:
      A=bank[(0,pair[0])][cells][common];B=sample_linear(mesh,bank[(1,pair[1])],q);key=f'{ci}_{arm}_{qa}_{variant}_{pair[0]}_{pair[1]}';samples[key+'_a']=A;samples[key+'_b']=B;samples[key+'_q']=q
      for head in ['native','agreement']:
       if head=='agreement':ch=agreement_channels(A,B,mean_penalty=True)
       else:
        aa,bb,an,bn,active,_=_center_pair(A,B);ch=np.divide((aa*bb).sum(0),an*bn,out=np.zeros(A.shape[1]),where=an*bn>0)*A.shape[1]/active.sum()
       score=None
       if arm!='retina32':
        h=heads[arm][head];features=np.r_[(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'],(ch-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta']);score=float(features@theta[:-1]+theta[-1])
       if pair==['index','index']:
        z=next(z for z in original['records'] if z['old_id']==r['old_id'] and z['representation']==arm and z['query']==qa and z['variant']==variant and z['support']=='common' and z['head']==head);e=float(abs(ch-z['channels']).max());err=max(err,e);assert e<1e-9
       rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,query=qa,variant=variant,pair=pair,support='common',count=len(A),head=head,channels=ch,mean_similarity=float(ch.mean()),score=score,accepted=None if score is None else bool(score>0),patch_key=key,map0_key=f'{ci}_{arm}_0_{pair[0]}',map1_key=f'{ci}_{arm}_1_{pair[1]}'))
with (OUT/'factorial_samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
out=dict(passed=True,records=rows,geometry=original['geometry'],original_pair_error=err,elapsed_seconds=time.monotonic()-started,samples_sha256=hashlib.sha256((OUT/'factorial_samples.npz').read_bytes()).hexdigest(),core_calls=0,new_fits=0,tracker_calls=0)
with (OUT/'factorial_result.json').open('x',encoding='utf8') as f:json.dump(cast(out),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print('FACTORIAL COMPLETE',len(rows))
