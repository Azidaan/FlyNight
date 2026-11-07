"""Prospective local brightness-order diagnostic, no FlyVis inference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,itertools,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_order_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_local_motion_gate_20261105 import image,features
from scripts.flyvis_brightness_gate_20261105 import corrected
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stimulus(g,c):
 rng=np.random.default_rng(c['seed']);patch=g[50:114,50:114].copy();res=[];kind=c['kind'];axis=c['axis']
 for t in range(11):
  z=g.copy()
  if kind in ('object','object_low','object_light','object_shadow'):
   x=200+(t if axis==0 else 0);y=200+(t if axis==1 else 0);z[y:y+64,x:x+64]=patch
  if kind=='camera':
   dx=.5*t if axis==1 else 0.;dy=.5*t if axis==0 else 0.;z=cv2.warpAffine(z,np.float32([[1,0,dx],[0,1,dy]]),(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
  if kind=='flat':z=np.full_like(z,127)
  a=z.astype(np.float32);a=127.5+(.25 if kind=='object_low' else 1.)*(a-127.5)
  if kind in ('light','object_light'):a+=2*t
  if kind in ('shadow','object_shadow'):a[:,max(0,200-t*4):min(512,300+t*4)]-=20
  if kind=='soft_shadow':
   xx=np.arange(512,dtype=np.float32);factor=np.clip((xx-(180-t*2))/24,0,1)*np.clip(((330+t*2)-xx)/24,0,1);a-=20*factor[None,:]
  if c['noise']:a+=rng.normal(0,c['noise'],a.shape).astype(np.float32)
  res.append(np.rint(np.clip(a,0,255)).astype(np.uint8))
 return np.stack(res)
def order_score(frames):
 # A strict sign flip, for four fixed offsets, never counts a near-zero pair.
 # Edge padding preserves 512x512 tiles, edge comparisons are ignored.
 maxscore=0.;minimum_active=1.; offsets=((0,1),(1,0),(0,2),(2,0))
 for i in range(1,11):
  a=frames[i-1].astype(np.int16);b=frames[i].astype(np.int16);flips=np.zeros((512,512),np.uint8);active=0;total=0
  for dy,dx in offsets:
   da=a[dy:,dx:]-a[:512-dy,:512-dx];db=b[dy:,dx:]-b[:512-dy,:512-dx];good=(np.abs(da)>3)&(np.abs(db)>3);flip=good&((da>0)!=(db>0));flips[:512-dy,:512-dx]+=flip;active+=int(good.sum());total+=good.size
  minimum_active=min(minimum_active,active/total);tile=flips.reshape(16,32,16,32).sum(axis=(1,3))/(4*32*32);maxscore=max(maxscore,float(tile.max()))
 return maxscore,minimum_active
def register():
 OUT.mkdir(exist_ok=False);used=set()
 for x in ('flyvis_offset_recovery2_20261104','flyvis_tau_half_20261104','flyvis_still_robust_gate_20261105','flyvis_local_motion_gate_20261105','flyvis_local_gate_stress_20261105','flyvis_brightness_gate_20261105'):used.update(json.loads((ROOT/'runs'/x/'protocol.json').read_text(encoding='utf-8'))['videos'])
 selection=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos'];c=[v for v in selection['train'] if v not in used and (ROOT/'data/bdd100k/videos/train'/(v+'.mov')).exists()][:5];e=[v for v in selection['dev'] if v not in used and (ROOT/'data/bdd100k/videos/train'/(v+'.mov')).exists()][:3];vids=c[:4]+e+c[4:5];assert len(vids)==8 and len(set(vids))==8
 kinds=('still','light','shadow','soft_shadow','object','object_low','object_light','object_shadow','camera','flat');cases=[]
 for scene,noise,axis,kind in itertools.product(range(8),(0.,1.),(0,1),kinds):
  key=f'{vids[scene]}_{noise}_{axis}_{kind}';cases.append(dict(scene=scene,noise=noise,axis=axis,kind=kind,seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)))
 prev=ROOT/'runs/flyvis_brightness_gate_20261105';r=json.loads((prev/'report.json').read_text(encoding='utf-8'));a=json.loads((prev/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'] and a['report_sha256']==sha(prev/'report.json');q=json.loads((prev/'protocol.json').read_text(encoding='utf-8'))
 criterion='For each of still/light/shadow/soft_shadow: candidate false motion <=10% and stationary accepted >=50%; for each object/object_low/object_light/object_shadow/camera: false stationary <=10%; flat must abstain >=90%; independent audit; experimental preliminary criterion, no guarantees.'
 p=dict(id='flyvis_order_gate_v1',date=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=vids,cases=cases,local_threshold=q['threshold'],global_threshold=q['global_threshold'],prior_report_sha256=sha(prev/'report.json'),prior_script_sha256=sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py'),brightness_script_sha256=sha(ROOT/'scripts/flyvis_brightness_gate_20261105.py'),script_sha256=sha(Path(__file__)),criterion=criterion,algorithm='Across 10 adjacent pairs, compare 4 fixed offsets (1,0),(0,1),(2,0),(0,2). Count strict sign reversals when both absolute differences >3 intensity levels; score maximum block fraction over nonoverlapping 32x32 blocks and 10 pairs. At least 1% pixel pairs must be above 3 in both frames in every pair or abstain. Threshold = max score among observable calibration still/light/shadow/soft_shadow + 1e-9; no fitting on evaluation.',controls='Old local, median-corrected local, global phase and never update, exact same frames. Pixel-only, no FlyVis.',literature='https://courses.cs.duke.edu/spring06/cps296.1/handouts/Zabih%20Woodfill%201994.pdf ; https://iacl.ece.jhu.edu/proceedings/cvpr2008/papers/328.pdf : local rank comparison and statistical uncertainty motivate the test; this simple sign inversion implementation is not either paper implementation.',scope='4 calibration and 4 disjoint evaluation images, BDD training videos. Three eval drawn from dev subset and one from held-out train subset; prior use outside recent experiments possible. Artificial 11-frame motion at 1px/frame, dim scene, 0.5px camera shift, noise sigma0/1, linear light, hard/soft shadow, object in shadow, flat scene. Geometry labels only for scoring. No real moving-video test, no FlyVis/HOTA. One decision at frame 10.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — локальный порядок яркости при движении и тени',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Можно ли отличать локальное движение от изменений освещения по порядку яркости?',hypothesis='Локальные инверсии порядка менее чувствительны к некоторым изменениям яркости, но могут пропускать слабые объекты и реагировать на край тени.',expected='Предварительное сравнение признака на отдельных сценах без запуска FlyVis.',acceptance=[dict(criterion=criterion,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_order_gate_20261105/protocol.json',claim='Протокол до вычисления')]));cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered',len(cases),'cases')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['script_sha256'] and sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py')==p['prior_script_sha256'] and sha(ROOT/'scripts/flyvis_brightness_gate_20261105.py')==p['brightness_script_sha256'];cv2.setNumThreads(2);images=[image(v) for v in p['videos']];rows=[];start=time.perf_counter()
 for c in p['cases']:
  f=stimulus(images[c['scene']],c);prior=features(f);norm,med=corrected(f);score,active=order_score(f);g=prior['global'];local='uncertain' if g['minimum_texture']<=1 else ('stationary' if prior['local_max_mean_difference']<p['local_threshold'] else 'moving');median='uncertain' if g['minimum_texture']<=1 else ('stationary' if norm<p['local_threshold'] else 'moving');glob='uncertain' if not g['observable'] else ('stationary' if g['maximum_displacement']<p['global_threshold'] else 'moving');rows.append(dict(case=c,input_sha256=hashlib.sha256(f.tobytes()).hexdigest(),score=score,minimum_active=active,prior_features=prior,median_max=norm,local=local,median=median,global_decision=glob))
  if len(rows)==10:save(OUT/'timing.json',dict(seconds10=time.perf_counter()-start,total=len(p['cases'])))
 stationary=('still','light','shadow','soft_shadow');cal=[x['score'] for x in rows if x['case']['scene']<4 and x['case']['kind'] in stationary and x['minimum_active']>=.01];assert cal;threshold=max(cal)+1e-9
 for x in rows:x['candidate']='uncertain' if x['minimum_active']<.01 else ('stationary' if x['score']<threshold else 'moving')
 groups={}
 for kind in stationary+('object','object_low','object_light','object_shadow','camera','flat'):
  z=[x for x in rows if x['case']['scene']>=4 and x['case']['kind']==kind];groups[kind]=dict(n=len(z),**{f'{m}_{v}':sum(x[m]==v for x in z) for m in ('candidate','local','median','global_decision') for v in ('stationary','moving','uncertain')})
 checks={}
 for k in stationary:
  x=groups[k];checks[k+'_false_motion_le_10pct']=x['candidate_moving']/x['n']<=.1;checks[k+'_accepted_ge_50pct']=x['candidate_stationary']/x['n']>=.5
 for k in ('object','object_low','object_light','object_shadow','camera'):
  x=groups[k];checks[k+'_false_still_le_10pct']=x['candidate_stationary']/x['n']<=.1
 checks['flat_uncertain_ge_90pct']=groups['flat']['candidate_uncertain']/groups['flat']['n']>=.9
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),threshold=threshold,calibration_count=len(cal),groups=groups,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start));print(json.dumps(dict(threshold=threshold,groups=groups,checks=checks,elapsed=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
