"""Prospective diagnostic of common brightness compensation; pixels only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,itertools,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_brightness_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_local_motion_gate_20261105 import image,features
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stimulus(g,c):
 rng=np.random.default_rng(c['seed']);patch=g[50:114,50:114].copy();res=[];kind=c['kind'];axis=c['axis']
 for t in range(11):
  x=g.copy()
  if kind in ('local','local_low','local_light','camera_object_light'):
   xx=200+(t if axis==0 else 0);yy=200+(t if axis==1 else 0);x[yy:yy+64,xx:xx+64]=patch
  if kind in ('camera','camera_light','camera_object_light'):
   dx=.5*t if axis==1 else 0.;dy=.5*t if axis==0 else 0.;x=cv2.warpAffine(x,np.float32([[1,0,dx],[0,1,dy]]),(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
  a=x.astype(np.float32);a=127.5+(.25 if kind=='local_low' else 1.)*(a-127.5)
  if kind in ('light','local_light','camera_light','camera_object_light'):a+=2*t
  if kind=='shadow':a[:,max(0,200-t*4):min(512,300+t*4)]-=20
  if c['noise']:a+=rng.normal(0,c['noise'],a.shape).astype(np.float32)
  res.append(np.rint(np.clip(a,0,255)).astype(np.uint8))
 return np.stack(res)
def corrected(frames):
 a=frames.astype(np.float32);d=np.diff(a,axis=0);med=np.median(d,axis=(1,2));res=np.abs(d-med[:,None,None]);return float(res.reshape(10,16,32,16,32).mean(axis=(2,4)).max()),med.tolist()
def register():
 OUT.mkdir(exist_ok=False);old=ROOT/'runs/flyvis_local_motion_gate_20261105';stress=ROOT/'runs/flyvis_local_gate_stress_20261105';base=json.loads((old/'report.json').read_text(encoding='utf-8'));olda=json.loads((old/'independent_audit.json').read_text(encoding='utf-8'));assert olda['passed'] and olda['report_sha256']==sha(old/'report.json');st=json.loads((stress/'report.json').read_text(encoding='utf-8'));sta=json.loads((stress/'independent_audit.json').read_text(encoding='utf-8'));assert sta['passed'] and sta['report_sha256']==sha(stress/'report.json');used=set()
 for f in ('runs/flyvis_offset_recovery2_20261104/protocol.json','runs/flyvis_tau_half_20261104/protocol.json','runs/flyvis_still_robust_gate_20261105/protocol.json','runs/flyvis_local_motion_gate_20261105/protocol.json','runs/flyvis_local_gate_stress_20261105/protocol.json'):used.update(json.loads((ROOT/f).read_text(encoding='utf-8'))['videos'])
 selected=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos'];vids=[x for x in selected['dev'] if x not in used and (ROOT/'data/bdd100k/videos/train'/(x+'.mov')).exists()][:8];assert len(set(vids))==8
 kinds=('still','light','shadow','local','local_low','local_light','camera','camera_light','camera_object_light');cases=[]
 for scene,noise,axis,kind in itertools.product(range(8),(0.,1.),(0,1),kinds):
  key=f'{vids[scene]}_{noise}_{axis}_{kind}';cases.append(dict(scene=scene,noise=noise,axis=axis,kind=kind,seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)))
 criterion='For each of still/light/shadow false motion <=10% and stationary accepted >=50%; for each of local/local_low/local_light/camera/camera_light/camera_object_light false stationary <=10%; unknown reported separately; audit reproducible. No tuning.'
 p=dict(id='flyvis_brightness_gate_v1',date=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=vids,cases=cases,threshold=base['threshold'],global_threshold=json.loads((old/'protocol.json').read_text(encoding='utf-8'))['global_threshold'],source_report_sha256=sha(old/'report.json'),stress_report_sha256=sha(stress/'report.json'),prior_feature_sha256=sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py'),script_sha256=sha(Path(__file__)),criterion=criterion,method='For every adjacent frame pair subtract signed median pixel difference over the whole image; compute maximum mean absolute residual over nonoverlapping 32x32 blocks and 10 transitions. Apply exactly the prior 1.247... threshold; do not tune. Frame std<=1 means uncertain. No labels used in inference.',controls='Prior uncorrected local gate and global phase correlation on same exact frames, plus never-update; all pixel-only without FlyVis.',literature='https://publications.tno.nl/publication/101316/mT237W/withagen-2007-global.pdf : general motivation for correcting global intensity, not a claim that our median algorithm is their implementation.',scope='Eight evaluation scenes new to preceding FlyVis offset/still/local experiments; BDD training video first frames, possible prior use elsewhere. Artificial 11-frame sequences; geometric object 64x64 at 1px/frame and camera .5px/frame, global additive lighting ramp and moving local shadow, noise sigma0/1. Frame clips may saturate. No independent real moving video, no FlyVis inference, no HOTA. One decision at end based on current/past images.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — локальное движение с компенсацией общей яркости',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Поможет ли вычитание общего изменения яркости отличить свет от движения без опасных пропусков?',hypothesis='Компенсация медианного изменения изображения может уменьшить ложные тревоги при общем изменении света; движение большой части изображения и локальные тени могут её нарушить.',expected='Проверка пиксельного правила до запуска FlyVis.',acceptance=[dict(criterion=criterion,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_brightness_gate_20261105/protocol.json',claim='Протокол и пороги до измерения')]));cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered',len(cases),'cases')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['script_sha256'] and sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py')==p['prior_feature_sha256'];cv2.setNumThreads(2);imgs=[image(x) for x in p['videos']];rows=[];start=time.perf_counter()
 for c in p['cases']:
  f=stimulus(imgs[c['scene']],c);previous=features(f);val,med=corrected(f);g=previous['global'];local='uncertain' if g['minimum_texture']<=1 else ('stationary' if previous['local_max_mean_difference']<p['threshold'] else 'moving');candidate='uncertain' if g['minimum_texture']<=1 else ('stationary' if val<p['threshold'] else 'moving');glob='uncertain' if not g['observable'] else ('stationary' if g['maximum_displacement']<p['global_threshold'] else 'moving');rows.append(dict(case=c,input_sha256=hashlib.sha256(f.tobytes()).hexdigest(),previous_features=previous,corrected_max=val,medians=med,local=local,candidate=candidate,global_decision=glob))
  if len(rows)==10:save(OUT/'timing.json',dict(first_10_seconds=time.perf_counter()-start,total=len(p['cases'])))
 summ={}
 for k in ('still','light','shadow','local','local_low','local_light','camera','camera_light','camera_object_light'):
  z=[x for x in rows if x['case']['kind']==k];summ[k]=dict(n=len(z),**{f'{m}_{d}':sum(x[m]==d for x in z) for m in ('candidate','local','global_decision') for d in ('stationary','moving','uncertain')})
 checks={}
 for k in ('still','light','shadow'):
  x=summ[k];checks[k+'_false_motion_le_10pct']=x['candidate_moving']/x['n']<=.1;checks[k+'_stationary_ge_50pct']=x['candidate_stationary']/x['n']>=.5
 for k in ('local','local_low','local_light','camera','camera_light','camera_object_light'):
  x=summ[k];checks[k+'_false_stationary_le_10pct']=x['candidate_stationary']/x['n']<=.1
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),summaries=summ,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start));print(json.dumps(dict(summaries=summ,checks=checks,seconds=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
