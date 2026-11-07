"""Prospective stress of the frozen pixel gate; no FlyVis inference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, hashlib, itertools, json, sys, time
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_local_gate_stress_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_local_motion_gate_20261105 import image,features
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def make(g,c):
 rng=np.random.default_rng(c['seed']);patch=g[50:114,50:114].copy();result=[];axis=c['axis'];k=c['kind']
 for t in range(11):
  f=g.copy()
  if k in ('local_low','local_normal','camera_object'):
   x=200+(t if axis==0 else 0);y=200+(t if axis==1 else 0);f[y:y+64,x:x+64]=patch
  if k in ('camera','camera_object'):
   dx=.5*t if axis==1 else 0.;dy=.5*t if axis==0 else 0.;f=cv2.warpAffine(f,np.float32([[1,0,dx],[0,1,dy]]),(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
  a=f.astype(np.float32);contrast=.25 if k in ('local_low','still_low') else 1.;a=127.5+contrast*(a-127.5)
  if k=='light_ramp':a+=2.*t
  if c['noise']:a+=rng.normal(0,c['noise'],a.shape).astype(np.float32)
  result.append(np.rint(np.clip(a,0,255)).astype(np.uint8))
 return np.stack(result)
def register():
 OUT.mkdir(exist_ok=False); prev=ROOT/'runs/flyvis_local_motion_gate_20261105';r=json.loads((prev/'report.json').read_text(encoding='utf-8'));a=json.loads((prev/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'] and a['report_sha256']==sha(prev/'report.json');old=json.loads((prev/'protocol.json').read_text(encoding='utf-8'));used=set(old['videos'])
 for x in ('runs/flyvis_offset_recovery2_20261104/protocol.json','runs/flyvis_tau_half_20261104/protocol.json','runs/flyvis_still_robust_gate_20261105/protocol.json'):used.update(json.loads((ROOT/x).read_text(encoding='utf-8'))['videos'])
 sel=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos'];vids=[x for x in sel['dev'] if x not in used and (ROOT/'data/bdd100k/videos/train'/(x+'.mov')).exists()][:8];assert len(set(vids))==8
 kinds=('still','still_low','light_ramp','local_normal','local_low','camera','camera_object');cases=[]
 for scene,noise,axis,kind in itertools.product(range(8),(0.,1.),(0,1),kinds):
  key=f'{vids[scene]}_{noise}_{axis}_{kind}';cases.append(dict(scene=scene,noise=noise,axis=axis,kind=kind,seed=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)))
 criterion='All evaluation: false motion on still, still_low and light_ramp separately <=10%; false stationary on local_normal, local_low, camera and camera_object separately <=10%; still accepted >=50% in each static condition; unknown counted separately; independent feature audit. Small scene count makes this preliminary.'
 p=dict(id='flyvis_local_gate_stress_v1',date=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=vids,cases=cases,local_threshold=r['threshold'],global_threshold=old['global_threshold'],previous_report_sha256=sha(prev/'report.json'),previous_script_sha256=sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py'),script_sha256=sha(Path(__file__)),criterion=criterion,source='https://cecas.clemson.edu/~stb/klt/shi-tomasi-good-features-cvpr1994.pdf',scope='Fixed thresholds from previous pilot, 8 new-to-recent-pilots BDD training-video first-frame backgrounds. 11-frame synthetic sequences: still, quarter contrast, additive brightness ramp +2 intensity/frame, pasted moving 64x64 texture at 1px/frame, camera simulated translation .5px/frame and object plus camera. Two axes, sigma0/1 noise. No natural motion, future frames in per-frame decisions, FlyVis, HOTA, or guarantee of real-video performance. All 10 transitions are available only at sequence end. Prior experiments outside recent pilots may have used the videos.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — локальное движение при изменении условий',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Сохраняет ли локальное правило точность при слабом контрасте, изменении света и сдвиге камеры?',hypothesis='Изменение освещения и слабый контраст могут вызвать ложные тревоги и пропуски локального правила.',expected='Проверка ранее зафиксированных порогов на новых для недавних опытов сценах.',acceptance=[dict(criterion=criterion,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_local_gate_stress_20261105/protocol.json',claim='Протокол зафиксирован до расчёта')]))
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered',len(cases),'cases')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['script_sha256'] and sha(ROOT/'scripts/flyvis_local_motion_gate_20261105.py')==p['previous_script_sha256'];cv2.setNumThreads(2);imgs=[image(v) for v in p['videos']];rows=[];start=time.perf_counter()
 for c in p['cases']:
  frames=make(imgs[c['scene']],c);f=features(frames);g=f['global'];local='uncertain' if g['minimum_texture']<=1 else ('stationary' if f['local_max_mean_difference']<p['local_threshold'] else 'moving');glob='uncertain' if not g['observable'] else ('stationary' if g['maximum_displacement']<p['global_threshold'] else 'moving');rows.append(dict(case=c,input_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),features=f,local=local,global_decision=glob))
  if len(rows)==10:save(OUT/'timing.json',dict(first_ten_seconds=time.perf_counter()-start,total=len(p['cases'])))
 summ={}
 for k in ('still','still_low','light_ramp','local_normal','local_low','camera','camera_object'):
  z=[x for x in rows if x['case']['kind']==k];summ[k]=dict(n=len(z),**{f'{m}_{d}':sum(x[m]==d for x in z) for m in ('local','global_decision') for d in ('stationary','moving','uncertain')})
 checks={}
 for k in ('still','still_low','light_ramp'):
  x=summ[k];checks[k+'_false_motion_le_10pct']=x['local_moving']/x['n']<=.1;checks[k+'_stationary_recall_ge_50pct']=x['local_stationary']/x['n']>=.5
 for k in ('local_normal','local_low','camera','camera_object'):
  x=summ[k];checks[k+'_false_stationary_le_10pct']=x['local_stationary']/x['n']<=.1
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),summaries=summ,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start));print(json.dumps({'summaries':summ,'checks':checks,'elapsed':time.perf_counter()-start},ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
