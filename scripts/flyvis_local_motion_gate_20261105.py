"""Prospective local-change gate diagnostic; operates on pixels only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, hashlib, itertools, json, sys, time
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/flyvis_local_motion_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_still_robust_gate_20261105 import extract as global_extract
def digest(x):return hashlib.sha256(x).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def image(vid):
 p=ROOT/'data/bdd100k/videos/train'/(vid+'.mov'); cap=cv2.VideoCapture(str(p));ok,frame=cap.read();cap.release();assert ok;g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);h,w=g.shape;return g[(h-512)//2:(h-512)//2+512,(w-512)//2:(w-512)//2+512]
def frames(g,c):
 rng=np.random.default_rng(c['seed']); patch=g[50:50+c['size'],50:50+c['size']].copy() if c['size'] else None; arr=[]
 for t in range(11):
  if c['kind']=='global': f=cv2.getRectSubPix(g,(512,512),(255.5-(t if c['axis']==0 else 0),255.5-(t if c['axis']==1 else 0)))
  else:
   f=g.copy()
   if c['kind']=='local':
    x=200+(t if c['axis']==0 else 0);y=200+(t if c['axis']==1 else 0);f[y:y+c['size'],x:x+c['size']]=patch
  f=f.astype(np.float32)
  if c['noise']:f+=rng.normal(0,c['noise'],f.shape).astype(np.float32)
  arr.append(np.rint(np.clip(f,0,255)).astype(np.uint8))
 return np.stack(arr)
def features(arr):
 a=arr.astype(np.float32);diff=np.abs(np.diff(a,axis=0)); tile=diff.reshape(10,16,32,16,32).mean(axis=(2,4));loc=float(tile.max());f=global_extract(arr,.2,1.);return {'local_max_mean_difference':loc,'global':f}
def register():
 OUT.mkdir(exist_ok=False); old=ROOT/'runs/flyvis_still_robust_gate_20261105'; global_report=json.loads((old/'report.json').read_text(encoding='utf-8')); selection=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos']; used=set()
 for x in ('runs/flyvis_offset_recovery2_20261104/protocol.json','runs/flyvis_tau_half_20261104/protocol.json','runs/flyvis_still_robust_gate_20261105/protocol.json'):used.update(json.loads((ROOT/x).read_text(encoding='utf-8'))['videos'])
 vids=[]
 for split in ('train','dev'):vids += [x for x in selection[split] if x not in used and (ROOT/'data/bdd100k/videos/train'/(x+'.mov')).exists()][:4]
 assert len(vids)==8 and len(set(vids))==8; cases=[]
 for scene,noise,axis,kindsize in itertools.product(range(8),(0.,1.),(0,1),(('still',0),('local',32),('local',64),('local',128),('global',0))):
  kind,size=kindsize; key=f'{vids[scene]}_{noise}_{axis}_{kind}_{size}';cases.append(dict(scene=scene,noise=noise,axis=axis,kind=kind,size=size,seed=int(digest(key.encode())[:16],16)))
 p=dict(id='flyvis_local_motion_gate_v1',date=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),videos=vids,cases=cases,split='first four train scenes calibration; remaining four dev scenes evaluation; all videos from BDD train, may have appeared elsewhere in project',candidate='maximum mean absolute consecutive-frame difference over nonoverlapping 32x32 tiles and 10 transitions; frame std <=1 yields uncertain; local decision threshold calibrated from calibration still cases only as maximum plus 1e-6',controls='previous global phase correlation with frozen threshold and its confidence floors; never-update; both use same input and are without FlyVis',global_threshold=global_report['threshold'],previous_report_sha256=sha(old/'report.json'),acceptance='Evaluation: local false stationary on all local-motion cases <= half global false stationary count; local false motion among still <= 10%; local still accepted >= 50%; independent replay matches; no claim of full solution.',caveat='Synthetic pasted textured moving square with translation 1 pixel/frame; 4 evaluation scenes only, repeated cases dependent; no FlyVis or HOTA, no natural moving video; illumination changes not evaluated; initial object patch may create a first-frame boundary.',script_sha256=sha(Path(__file__)))
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — локальный признак движения на неподвижном фоне',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Заметит ли локальный признак малый объект, не принимая шум за движение?',hypothesis='Максимум локальных изменений может обнаружить объект при неподвижном фоне, но может давать ложные тревоги от шума и света.',expected='Предварительная проверка пиксельного признака без запуска FlyVis.',acceptance=[dict(criterion=p['acceptance'],met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_local_motion_gate_20261105/protocol.json',claim='Протокол зарегистрирован до расчёта')]))
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered',len(cases),'cases')
def run(replay=False):
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert p['script_sha256']==sha(Path(__file__));cv2.setNumThreads(2);images=[image(x) for x in p['videos']];rows=[];start=time.perf_counter()
 for c in p['cases']:
  a=frames(images[c['scene']],c);f=features(a);rows.append(dict(case=c,features=f,input_sha256=digest(a.tobytes())))
  if len(rows)==10 and not replay:
   elapsed=time.perf_counter()-start;save(OUT/'timing.json',dict(first_ten_seconds=elapsed,forecast_seconds=elapsed*len(p['cases'])/10,scope='calculation only; excludes checks and documentation'))
 still=[x['features']['local_max_mean_difference'] for x in rows if x['case']['scene']<4 and x['case']['kind']=='still'];threshold=max(still)+1e-6
 for x in rows:
  f=x['features'];g=f['global'];x['local']='uncertain' if g['minimum_texture']<=1 else ('stationary' if f['local_max_mean_difference']<threshold else 'moving');x['global']='uncertain' if not g['observable'] else ('stationary' if g['maximum_displacement']<p['global_threshold'] else 'moving')
 evaluate=[x for x in rows if x['case']['scene']>=4];summ={}
 for k in ('still','local','global'):
  z=[x for x in evaluate if x['case']['kind']==k];summ[k]={'n':len(z),**{f'{m}_{d}':sum(x[m]==d for x in z) for m in ('local','global') for d in ('stationary','moving','uncertain')}}
 m=summ['local']; st=summ['still'];checks=dict(local_false_stationary_at_most_half_global=m['local_stationary']<=m['global_stationary']/2,local_false_motion_still_le_10pct=st['local_moving']/st['n']<=.1,local_still_accepted_ge_50pct=st['local_stationary']/st['n']>=.5)
 result=dict(complete=True,threshold=threshold,summaries=summ,checks=checks,preliminary_gate=all(checks.values()),rows=rows,protocol_sha256=sha(OUT/'protocol.json'),elapsed_seconds=time.perf_counter()-start,caveat=p['caveat'])
 if replay:
  orig=json.loads((OUT/'report.json').read_text(encoding='utf-8'));passed=orig['threshold']==threshold and orig['rows']==rows and orig['summaries']==summ and orig['checks']==checks;save(OUT/'replay_audit.json',dict(passed=passed,report_sha256=sha(OUT/'report.json'),scope='Separate repeat of same implementation, not independent implementation or independent dataset'));assert passed
 else:save(OUT/'report.json',result)
 print(json.dumps({'replay':replay,'summaries':summ,'checks':checks,'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','run','replay']);arg=ap.parse_args();register() if arg.stage=='register' else run(arg.stage=='replay')
