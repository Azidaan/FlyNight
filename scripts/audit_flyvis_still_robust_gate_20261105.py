"""Повтор генерации каждого входа и всех решений без готовой функции пилота."""
from pathlib import Path
import hashlib,json,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_robust_gate_20261105'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert p['audit_sha256']==sha(Path(__file__)) and r['protocol_sha256']==sha(OUT/'protocol.json');cv2.setNumThreads(2)
 images={}
 for i,vid in enumerate(p['videos']):
  path=ROOT/'data/bdd100k/videos/train'/(vid+'.mov');assert sha(path)==r['sources'][i]['sha256'];cap=cv2.VideoCapture(str(path));ok,img=cap.read();cap.release();assert ok;images[i]=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);assert hashlib.sha256(images[i].tobytes()).hexdigest()==r['sources'][i]['frame_sha256']
 maxdiff=0.;start=time.perf_counter();features=[]
 for c,row in zip(p['cases'],r['rows']):
  assert c==row['case'];gray=images[c['scene']] if c['texture']=='natural' else np.full_like(images[c['scene']],127);h,w=gray.shape;rng=np.random.default_rng(c['seed']);stack=[]
  for t in range(11):
   step=t if c['trajectory']=='constant' else min(t,10-t);dx=step*c['speed'] if c['axis']==0 else 0.;dy=step*c['speed'] if c['axis']==1 else 0.
   z=cv2.getRectSubPix(gray,(512,512),(w/2-dx,h/2-dy)).astype(np.float32);z=127.5+c['contrast']*(z-127.5)
   if c['noise_sigma']:z+=rng.normal(0,c['noise_sigma'],size=(512,512)).astype(np.float32)
   stack.append(np.rint(np.clip(z,0,255)).astype(np.uint8))
  a=np.stack(stack);assert hashlib.sha256(a.tobytes()).hexdigest()==row['frames_sha256']
  exact=bool(np.all(a[1:]==a[:-1]));delta=[];response=[];texture=[]
  for j in range(10):
   x=a[j].astype(np.float32);y=a[j+1].astype(np.float32);(u,v),quality=cv2.phaseCorrelate(x,y);delta.append((u*u+v*v)**.5);response.append(quality);texture.append(min(x.std(),y.std()))
  f=row['features'];vals={'maximum_displacement':max(delta),'minimum_response':min(response),'minimum_texture':min(texture)}
  for k,x in vals.items():diff=abs(float(x)-f[k]);maxdiff=max(maxdiff,diff);assert diff<1e-8,(c['id'],k,diff)
  obs=min(response)>=.2 and min(texture)>1.;assert exact==f['exact'] and bool(obs)==f['observable'];features.append((c,vals['maximum_displacement'],bool(obs),exact,row))
  if len(features)%40==0:(OUT/'audit_status.json').write_text(json.dumps(dict(done=len(features),total=len(p['cases']),elapsed_seconds=time.perf_counter()-start))+'\n')
 train=[x[1] for x in features if x[0]['split']=='calibration' and x[0]['speed']>0 and x[2]];threshold=.5*min(train) if train else 0.;assert abs(threshold-r['threshold'])<1e-8
 for c,dist,obs,exact,row in features:
  decision='uncertain' if not obs else ('stationary' if dist<threshold else 'moving');assert decision==row['decision'] and ('stationary' if exact else 'uncertain')==row['exact_decision']
 out=dict(passed=True,cases=len(features),all_input_hashes=True,all_features_and_decisions=True,calibration_excludes_evaluation=True,max_abs_diff=maxdiff,
  report_sha256=sha(OUT/'report.json'),scope='Отдельное воспроизведение синтетических кадров, признаков и решений; не новый независимый набор данных.')
 (OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
