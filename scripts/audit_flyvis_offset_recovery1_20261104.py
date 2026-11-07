"""Независимая повторная проверка смещения, разбиения и итоговых ошибок."""
from pathlib import Path
import hashlib,json
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery1_20261104'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json')
 assert not set(p['videos'][:4])&set(p['videos'][4:]);mask=np.load(OUT/'geometry.npz')['mask'];count=int(mask.sum());data=[]
 original={}
 for i,video in enumerate(p['videos']):
  path=ROOT/'data/bdd100k/videos/train'/(video+'.mov');source=json.loads((OUT/'source_images.json').read_text(encoding='utf-8'))['sources'][i];assert sha(path)==source['source_sha256']
  cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release();assert ok;original[i]=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 for case in p['cases']:
  q={}
  inp=OUT/'inputs'/(case['id']+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256']
  frames=np.load(inp)['frames'];gray=original[case['scene']];h,w=gray.shape;x0=(w-512)//2;y0=h-512 if case['axis']==1 and case['speed']==4 else (h-512)//2
  for j in (0,20,21,54,55,78):
   first=31 if case['start']=='still' else 1;steps=max(0,min(j,54)-first+1);steps-=max(0,j-54);shift=steps*case['speed']
   if case['start']=='moving' and case['speed']==4:shift-=2*min(j,4)
   x=x0-(shift if case['axis']==0 else 0);y=y0-(shift if case['axis']==1 else 0)
   expected=np.rint(np.clip(127.5+case['contrast']*(gray[y:y+512,x:x+512].astype(np.float32)-127.5),0,255)).astype(np.uint8)
   assert np.array_equal(frames[j],expected),(case['id'],j)
  for arm in ('stock','fast'):
   f=OUT/arm/'outputs'/(case['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256']
   x=np.load(f)['flow'][0,1];x=np.einsum('tnc,n->tc',x,mask.astype(float))/count
   q[arm+'_raw']=x;q[arm+'_subtract']=x-sum(x[11:21])/10
  target=np.zeros((79,2));start=31 if case['start']=='still' else 1;target[start:55,case['axis']]=case['speed'];target[55:,case['axis']]=-case['speed']
  if case['start']=='moving' and case['speed']==4:target[1:5,case['axis']]=2
  data.append((case,q,target))
 diff=0.
 for arm in ('stock_raw','stock_subtract','fast_raw','fast_subtract'):
  x=np.concatenate([z[1][arm][21:79].flatten() for z in data if z[0]['split']=='calibration']);y=np.concatenate([z[2][21:79].flatten() for z in data if z[0]['split']=='calibration'])
  g=float(np.linalg.lstsq(x[:,None],y,rcond=None)[0][0]);diff=max(diff,abs(g-r['gains'][arm]));assert abs(g-r['gains'][arm])<1e-9
 for c,v,target in data:
  result=next(x for x in r['rows'] if x['case']['id']==c['id'])['results']
  for name,q in v.items():
   for suffix,g in [('uncalibrated',1.),('calibrated',r['gains'][name])]:
    z=q*g;dv=z[21:79]-target[21:79];e=float(np.sqrt(dv[:,0]**2+dv[:,1]**2).mean());key=name+'_'+suffix
    diff=max(diff,abs(e-result[key]['epe']));assert abs(e-result[key]['epe'])<1e-9
    good=z[55:79,c['axis']]<0; hits=[i for i in range(22) if good[i] and good[i+1] and good[i+2]]
    assert (hits[0] if hits else None)==result[key]['latency']
 audit=dict(passed=True,cases=len(data),distinct_training_evaluation_scenes=True,max_abs_diff=diff,report_sha256=sha(OUT/'report.json'),
  scope='Все 128 случаев и обе модели; SHA входов и исходных видео, 6 независимо построенных кадров каждого случая, повтор коэффициентов, ошибок и знаковой задержки. Повтор нейросети не заявляется.')
 (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
