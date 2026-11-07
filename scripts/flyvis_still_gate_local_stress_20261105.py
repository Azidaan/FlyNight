"""Дополнительный постфактум контроль локального движения: не новый тест FlyVis."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_robust_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.flyvis_still_robust_gate_20261105 import extract
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));audit=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'));assert audit['passed'] and audit['report_sha256']==sha(OUT/'report.json');cv2.setNumThreads(2);rows=[]
 for vid in p['videos'][4:]:
  source=ROOT/'data/bdd100k/videos/train'/(vid+'.mov');cap=cv2.VideoCapture(str(source));ok,frame=cap.read();cap.release();assert ok;gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);h,w=gray.shape;image=gray[(h-512)//2:(h-512)//2+512,(w-512)//2:(w-512)//2+512]
  for size in (64,128):
   patch=image[50:50+size,50:50+size].copy()
   for axis in (0,1):
    for noise in (0.,1.):
     seed=int(hashlib.sha256(f'{vid}_{size}_{axis}_{noise}'.encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);frames=[]
     for t in range(11):
      f=image.copy();x=200+(t if axis==0 else 0);y=200+(t if axis==1 else 0);f[y:y+size,x:x+size]=patch
      if noise:f=np.rint(np.clip(f.astype(np.float32)+rng.normal(0,noise,f.shape),0,255)).astype(np.uint8)
      frames.append(f)
     frames=np.stack(frames);assert not np.array_equal(frames[0],frames[-1]);features=extract(frames,.2,1.);decision='uncertain' if not features['observable'] else ('stationary' if features['maximum_displacement']<r['threshold'] else 'moving')
     rows.append(dict(video=vid,source_sha256=sha(source),size=size,axis=axis,noise_sigma=noise,actual_object_motion_px_per_frame=1,features=features,decision=decision,frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest()))
 result=dict(complete=True,post_hoc=True,cases=len(rows),false_stationary=sum(x['decision']=='stationary' for x in rows),detected_moving=sum(x['decision']=='moving' for x in rows),uncertain=sum(x['decision']=='uncertain' for x in rows),rows=rows,
  protocol_report_sha256=sha(OUT/'report.json'),source_sha256=sha(Path(__file__)),scope='Создано после основного результата как контрпример: искусственный движущийся квадрат на неподвижном фоне. Не измерена частота на реальном видео; повтор FlyVis не проводился.')
 (OUT/'local_motion_stress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
