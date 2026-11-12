"""CPU-only independent shape/slot/source/rawpixel audit of shared A bank."""
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SHARED=OUT.parent/'shared'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
p=read(SHARED/'extraction_protocol.json');r=read(SHARED/'extraction_result.json');plan=read(ROOT/p['plan']);runtime=read(ROOT/p['runtime']);assert r['passed'] and sha(SHARED/'descriptors_A.npz')==r['bank_sha256']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
total=0;normerror=0.;slots=0
with np.load(SHARED/'descriptors_A.npz') as bank:
 assert set(bank.files)==set(plan['frames'])
 for key,frame in plan['frames'].items():
  a=bank[key];assert a.shape==(len(frame['boxes']),256) and np.isfinite(a).all();total+=len(a);normerror=max(normerror,float(abs(np.linalg.norm(a.astype(float),axis=1)-1).max()))
 for event in runtime['events']:
  assert event['video'] not in runtime['excluded_videos']
  for row in event['rows']:
   ref=row['reference'];assert ref and ref['frame']<event['frame'];key,i=ref['slot'];assert plan['frames'][key]['frame']==ref['frame'] and plan['frames'][key]['boxes'][i]==list(map(float,ref['box']));assert bank[key][i].shape==(256,);slots+=1
  for raw in event['raw_candidates']:
   key,i=raw['current_slot'];assert plan['frames'][key]['frame']==event['frame'] and plan['frames'][key]['boxes'][i]==raw['box'];assert bank[key][i].shape==(256,);slots+=1
assert total==p['rois'] and normerror<1e-5
pixels=0;cv2.setNumThreads(2)
for video in sorted({f['video'] for f in plan['frames'].values()}):
 wanted={f['frame']:k for k,f in plan['frames'].items() if f['video']==video};entry=plan['frames'][next(iter(wanted.values()))];cap=cv2.VideoCapture(str(ROOT/entry['video_path']));assert cap.isOpened()
 try:
  for fi in range(max(wanted)+1):
   ok,image=cap.read();assert ok
   if fi in wanted:
    key=wanted[fi];assert hashlib.sha256(image.tobytes()).hexdigest()==next(f['pixel_sha256'] for f in r['frames'] if f['key']==key);pixels+=1
 finally:cap.release()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with (SHARED/'extraction_audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,frames=pixels,rois=total,slots_checked=slots,max_unit_norm_error=normerror,bank_sha256=r['bank_sha256'],model_calls=0,scope='IndependentCPU rawpixel/sourceSHA/slot/shape/unitnorm audit. Savedneuralvectors shared; no independentneuralforward. No repairoutcome claim.'),f,ensure_ascii=False,indent=2)
print('A_BANK_AUDIT_PASS',pixels,total,slots,normerror)
