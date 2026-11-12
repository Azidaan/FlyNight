"""CPU audit of source/slot/pixel ledgers and exact target reuse; no neural call."""
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
p=read(OUT/'bank_protocol.json');r=read(OUT/'bank_result.json');assert r['passed'] and sha(OUT/'reference_bank.npz')==r['bank_sha256']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
total=0;maxnorm=0.;snapshot=read(OUT/'assignment_snapshots.json')
with np.load(OUT/'reference_bank.npz') as bank,np.load(ROOT/p['original_descriptor_path']) as old:
 assert set(bank.files)==set(p['frames'])
 for key,frame in p['frames'].items():
  a=bank[key];assert a.shape==(len(frame['boxes']),256) and np.isfinite(a).all();maxnorm=max(maxnorm,float(abs(np.linalg.norm(a.astype(float),axis=1)-1).max()));total+=len(a)
  for uses in frame['usages']:
   for use in uses:assert frame['frame']<next(s['frame'] for s in snapshot['snapshots'] if s['old_case']==use['old_case'])
 for reuse in p['target_reuse']:
  dst,src=reuse['target_slot'],reuse['source_slot'];assert np.array_equal(bank[dst[0]][dst[1]],old[src[0]][src[1]])
assert maxnorm<1e-5
pixelcount=0;cv2.setNumThreads(2)
for v in sorted({f['video'] for f in p['frames'].values()}):
 wanted={f['frame']:k for k,f in p['frames'].items() if f['video']==v};entry=p['frames'][next(iter(wanted.values()))];cap=cv2.VideoCapture(str(ROOT/entry['video_path']));assert cap.isOpened()
 try:
  for fi in range(max(wanted)+1):
   ok,image=cap.read();assert ok
   if fi in wanted:
    key=wanted[fi];assert hashlib.sha256(image.tobytes()).hexdigest()==next(f['pixel_sha256'] for f in r['frames'] if f['key']==key);pixelcount+=1
 finally:cap.release()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with (OUT/'bank_audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,reference_vectors=total,frames=pixelcount,target_vectors_exactly_reused=len(p['target_reuse']),max_unit_norm_error=maxnorm,bank_sha256=r['bank_sha256'],new_model_calls=0,scope='Independent shape, exacttargetreuse, pastframeusage, sourceSHA and sequentialCPUrawpixelhash audit. Neural features shared; not an independent model-forward validation.'),f,ensure_ascii=False,indent=2)
print('BANK_AUDIT_PASS',total,pixelcount,maxnorm)
