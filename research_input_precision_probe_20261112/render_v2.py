import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
end=datetime.fromisoformat('2026-11-12T18:15:00+03:00');remaining=(end-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=min(1200,remaining))
import numpy as np,cv2,torch,math
sys.path.append(str(ROOT/'runs/full_template_native_20261109'))
from sequential_views import SequentialViews
p=verify();torch.set_num_threads(2);cv2.setNumThreads(2);inputs={};records=[];sources={}
save('render_v2_supervisor.json',observation.read())
for ci,c in enumerate(p['cases']):
 view=SequentialViews(c['case']['video']);terminal=c['sequence'][-1][0];f32=[];old=[]
 try:
  for j,(frame,crop) in enumerate(c['sequence']):
   gray=view.frames.get(frame,terminal);x1,y1,x2,y2=map(int,crop)
   resized=cv2.resize(gray[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
   x=torch.from_numpy(resized).reshape(1,1,391,391)
   with torch.inference_mode():y=torch.nn.functional.conv2d(torch.nn.functional.pad(x,(6,6,6,6)),view.kernel,view.bias)/169
   actual=y[0,0,view.py,view.px].numpy().copy();assert actual.dtype==np.float32 and actual.shape==(721,) and np.isfinite(actual).all()
   quantized=actual.astype(np.float16);saved=np.load(ROOT/c['views'][j],allow_pickle=False);native=view.render(frame,crop,terminal)
   assert np.array_equal(quantized,saved) and np.array_equal(native,saved),(ci,frame,'input restoration mismatch')
   padded=np.pad(resized,6);alternate=np.array([math.fsum(float(v) for v in padded[yy:yy+13,xx:xx+13].ravel())/169 for xx,yy in view.points])
   independent_error=float(abs(actual.astype(float)-alternate).max())
   scalar=np.zeros(721,dtype=np.float32)
   for dy in range(13):
    for dx in range(13):scalar+=padded[view.points[:,1]+dy,view.points[:,0]+dx]
   scalar/=np.float32(169)
   fp32_error=float(abs(actual-scalar).max());assert fp32_error<=1e-6,('float32_reference',ci,frame,fp32_error)
   original=saved.astype(np.float32);error=actual.astype(float)-original.astype(float)
   records.append(dict(case=ci,old_id=c['case']['old_id'],observation=j,frame=frame,crop=crop,processed_frame_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),float32_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),float16_exact=True,changed_cells=int(np.count_nonzero(error)),max_error=float(abs(error).max()),rms_error=float(np.sqrt(np.mean(error**2))),independent_averaging_error=independent_error,independent_float32_error=fp32_error))
   f32.append(actual);old.append(original)
   (OUT/'status.json').write_text(json.dumps(dict(stage='render',old_id=c['case']['old_id'],observation=j+1,total=len(c['sequence']),updated=now())),encoding='utf8')
  inputs[f'index_{ci}']=np.stack(old);inputs[f'identity_{ci}']=np.stack(old).copy();inputs[f'precision32_{ci}']=np.stack(f32);sources.update(view.frames.sources)
 finally:view.close()
 print('RENDERED',c['case']['old_id'],flush=True)
for n,h in sources.items():assert p['inputs'].get(n)==h,(n,'decoder provenance')
verify()
with (OUT/'expanded_inputs.npz').open('xb') as f:np.savez_compressed(f,**inputs)
save('render_manifest.json',dict(passed=True,records=records,source_hashes=sources,inputs_sha256=sha(OUT/'expanded_inputs.npz'),all_original_views_exact=True,identity_exact=True,cpu_only=True))
print('RENDER_COMPLETE',len(records),flush=True)
