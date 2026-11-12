import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-12T07:40:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
import numpy as np
import cv2,torch
sys.path.append(str(ROOT/'runs/full_template_native_20261109'))
from sequential_views import SequentialViews
p=verify();torch.set_num_threads(2);values={};records=[];sources={};originals={};max_alt=0.
save('render_supervisor.json',observation.read())
def key(ci,frame,crop):return str(ci)+'_'+hashlib.sha256(json.dumps([frame,crop],separators=(',',':')).encode()).hexdigest()[:32]
def get(view,ci,frame,crop,now_frame):
    global max_alt
    k=key(ci,int(frame),list(crop))
    if k in values:return values[k]
    actual=view.render(frame,crop,now_frame);gray=view.frames.get(frame,now_frame);x1,y1,x2,y2=map(int,crop)
    # Independent neighborhood aggregation; decoder and resize are shared explicitly.
    resized=cv2.resize(gray[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    padded=np.pad(resized,6);alternate=np.array([np.sum(padded[y:y+13,x:x+13],dtype=np.float64)/169 for x,y in view.points],dtype=np.float16)
    error=float(np.max(abs(actual.astype(float)-alternate.astype(float))));max_alt=max(max_alt,error);assert error<=0.001,(ci,frame,crop,error)
    values[k]=actual;records.append(dict(key=k,case=ci,frame=int(frame),crop=list(crop),alternative_error=error,processed_frame_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
    return actual
for ci,c in enumerate(p['cases']):
    view=SequentialViews(c['case']['video']);terminal=c['sequence'][-1][0];rows=[]
    try:
        for j,(frame,crop) in enumerate(c['sequence']):
            actual=get(view,ci,frame,crop,terminal);expected=np.load(ROOT/c['views'][j],allow_pickle=False);assert np.array_equal(actual,expected),(ci,frame,'original retina mismatch');rows.append(actual)
        originals[ci]=np.stack(rows);sources.update(view.frames.sources)
    finally:view.close()
    del view;print('ORIGINAL_RETINA_PARITY',c['case']['old_id'],flush=True)
save('retinal_baseline_gate.json',dict(passed=True,all_cases_checked_before_changed_inputs=True,original_views=sum(len(c['views']) for c in p['cases'])))
inputs={};schedules=[]
for ci,c in enumerate(p['cases']):
    view=SequentialViews(c['case']['video']);terminal=c['sequence'][-1][0];requests=set();case_schedules=[]
    for mode in p['modes']:
        for ei,ep in enumerate(c['endpoints']):
            seq=[[int(f),list(crop)] for f,crop in c['sequence'][:ep['length']]];end_crop=seq[-1][1].copy()
            if mode=='last_pair_fixed':seq[-2][1]=end_crop.copy()
            elif mode=='all_fixed':seq=[[f,end_crop.copy()] for f,_ in seq]
            assert seq[-1]==c['sequence'][ep['length']-1]
            case_schedules.append(dict(case=ci,endpoint=ei,mode=mode,sequence=seq));requests.update((f,tuple(crop)) for f,crop in seq)
    try:
        for frame,crop in sorted(requests):get(view,ci,frame,list(crop),terminal)
        for s in case_schedules:
            hx=np.stack([values[key(ci,f,crop)] for f,crop in s['sequence']]);assert np.array_equal(hx[-1],originals[ci][len(hx)-1]);inputs[f"{s['mode']}_{ci}_{s['endpoint']}"]=hx
        sources.update(view.frames.sources)
    finally:view.close()
    del view;schedules+=case_schedules;print('FIXED_CROPS_RENDERED',c['case']['old_id'],flush=True)
for n,h in sources.items():assert p['inputs'].get(n)==h,(n,'decoder source was not preregistered')
verify()
with (OUT/'rendered_views.npz').open('xb') as f:np.savez_compressed(f,**values)
with (OUT/'rendered_inputs.npz').open('xb') as f:np.savez_compressed(f,**inputs)
save('render_manifest.json',dict(passed=True,created=now(),records=records,schedules=schedules,sources=sources,inputs_sha256=sha(OUT/'rendered_inputs.npz'),views_sha256=sha(OUT/'rendered_views.npz'),max_alternative_error=max_alt,terminal_inputs_exact=True,independence='Original cached retinal values are external witnesses. Alternative averaging uses NumPy float64 sums; native decoder, retina preprocessing and OpenCV resize shared. No flow estimation or tracking.'))
print('RENDER_COMPLETE',len(records),flush=True)
