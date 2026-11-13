"""Numerically bind saved retinal inputs, sequential MOV frames and mask PNGs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time,pickle
from pathlib import Path
ROOT=Path('E:/Dissertation');OUT=Path(__file__).resolve().parent
for v in ('TEMP','TMP','TMPDIR'):os.environ[v]=str(OUT/'tmp')
sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch,psutil
torch.set_num_threads(2);cv2.setNumThreads(2)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    with (OUT/n).open('x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
p=read(OUT/'protocol.json');old=ROOT/'runs/identity_assignment_20261108/full_precision'
legacy=read(old/'protocol.json');vid=p['video'];entry=legacy['videos'][vid]
video=ROOT/entry['video_path'];meta=old/'processed'/(vid+'.checkpoints')/'001100.json'
checkpoint=read(meta);statefile=meta.parent/checkpoint['state_file']
assert sha(video)==entry['sources'][entry['video_path']]==checkpoint['video_sha256']
assert sha(statefile)==checkpoint['state_sha256']
sources=legacy['source_hashes']
retinapath='tracking/retina.py'
assert sources[retinapath]==sha(ROOT/retinapath),'Historical retina source mismatch'
inputs={str(q.relative_to(ROOT)):sha(q) for q in [video,meta,statefile,ROOT/retinapath,OUT/'protocol.json',OUT/'result.json',OUT/'predictions.npz',Path(__file__)]}
save('alignment_protocol.json',dict(question='Do saved retinal values and current mask PNGs denote the same sequential raw frames?',
    method='Sequential VideoCapture.read from raw frame0; restore verified Lamina state at1100, process1100..1165. Independently regenerate13x13 CPU BoxEye for every actually used retinal record. Compare float16 values exactly and three mask-source PNGs pixel-for-pixel. No new neural forward, no reader refit, no mask changes.',
    inputs=inputs,expected_retina_source_sha256=sources[retinapath],acceptance='All9 retinal721 vectors exact in archivedfloat16; all3 sourcePNG pixels equal same-index sequentialdecode; otherwise do not finalize mask scientific conclusion.',
    explanation='Rawgray/255 differs by design: archive uses stateful nonlinear Lamina before BoxEye. Reconstruct that path, not merely gray normalization.'))
vm=psutil.virtual_memory();assert vm.available>max(6*1024**3,.2*vm.total)+2*1024**3
state=pickle.loads(statefile.read_bytes());retina=state['retina'];assert checkpoint['next_frame']==1100
records={}
for rec in p['records']:
    for r in rec['retinal']:records[(r['frame'],tuple(r['crop']))]=r
byframe={}
for (f,c),r in records.items():byframe.setdefault(f,[]).append(r)
points=np.asarray(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu')
cap=cv2.VideoCapture(str(video));assert cap.isOpened();rows=[];png=[];start=time.perf_counter()
for frame in range(1166):
    ok,image=cap.read();assert ok,frame
    if frame in (1159,1164,1165):
        path=ROOT/f'research_visible_part_2179_20261114/frames/{frame}.png'
        expected=cv2.imread(str(path));diff=np.abs(image.astype(int)-expected.astype(int))
        png.append(dict(frame=frame,exact=bool(np.array_equal(image,expected)),maximum_pixel_error=int(diff.max()),
            different_elements=int(np.count_nonzero(diff)),sequential_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            png_pixel_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    if frame<1100:continue
    processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    for record in byframe.get(frame,[]):
        x1,y1,x2,y2=record['crop']
        resized=cv2.resize(processed[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        x=torch.from_numpy(resized).reshape(1,1,391,391)
        with torch.inference_mode():filtered=torch.nn.functional.conv2d(torch.nn.functional.pad(x,(6,6,6,6)),kernel)/169
        actual=filtered[0,0,points[:,1],points[:,0]].numpy().astype(np.float16)
        expected=np.load(ROOT/record['path'],allow_pickle=False)
        error=float(abs(actual.astype(float)-expected.astype(float)).max())
        rows.append(dict(frame=frame,crop=record['crop'],source=record['path'],exact=bool(np.array_equal(actual,expected)),maximum_error=error,
            different_cells=int(np.count_nonzero(actual!=expected))))
cap.release()
for n,h in inputs.items():assert sha(ROOT/n)==h,n
passed=all(r['exact'] for r in rows+png)
save('alignment_audit.json',dict(passed=passed,retinal_inputs=rows,png_sources=png,
    maximum_retinal_error=max(r['maximum_error'] for r in rows),
    seconds=time.perf_counter()-start,new_core_calls=0,cuda_initialized=torch.cuda.is_initialized(),
    interpretation='Exactsameframealignmentconfirmed' if passed else 'Currentmaskresultnotfullyvalidateduntilmismatchresolved',
    raw_gray_control='DirectRGB-to-gray average is intentional image control, not claimed same as stateful Lamina input.'))
print('ALIGNMENT_CHECK',passed,json.dumps(dict(retina=rows,png=png),ensure_ascii=False))
