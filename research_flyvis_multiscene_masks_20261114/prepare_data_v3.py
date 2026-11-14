from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,torch,pickle,time,psutil
from PIL import Image,ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2)
p=read(OUT/'protocol.json');start=time.perf_counter()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
assert psutil.virtual_memory().available>max(6*1024**3,.2*psutil.virtual_memory().total)+2*1024**3
legacy=ROOT/'runs/identity_assignment_20261108/full_precision';lp=read(legacy/'protocol.json')
points=np.asarray(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);px,py=points.T
kernel=torch.ones((1,1,13,13),dtype=torch.float32);audits=[];features={};snapshots={};extra_sources={}
(OUT/'frames').mkdir(exist_ok=True);(OUT/'qa').mkdir(exist_ok=True)
for vid in p['train_videos']+p['test_videos']:
 rows=[r for r in p['records'] if r['video']==vid];needs={}
 for r in rows:
  for t in r['retinal']:needs.setdefault(t['frame'],{})[tuple(t['crop'])]=t['path']
 states=[]
 for file in sorted((legacy/'processed'/(vid+'.checkpoints')).glob('*.json')):
  m=read(file);states.append((m['next_frame'],file,m))
 groups={}
 for f in needs:
  valid=[s for s in states if s[0]<=f]
  s=valid[-1] if valid else (0,None,None);groups.setdefault(s[0],dict(state=s,last=f))['last']=max(groups.get(s[0],{}).get('last',f),f)
 cap=cv2.VideoCapture(str(ROOT/lp['videos'][vid]['video_path']));assert cap.isOpened();retina=None;active_end=-1;count=0
 for f in range(max(max(needs),max(r['frame'] for r in rows))+1):
  ok,img=cap.read();assert ok,(vid,f)
  if f in groups:
   _,meta,m=groups[f]['state']
   if meta is None:
    from tracking.retina import make_retina
    retina=make_retina('lamina',lp['videos'][vid]['fps'],blur_downscale=4)
   else:
    state=meta.parent/m['state_file'];assert sha(state)==m['state_sha256'];extra_sources[rel(meta)]=sha(meta);extra_sources[rel(state)]=sha(state);retina=pickle.loads(state.read_bytes())['retina']
   active_end=groups[f]['last']
  if retina is not None and f<=active_end:
   processed=retina.process(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY));count+=1
   for crop,path in needs.get(f,{}).items():
    x1,y1,x2,y2=crop;im=cv2.resize(processed[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    with torch.inference_mode():v=torch.nn.functional.conv2d(torch.nn.functional.pad(torch.from_numpy(im)[None,None],(6,6,6,6)),kernel)/169
    actual=v[0,0,py,px].numpy().astype(np.float16);expected=np.load(ROOT/path,allow_pickle=False)
    audits.append(dict(video=vid,frame=f,crop=list(crop),exact=bool(np.array_equal(actual,expected)),max_error=float(abs(actual.astype(float)-expected.astype(float)).max()),source=path))
  for r in rows:
   if f!=r['frame']:continue
   dest=ROOT/r['image']
   if dest.exists():assert np.array_equal(cv2.imread(str(dest)),img),'Existing raw frame differs'
   else:assert cv2.imwrite(str(dest),img)
   x1,y1,x2,y2=r['crop'];patch=cv2.resize(img[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA)
   rgb=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).astype(float)/255
   raw=cv2.boxFilter(rgb,-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)[py,px]
   hx=np.stack([np.load(ROOT/t['path'],allow_pickle=False) for t in r['retinal']]).astype(float);ema=np.repeat(hx[:1],17,axis=0);taus=np.geomspace(1.5,32,17)
   for cur in hx:ema+=(cur[None]-ema)/taus[:,None]
   history=np.concatenate([ema,cur[None]-ema]).T
   cx=x1+(px+.5)*(x2-x1)/391-.5;cy=y1+(py+.5)*(y2-y1)/391-.5;bx1,by1,bx2,by2=r['box']
   rx=(cx-(bx1+bx2)/2)/max(bx2-bx1,1);ry=(cy-(by1+by2)/2)/max(by2-by1,1)
   geom=np.stack([rx,ry,rx**2,ry**2,rx*ry,((abs(rx)<=.5)&(abs(ry)<=.5)).astype(float)],axis=1)
   fly=np.load(ROOT/r['map'],allow_pickle=False).astype(float)
   for arm,arr in [('FlyVis34_G6',np.c_[fly,geom]),('Retina_history34_G6',np.c_[history,geom]),('Geometry6',geom),('RGB3_G6',np.c_[raw,geom]),('FlyVis34',fly),('Retina_history34',history)]:
    assert np.isfinite(arr).all();features[r['id']+'__'+arm]=arr.astype(np.float32)
   snapshots[r['id']]=patch
 cap.release();print('DECODED',vid,'frames',len(rows),'retina_frames',count,flush=True)
save('alignment_audit.json',dict(passed=all(a['exact'] for a in audits),records=audits,checkpoint_sources=extra_sources,seconds=time.perf_counter()-start,method='Sequential decode from0; verified causal Lamina checkpoints; actual13x13 CPU BoxEye; every selected retinal vector compared in archivedfloat16.'))
assert all(a['exact'] for a in audits),'RETINAL_ALIGNMENT_FAILED: do not fit'
with (OUT/'features.npz').open('xb') as f:np.savez_compressed(f,**features)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
test=[r for r in p['records'] if r['split']=='test']
for i,r in enumerate(test):
 image=Image.fromarray(cv2.cvtColor(snapshots[r['id']],cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image);x1,y1,x2,y2=r['crop'];b=r['box'];box=[(b[0]-x1)*391/(x2-x1),(b[1]-y1)*391/(y2-y1),(b[2]-x1)*391/(x2-x1),(b[3]-y1)*391/(y2-y1)];d.rectangle(box,outline='#00ffff',width=2)
 image.save(OUT/'qa'/f'test_{i}_original.png')
sheet=Image.new('RGB',(4*391,2*425),'#202020');d=ImageDraw.Draw(sheet)
for i,r in enumerate(test):
 x=(i%4)*391;y=(i//4)*425;sheet.paste(Image.open(OUT/'qa'/f'test_{i}_original.png'),(x,y+34));d.text((x+4,y+5),f"{i}: {r['video'][:8]} / {r['frame']}",font=font,fill='white')
sheet.save(OUT/'qa/test_originals.png')
save('data_receipt.json',dict(created=now(),features_sha256=sha(OUT/'features.npz'),images={r['image']:sha(ROOT/r['image']) for r in p['records']},source_sha256=sha(__file__),core_calls=0,tracker_calls=0,seconds=time.perf_counter()-start))
print('DATA_READY',len(p['records']),flush=True)
