import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-12T18:15:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=min(600,remaining))
import numpy as np,cv2,math
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.z13_readout import HexGeometry
cv2.setNumThreads(2);p=verify();g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors
boxes=read(ROOT/'research_image_correspondence_probe_20261112/detector_box_audit.json')['records'];old=read(ROOT/'research_input_precision_probe_20261112/scores.json');frames={};framehash={};maxerr=0.;rows=[];pixels={};baseline=[]
videos=read(LEGACY/'protocol.json')['videos']
for v in sorted({c['case']['video'] for c in p['cases']}):
 wanted={c['sequence'][ep['length']-1][0] for c in p['cases'] if c['case']['video']==v for ep in c['endpoints']};entry=videos[v];cap=cv2.VideoCapture(str(ROOT/entry['video_path']));assert cap.isOpened()
 for f in range(max(wanted)+1):
  ok,img=cap.read();assert ok
  if f in wanted:frames[(v,f)]=img;framehash[v+'|'+str(f)]=hashlib.sha256(img.tobytes()).hexdigest()
 cap.release()
def bilinear(image,xy):
 global maxerr
 image=image.astype(float);image=image[:,:,None] if image.ndim==2 else image;h,w=image.shape[:2];x=xy[:,0];y=xy[:,1];assert np.all((x>=0)&(x<w-1)&(y>=0)&(y<h-1));ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);u=(x-ix)[:,None];v=(y-iy)[:,None]
 out=(1-u)*(1-v)*image[iy,ix]+u*(1-v)*image[iy,ix+1]+(1-u)*v*image[iy+1,ix]+u*v*image[iy+1,ix+1]
 other=RegularGridInterpolator((np.arange(h),np.arange(w)),image)(np.c_[y,x]);maxerr=max(maxerr,float(abs(out-other).max()));assert maxerr<1e-9;return out
def eye_gray(image):
 # Identical resize/13x13 area support, using independent float64 box sums here.
 patch=cv2.resize(image,(391,391),interpolation=cv2.INTER_AREA).astype(float)/255.;pad=np.pad(patch,6);pt=np.array(g['points'],int);return np.array([math.fsum(float(v) for v in pad[y:y+13,x:x+13].ravel())/169 for x,y in pt])[:,None]
def metric(a,b):
 global maxerr
 aa=a-a.mean(0);bb=b-b.mean(0);va=np.mean(aa*aa,0);vb=np.mean(bb*bb,0);cov=np.mean(aa*bb,0);den=va+vb+(a.mean(0)-b.mean(0))**2
 pear=np.divide(cov,np.sqrt(va*vb),out=np.zeros_like(cov),where=va*vb>0);ccc=np.divide(2*cov,den,out=np.zeros_like(cov),where=den>0);refP=[];refC=[]
 for j in range(a.shape[1]):
  ma=math.fsum(a[:,j])/len(a);mb=math.fsum(b[:,j])/len(b);V=math.fsum((float(x)-ma)**2 for x in a[:,j])/len(a);W=math.fsum((float(x)-mb)**2 for x in b[:,j])/len(b);C=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a[:,j],b[:,j]))/len(a);D=V+W+(ma-mb)**2;refP.append(C/math.sqrt(V*W) if V*W>0 else 0);refC.append(1-math.fsum((float(x)-float(y))**2 for x,y in zip(a[:,j],b[:,j]))/len(a)/D if D else 0)
 maxerr=max(maxerr,float(abs(pear-refP).max()),float(abs(ccc-refC).max()));assert maxerr<1e-9
 return dict(pearson=float(pear.mean()),concordance=float(ccc.mean()),per_channel_pearson=pear,per_channel_concordance=ccc)
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as maps,np.load(ROOT/'research_input_precision_probe_20261112/expanded_inputs.npz') as inputs:
 for ci,c in enumerate(p['cases']):
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];refworld=reader.points[cells] if hasattr(reader,'points') else np.array(g['points'])[cells];refworld=refworld/reader.eye_size*(np.array(r['reference_crop'][2:])-r['reference_crop'][:2])+r['reference_crop'][:2]
  graymaps=[];raw=[];grays=[]
  for ei,ep in enumerate(c['endpoints']):
   f,crop=c['sequence'][ep['length']-1];im=frames[(r['video'],f)];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);raw.append(im);grays.append(gray);x1,y1,x2,y2=crop;graymaps.append(eye_gray(gray[y1:y2,x1:x2]))
  for query_arm in ['C34','R134_temporal']:
   query=plan['query'][:,r['old_variants'][query_arm]['winner']];cc=np.array(r['current_crop']);world=query/reader.eye_size*(cc[2:]-cc[:2])+cc[:2];det=np.array(r['current_box']);gt=np.array(next(x for x in boxes if x['old_id']==r['old_id'])['annotated_candidate_box']);mask=((world>=det[:2])&(world<=det[2:])&(world>=gt[:2])&(world<=gt[2:])).all(1)
   pairs={'raw_color_pixels':(bilinear(raw[0],refworld),bilinear(raw[1],world)),'raw_gray_pixels':(bilinear(grays[0],refworld),bilinear(grays[1],world)),'raw_gray_eye':(graymaps[0][cells],sample_linear(reader.mesh,graymaps[1],query))}
   for mode in ['index','precision32']:
    h=inputs[f'{mode}_{ci}'];pairs['retina16' if mode=='index' else 'retina32']=(h[c['endpoints'][0]['length']-1,cells,None].astype(float),sample_linear(reader.mesh,h[c['endpoints'][1]['length']-1,:,None].astype(float),query))
   for arm in ARMS:
    arr=[]
    for ei in [0,1]:
     m=maps[f'index_{ci}_{ei}_C34'].astype(float) if arm=='C34' else maps[f'index_{ci}_{ei}_R1'].astype(float)
     if arm=='R134_spatial':m=np.c_[m[:,:17],m[nb[:,1:],:17].mean(1)]
     arr.append(m)
    a=arr[0][cells];b=sample_linear(reader.mesh,arr[1],query);other=LinearNDInterpolator(reader.mesh,arr[1])(query);maxerr=max(maxerr,float(abs(b-other).max()));assert maxerr<1e-9;pairs[arm]=(a,b)
   for support,M in [('full',np.ones(len(cells),bool)),('detector_and_annotation',mask)]:
    for stage,(A,B) in pairs.items():
     a=A[M];b=B[M];z=metric(a,b);rows.append(dict(old_id=r['old_id'],label=r['label'],query=query_arm,support=support,stage=stage,count=len(a),**z));pixels[f'{ci}_{query_arm}_{support}_{stage}_a']=a;pixels[f'{ci}_{query_arm}_{support}_{stage}_b']=b
     if stage=='C34' and query_arm=='C34' and support=='full':
      for head,field in [('native','per_channel_pearson'),('agreement','per_channel_concordance')]:
       q=next(x['channels'] for x in old['records'] if x['old_id']==r['old_id'] and x['mode']=='index' and x['representation']=='C34' and x['head']==head);err=float(abs(np.array(q)-z[field]).max());assert err<1e-9;baseline.append(dict(old_id=r['old_id'],head=head,error=err))
  (OUT/'status.json').write_text(json.dumps(dict(stage='measured',old_id=r['old_id'],updated=now())),encoding='utf8');print('MEASURED',r['old_id'],flush=True)
verify()
with (OUT/'paired_samples.npz').open('xb') as f:np.savez_compressed(f,**pixels)
save('result.json',dict(passed=True,records=rows,raw_frame_hashes=framehash,max_independent_error=maxerr,baseline_checks=baseline,samples_sha256=sha(OUT/'paired_samples.npz'),core_calls=0,new_fits=0,tracker_calls=0))
print('COMPLETE',len(rows),flush=True)
