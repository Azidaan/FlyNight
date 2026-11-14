from official import *
import torch,time
from scipy.optimize import linear_sum_assignment
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from PIL import ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol_v3.json');source=ROOT/'research_missing_candidate_repair_20261114';dd=read(source/'detection_result.json');cases=read(source/'cases.json')['cases'];lab=zipfile.ZipFile(ROOT/p['labelzip']);cache={};rows=[];pairs=[];best_masks={};dates=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def assign(q,fixed=None):
 fixed=[None,None] if fixed is None else fixed;out=fixed.copy();free=[i for i,a in enumerate(out) if a is None];reserved=[x for x in out if x is not None]
 if free and q.shape[1]:
  val=q[free].copy();val[:,reserved]=0;obj=np.where(val>=.5,1+val,0);ii,jj=linear_sum_assignment(-np.c_[obj,np.zeros((len(free),len(free)))])
  for i,j in zip(ii,jj):
   if j<q.shape[1] and obj[i,j]>0:out[free[i]]=int(j)
 return out
def best(vals):return int(np.argmax(vals)) if len(vals) else None
def bounds(m):
 ys,xs=np.where(m);return np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float)
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 m=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False))
 for c in cases:
  cid=c['id'];v=c['video'];idx=c['index'];im=cv2.imread(str(ROOT/c['frames'][0]['image']));gt=cache.setdefault(v,labels(lab,v)) if v not in cache else cache[v];frame=gt[idx];objs=[next(o for o in frame['labels'] if o['id']==n) for n in c['ids']];states=[masks(lab,v,frame,n) for n in c['ids']];t=[g[0] for g in states];k=states[0][1];raw=dd['raw_boxes'][cid];qr=sorted([x for x in dd['records'] if x['case']==cid],key=lambda r:r['slot']);source_choice=[q['full']['index'] if q['full'] is not None else None for q in qr];missing=any(i is None for i in source_choice)
  # Full car proposals for every episode; tiled car and other-class proposals only for the six registered missing references.
  variants=['full']+(['tile','fullall','tileall'] if missing else []);m.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));memo={}
  for arm in variants:
   pool=raw[arm];mask_values=np.zeros((2,len(pool)));bbox_values=np.zeros_like(mask_values);sums=[];bestpix=[None,None];highest=[-1.,-1.];bestall=[None,None];highestall=[-1.,-1.]
   truth_codes=[M.encode(np.asfortranarray(y.astype('uint8'))) for y in t]
   for j,obj in enumerate(pool):
    b=np.array(obj['box'],'float32');key=tuple(b.tolist())
    if key not in memo:
     pr=m.predict(box=b,multimask_output=False)[0][0].astype(bool);cr=M.encode(np.asfortranarray((pr&k).astype('uint8')));memo[key]=cr
    cr=memo[key];over=M.iou([cr],truth_codes,[0,0])[0]
    # Ignore pixels explicitly for both sides; sanity-check official instance has no ignored pixels.
    assert all(int(M.area(o['rle']))>=int(M.area(code)) for o,code in zip(objs,truth_codes))
    mask_values[:,j]=over;bbox_values[:,j]=[iou(b,bbox(o)) for o in objs];sums.append(int(M.area(cr)))
    for i,val in enumerate(over):
     if val>highest[i]:highest[i]=float(val);bestpix[i]=cr
   bybox=assign(bbox_values);bymask=assign(mask_values);protected=assign(mask_values,source_choice) if arm=='full' else None
   if arm=='full':assert bybox==source_choice
   pairs.append(dict(case=cid,video=v,split=c['split'],arm=arm,items=pool,box_scores=bbox_values.tolist(),mask_scores=mask_values.tolist(),box_assign=bybox,mask_assign=bymask,preserve_box_assign=protected))
   for i in [0,1]:
    bi=best(bbox_values[i]);mi=best(mask_values[i]);idc=source_choice[i];row=dict(case=cid,video=v,split=c['split'],slot=i,identity=c['ids'][i],arm=arm,initial_missed=idc is None,best_box_idx=bi,best_mask_idx=mi,best_box_iou=float(bbox_values[i,bi]) if bi is not None else 0.,best_mask_iou=float(mask_values[i,mi]) if mi is not None else 0.,best_mask_source_box=pool[mi] if mi is not None else None,mask_iou_of_best_box=float(mask_values[i,bi]) if bi is not None else None,visible_tight_box_iou=iou(bbox(objs[i]),bounds(t[i])) if t[i].any() else None,official_pixels=int(t[i].sum()),assigned_by_box=bybox[i],assigned_by_mask=bymask[i],mask_quality_of_box_choice=float(mask_values[i,bybox[i]]) if bybox[i] is not None else None,conditional_extra_owner=protected[i] if protected is not None and idc is None else None)
    rows.append(row)
    if row['initial_missed'] and bestpix[i] is not None:
     key=f'{cid}__{i}__{arm}';best_masks[key]=M.decode(bestpix[i]).astype(bool)
     if arm=='full':
      b=bbox(objs[i]);s=pool[mi]['box'];xs=[b[0],s[0]];xe=[b[2],s[2]];ys=[b[1],s[1]];ye=[b[3],s[3]];left=max(0,int(min(xs))-26);right=min(1280,int(max(xe))+27);top=max(0,int(min(ys))-23);bottom=min(720,int(max(ye))+24);img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);mm=best_masks[key];img[mm]=(img[mm]*.45+np.array([45,231,83])*.55).astype('uint8');co,_=cv2.findContours(t[i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,co,-1,(255,218,29),2);patch=Image.fromarray(img[top:bottom,left:right]);patch.thumbnail((330,190));dates.append((patch,round(row['best_box_iou'],3),round(row['best_mask_iou'],3),cid))
  print('GEOMETRY_VS_CONTENT',cid,flush=True)
with (OUT/'best_masks.npz').open('xb') as f:np.savez_compressed(f,**best_masks)
save('scored_candidates.json',dict(records=rows,pairs=pairs));sheet=Image.new('RGB',(990,480),'#252a30');dr=ImageDraw.Draw(sheet)
for i,(patch,b,m,cid) in enumerate(dates):
 x=330*(i%3);y=240*(i//3);sheet.paste(patch,(x+(330-patch.width)//2,y+27+(190-patch.height)//2));dr.text((x+5,y+5),cid,font=font,fill='white');dr.text((x+5,y+213),f'Лучшая рамка {b:.3f}; лучшая маска {m:.3f}',font=font,fill='white')
sheet.save(OUT/'six_missing.jpg',quality=91)
