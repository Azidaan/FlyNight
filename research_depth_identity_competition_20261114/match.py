from common import *
import cv2,numpy as np,torch,time
from scipy.stats import rankdata
from tracking.gpu_lock import GpuLock
from torchvision.transforms import Compose
src=ROOT/'third_party/depth_anything_v2/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf';sys.path.insert(0,str(src))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize,NormalizeImage,PrepareForNet
class Explicit(DepthAnythingV2):
 def image2tensor(self,raw_image,input_size=518):
  tr=Compose([Resize(width=input_size,height=input_size,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()]);im=tr({'image':cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)/255.0})['image'];return torch.from_numpy(im).unsqueeze(0).to(next(self.parameters()).device),raw_image.shape[:2]
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=read(OUT/'protocol.json');pairs=read(OUT/'pairs.json')['cases'];data={a:np.load(OUT/(a+'_candidates.npz')) for a in ['auto','oracle']};desc={};outputs=[];q=[.1,.25,.5,.75,.9];timings=[];chosen={}
def boxiou(a,b):
 ii=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return ii/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-ii)
def pick(c):return int(np.argmin(c)) if c[0]!=c[1] else None
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=Explicit(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(ROOT/p['weight'],map_location='cpu',weights_only=True));model.eval().to('cuda')
 for c in pairs:
  cid=c['id']
  for j in [0,1,2,4]:
   pix=cv2.imread(str(ROOT/c['frames'][j]['image']));torch.cuda.synchronize();ts=time.monotonic();d=model.infer_image(pix,1036);torch.cuda.synchronize();timings.append(time.monotonic()-ts);assert d.shape==pix.shape[:2] and np.isfinite(d).all();dd=(rankdata(d.ravel(),method='average').reshape(d.shape)-.5)/d.size;yy=np.broadcast_to(np.arange(d.shape[0])[:,None]/(d.shape[0]-1),d.shape);rgb=cv2.cvtColor(pix,cv2.COLOR_BGR2RGB)
   for typ,zz in data.items():
    for i in [0,1]:
     key=f'{cid}__{i}__{j}';m=zz[key];a=typ+'__'+key
     if not m.any():continue
     y,x=np.where(m);desc[a+'__RGB']=np.quantile(rgb[m]/255,q,axis=0).ravel();desc[a+'__Depth']=np.quantile(dd[m],q);desc[a+'__Y']=np.quantile(yy[m],q);desc[a+'__Geom']=np.array([x.min(),y.min(),x.max()+1,y.max()+1])
  for j in [1,2,4]:
   swapped=hashlib.sha256(f'{cid}:{j}'.encode()).digest()[0]%2
   for typ in data:
    names=[typ+'__'+f'{cid}__{i}__0' for i in [0,1]]+[typ+'__'+f'{cid}__{i}__{j}' for i in [swapped,1-swapped]];missing=any(n+'__RGB' not in desc for n in names);totals={};decisions={}
    if not missing:
     for metric in ['RGB','Depth','Y','Geom']:
      a,b,c1,d1=[desc[n+'__'+metric] for n in names];distance=lambda x,y:(1-boxiou(x,y)) if metric=='Geom' else np.abs(x-y).mean();cost=np.array([distance(a,c1)+distance(b,d1),distance(a,d1)+distance(b,c1)]);totals[metric]=cost.tolist();decisions[metric]=pick(cost)
     for extra in ['Depth','Y']:
      a=np.array(totals['RGB']);b=np.array(totals[extra]);na=a/a.sum() if a.sum()>0 else np.full(2,.5);nb=b/b.sum() if b.sum()>0 else np.full(2,.5);decisions['RGB+'+extra]=pick(na+nb);ra=rankdata(a)+rankdata(b);decisions['rank_RGB+'+extra]=pick(ra) if ra[0]!=ra[1] else pick(a)
    outputs.append(dict(case=cid,video=c['video'],offset=j,branch=typ,shuffled=swapped,missing_candidate_or_reference=missing,costs=totals,decisions=decisions))
  print('DEPTH_PAIR',cid,flush=True)
with (OUT/'descriptors.npz').open('xb') as f:np.savez_compressed(f,**desc)
save('choices.json',dict(records=outputs,depth_frames=len(timings),seconds_per_map=timings,source_sha256=sha(__file__),no_truth_used_for_auto_assignment=True))
