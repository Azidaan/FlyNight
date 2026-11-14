from common import *
import cv2,numpy as np,torch,time
from tracking.gpu_lock import GpuLock
from torchvision.transforms import Compose
src=ROOT/'third_party/depth_anything_v2/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf';sys.path.insert(0,str(src))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize,NormalizeImage,PrepareForNet
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
class Explicit(DepthAnythingV2):
 def image2tensor(self,raw_image,input_size=518):
  tr=Compose([Resize(width=input_size,height=input_size,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()]);im=tr({'image':cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)/255.0})['image'];return torch.from_numpy(im).unsqueeze(0).to(next(self.parameters()).device),raw_image.shape[:2]
p=read(OUT/'protocol.json');maps={};rows=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=Explicit(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(ROOT/p['weight'],map_location='cpu',weights_only=True));model.eval().to('cuda')
 for c in p['cases']:
  for fr in [c['reference']]+c['queries']:
   im=cv2.imread(str(ROOT/c['image_dir']/c['image_pattern'].format(frame=fr)));assert im is not None;torch.cuda.synchronize();t=time.perf_counter();d=model.infer_image(im,p['depth_size']);torch.cuda.synchronize();assert np.isfinite(d).all() and d.shape==im.shape[:2];maps[c['id']+'__'+str(fr)]=d;rows.append(dict(case=c['id'],frame=fr,seconds=time.perf_counter()-t));print('DEPTH_FRAME',c['id'],fr,flush=True)
with (OUT/'maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('inference.json',dict(passed=True,records=rows,maps_sha256=sha(OUT/'maps.npz'),source_unchanged=True,weights_unchanged=True))
