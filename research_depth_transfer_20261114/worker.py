from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np,cv2,torch,time,psutil
from tracking.gpu_lock import GpuLock
from torchvision.transforms import Compose
SRC=ROOT/'third_party/depth_anything_v2/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
sys.path.insert(0,str(SRC))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize,NormalizeImage,PrepareForNet
torch.set_num_threads(2);cv2.setNumThreads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
class ExplicitDeviceDepth(DepthAnythingV2):
    def image2tensor(self,raw_image,input_size=518):
        tr=Compose([Resize(width=input_size,height=input_size,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()])
        im=tr({'image':cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)/255.0})['image']
        return torch.from_numpy(im).unsqueeze(0).to(next(self.parameters()).device),raw_image.shape[:2]
def status(stage,done,total):
    value=dict(time=now(),stage=stage,done=done,total=total,deadline='2026-11-14T08:55:00+03:00');tmp=OUT/'status.pending';tmp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');tmp.replace(OUT/'status.json');print(json.dumps(value),flush=True)
p=read(OUT/'protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
mode=sys.argv[1];assert mode in ['benchmark','holdout'];rows=[];maps={}
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
    assert torch.cuda.is_available();model=ExplicitDeviceDepth(encoder='vits',features=64,out_channels=[48,96,192,384]);model.load_state_dict(torch.load(ROOT/p['weight'],map_location='cpu',weights_only=True));model.eval().to('cuda');assert next(model.parameters()).device.type=='cuda'
    if mode=='benchmark':
        im=cv2.imread(str(ROOT/p['benchmark_image']))
        for size in [518,1036]:
            model.infer_image(im,size);torch.cuda.synchronize();torch.cuda.empty_cache()
            for rep in range(5):
                status('benchmark '+str(size),rep,5);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();d=model.infer_image(im,size);torch.cuda.synchronize();elapsed=time.perf_counter()-t;assert np.isfinite(d).all()
                rows.append(dict(input_size=size,repeat=rep,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),rss_bytes=psutil.Process().memory_info().rss))
                if rep==0:maps[str(size)]=d.copy()
        with (OUT/'benchmark_maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
        cpu=np.load(ROOT/'research_depth_order_20261114/resolution_maps.npz')['1165'];gpu=maps['1036'];nrmse=float(np.sqrt(np.mean((cpu-gpu)**2))/np.ptp(cpu));same=nrmse<.001
        save('benchmark_result.json',dict(records=rows,device=torch.cuda.get_device_name(),cpu_gpu_nrmse=nrmse,device_gate_passed=same,source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items()),scope='GPU float32, explicit device, one warmup per size then5repeats; image already decoded, includes preprocessing/network/CPU output, no tracker. Allocated/reserved exclude non-PyTorch CUDA context. CPU comparison uses frozen1036map of same frame; not a paired timing comparison.'))
        assert same,('CPU/GPU mismatch',nrmse)
    else:
        assert read(OUT/'benchmark_result.json')['device_gate_passed']
        total=len(p['records'])*2
        for r in p['records']:
            im=cv2.imread(str(ROOT/r['image']));assert im is not None
            for size in [518,1036]:
                status('holdout '+r['id']+' '+str(size),len(rows),total);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();d=model.infer_image(im,size);torch.cuda.synchronize();assert d.shape==im.shape[:2] and np.isfinite(d).all();maps[r['id']+'__'+str(size)]=d
                rows.append(dict(id=r['id'],input_size=size,seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        with (OUT/'holdout_maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
        save('inference_result.json',dict(records=rows,maps_sha256=sha(OUT/'holdout_maps.npz'),source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items()),device=torch.cuda.get_device_name(),labels_used_in_prediction=False))
status(mode+' complete',len(rows),len(rows))
