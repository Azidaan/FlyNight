from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import cv2,numpy as np,torch,time,psutil
from ultralytics import YOLO
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol.json');torch.set_num_threads(2);cv2.setNumThreads(2)
images=[cv2.imread(str(ROOT/r['image'])) for r in read(OUT/'case2179.json')['events'][0]['roles']]
rows=[]
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=YOLO(str(ROOT/read(OUT/'weights_receipt.json')['path']))
 def run(img):
  r=model.predict(source=img,**p['settings'])[0]
  boxes=r.boxes.xyxy.cpu().numpy().copy()
  masks=np.zeros((0,*img.shape[:2]),bool) if r.masks is None else r.masks.data.cpu().numpy().astype(bool)
  return len(boxes),int(masks.sum())
 for _ in range(p['warmup']):
  for image in images:run(image)
 torch.cuda.synchronize();torch.cuda.empty_cache()
 for rep in range(p['repeats']):
  for i,img in enumerate(images):
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();count,pixels=run(img);torch.cuda.synchronize();elapsed=time.perf_counter()-start
   rows.append(dict(repeat=rep,image_index=i,seconds=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),process_rss_bytes=psutil.Process().memory_info().rss,mask_count=count,mask_pixels=pixels))
save('seg_only_benchmark.json',dict(passed=True,records=rows,device=torch.cuda.get_device_name(),weight_sha256=sha(ROOT/read(OUT/'weights_receipt.json')['path']),source_sha256=sha(__file__),scope='Isolated YOLO26n-seg process; frame already decoded; includes preprocessing, network, mask decoding and CPU transfers, not MASA, matching, detector or full tracker. PyTorch allocated/reserved peaks exclude non-PyTorch CUDA context and other applications.'))
print('SEGMENTATION_BENCHMARK_COMPLETE',len(rows))
