from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false';os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import cv2,numpy as np,torch
from ultralytics import YOLO
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);cv2.setNumThreads(2)
protocol=read(OUT/'class_filter_protocol.json');settings=dict(read(OUT/'protocol.json')['settings']);settings['classes']=None
bank={};records=[]
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=YOLO(str(ROOT/read(OUT/'weights_receipt.json')['path']))
 for idx,path in enumerate(protocol['images']):
  img=cv2.imread(str(ROOT/path));r=model.predict(source=img,**settings)[0];key='image'+str(idx)
  bank[key]=np.zeros((0,*img.shape[:2]),bool) if r.masks is None else r.masks.data.cpu().numpy().astype(bool)
  records.append(dict(image=path,key=key,boxes=r.boxes.xyxy.cpu().numpy().tolist(),classes=r.boxes.cls.cpu().numpy().tolist(),scores=r.boxes.conf.cpu().numpy().tolist()))
with (OUT/'class_filter_masks.npz').open('xb') as f:np.savez_compressed(f,**bank)
save('class_filter_result.json',dict(passed=True,records=records,names=model.names,settings=settings,scope='Post-hoc cause check of class filter only. Original registered pipeline results and confidence threshold unchanged; no promotion or retuning.',source_sha256=sha(__file__),masks_sha256=sha(OUT/'class_filter_masks.npz')))
print('CLASS_FILTER_PROBE_COMPLETE')
