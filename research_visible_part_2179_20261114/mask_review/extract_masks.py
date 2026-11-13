from pathlib import Path
import sys,os,json,hashlib,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import ROOT,sha,now
os.environ['CUDA_VISIBLE_DEVICES']='0'
for p in [ROOT/'tools/sam2-runtime-20261027',ROOT/'tools/sam2-deps-20261027']:sys.path.insert(0,str(p))
import cv2,numpy as np,torch
from tracking.gpu_lock import GpuLock
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(2);cv2.setNumThreads(2)
prompts=json.loads((HERE/'prompts.json').read_text())
weights=ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'
assert sha(weights)=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
arrays={};records=[];started=time.monotonic()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device='cuda',apply_postprocessing=False)
 predictor=SAM2ImagePredictor(model)
 for frame,objects in prompts.items():
  image=cv2.imread(str(HERE.parent/'frames'/f'{frame}.png'));assert image is not None
  predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));masks={};scores={}
  for name,entry in objects.items():
   points=np.asarray(entry['points'],np.float32);labels=np.asarray(entry['labels'],np.int32)
   result,quality,logits=predictor.predict(point_coords=points,point_labels=labels,box=np.asarray(entry['box'],np.float32),multimask_output=False)
   mask=result[0]>0;masks[name]=mask;scores[name]=float(quality[0]);arrays[f'{frame}_{name}_raw']=mask
  occluder=masks['person_gray']|masks['person_red']|masks['cart']
  near=masks['neighbor']&~occluder
  target=masks['target']&~(occluder|near)
  background=~(occluder|near|target)
  for name,mask in [('target',target),('neighbor',near),('occluder',occluder),('background',background)]: arrays[f'{frame}_{name}']=mask
  panels=[]
  for name,mask,color in [('target',target,(20,220,20)),('neighbor',near,(220,90,20)),('occluder',occluder,(20,20,240))]:
   view=image.copy();view[mask]=(.4*view[mask]+.6*np.asarray(color)).astype(np.uint8)
   cv2.putText(view,name,(660,235),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
   panels.append(cv2.resize(view[235:420,880:1280],(800,370),interpolation=cv2.INTER_NEAREST))
  cv2.imwrite(str(HERE/f'{frame}_mask_review.png'),np.vstack(panels))
  records.append(dict(frame=int(frame),scores=scores,pixel_counts={n:int(arrays[f'{frame}_{n}'].sum()) for n in ['target','neighbor','occluder','background']},image_sha256=sha(HERE.parent/'frames'/f'{frame}.png')))
  (HERE/'status.json').write_text(json.dumps(dict(updated=now(),completed=len(records),total=len(prompts))))
with (HERE/'masks.npz').open('xb') as f:np.savez_compressed(f,**arrays)
(HERE/'mask_receipt.json').write_text(json.dumps(dict(completed=True,elapsed_seconds=time.monotonic()-started,source_sha256=sha(__file__),prompts_sha256=sha(HERE/'prompts.json'),weights_sha256=sha(weights),mask_sha256=sha(HERE/'masks.npz'),records=records,scope='Diagnostic image-prompted SAM masks. No segmentation GT. Raw and visibility-priority masks both retained. Human QA required.',precision='FP32',mask_threshold=0),indent=2))
print('MASKS_READY',flush=True)
