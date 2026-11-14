from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np,cv2,torch,time,psutil
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol.json');assert read(OUT/'alignment_audit.json')['passed'];torch.set_num_threads(2);cv2.setNumThreads(2)
weight=ROOT/'weights/yolo26seg_official_20261114/yolo26n-seg.pt';assert sha(weight)==p['inputs'][rel(weight)]
rows=[];masks={};tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);start=time.perf_counter()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=YOLO(str(weight))
 for r in [r for r in p['records'] if r['split']=='train']:
  image=cv2.imread(str(ROOT/r['image']));result=model.predict(image,**p['teacher'])[0];boxes=result.boxes.xyxy.cpu().numpy();selected=None;iou=0.;purity=0.;mask=np.zeros(image.shape[:2],bool)
  b=np.array(r['box']);area=np.prod(np.maximum(b[2:]-b[:2],0))
  if len(boxes):
   intersection=np.prod(np.maximum(0,np.minimum(boxes[:,2:],b[2:])-np.maximum(boxes[:,:2],b[:2])),axis=1);union=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)+area-intersection;ious=intersection/np.maximum(union,1e-12);selected=int(ious.argmax());iou=float(ious[selected])
   mask=result.masks.data[selected].cpu().numpy().astype(bool);xx1,yy1,xx2,yy2=np.floor(b+np.array([0,0,1,1])).astype(int);purity=float(mask[max(yy1,0):min(yy2,720),max(xx1,0):min(xx2,1280)].sum()/max(mask.sum(),1))
  x1,y1,x2,y2=r['crop'];cropmask=mask[y1:y2,x1:x2];masks[r['id']]=cropmask.astype(np.uint8)
  accepted=selected is not None and iou>=.5 and purity>=.8
  row=dict(id=r['id'],video=r['video'],frame=r['frame'],selected=selected,bbox_iou=iou,mask_inside_supplied_roi=purity,automatic_gate=bool(accepted),mask_pixels=int(cropmask.sum()));rows.append(row)
  patch=image[y1:y2,x1:x2].copy();patch[cropmask]=(patch[cropmask]*.5+np.array([0,230,0])*.5).astype(np.uint8);patch=cv2.resize(patch,(391,391));tile=Image.new('RGB',(391,425),'#18331f' if accepted else '#542626');tile.paste(Image.fromarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)),(0,34));d=ImageDraw.Draw(tile);d.text((3,3),f"{len(rows)-1} {r['video'][:8]}:{r['frame']} {'PASS' if accepted else 'SKIP'}",font=font,fill='white');tiles.append(tile)
  print('TEACHER',len(rows),r['id'],accepted,flush=True)
 with (OUT/'teacher_masks.npz').open('xb') as f:np.savez_compressed(f,**masks)
 perf=dict(seconds=time.perf_counter()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),rss_bytes=psutil.Process().memory_info().rss,device=torch.cuda.get_device_name())
for page in range((len(tiles)+15)//16):
 sheet=Image.new('RGB',(4*391,4*425),'#202020')
 for i,t in enumerate(tiles[page*16:(page+1)*16]):sheet.paste(t,((i%4)*391,(i//4)*425))
 sheet.save(OUT/'qa'/f'train_teacher_{page}.jpg',quality=85)
save('teacher_result.json',dict(records=rows,performance=perf,weight_sha256=sha(weight),masks_sha256=sha(OUT/'teacher_masks.npz'),source_sha256=sha(__file__),test_labels_seen=False,scope='Automatic pseudo-labels, review exclusion pending. Training clips only.'))
print('TEACHER_READY',sum(r['automatic_gate'] for r in rows),flush=True)
