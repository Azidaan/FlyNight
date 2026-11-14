from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np,cv2,torch,time
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol.json');previous=read(OUT/'teacher_result.json');pm=np.load(OUT/'teacher_masks.npz',allow_pickle=False);assert read(OUT/'teacher_crop_protocol.json')['registered'];torch.set_num_threads(2);cv2.setNumThreads(2)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);rows=[];masks={};tiles=[]
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=YOLO(str(ROOT/'weights/yolo26seg_official_20261114/yolo26n-seg.pt'))
 for index,r in enumerate([r for r in p['records'] if r['split']=='train']):
  old=next(t for t in previous['records'] if t['id']==r['id']);image=cv2.imread(str(ROOT/r['image']));x1,y1,x2,y2=r['crop'];patch=cv2.resize(image[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA)
  if old['automatic_gate']:
   mask=cv2.resize(pm[r['id']],(391,391),interpolation=cv2.INTER_NEAREST).astype(bool);row=dict(old,teacher_view='fullframe_original')
  else:
   result=model.predict(patch,**p['teacher'])[0];boxes=result.boxes.xyxy.cpu().numpy();b=(np.array(r['box'])-np.array([x1,y1,x1,y1]))*391/np.array([x2-x1,y2-y1,x2-x1,y2-y1]);mask=np.zeros((391,391),bool);selected=None;iou=0;purity=0
   if len(boxes):
    inter=np.prod(np.maximum(0,np.minimum(boxes[:,2:],b[2:])-np.maximum(boxes[:,:2],b[:2])),axis=1);union=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)+np.prod(np.maximum(b[2:]-b[:2],0))-inter;ious=inter/np.maximum(union,1e-12);selected=int(ious.argmax());iou=float(ious[selected]);mask=result.masks.data[selected].cpu().numpy().astype(bool)
    bx1,by1,bx2,by2=np.floor(b+[0,0,1,1]).astype(int);purity=float(mask[max(0,by1):min(391,by2),max(0,bx1):min(391,bx2)].sum()/max(1,mask.sum()))
   row=dict(id=r['id'],video=r['video'],frame=r['frame'],selected=selected,bbox_iou=iou,mask_inside_supplied_roi=purity,automatic_gate=bool(selected is not None and iou>=.5 and purity>=.8),mask_pixels=int(mask.sum()),teacher_view='crop391_fallback')
  rows.append(row);masks[r['id']]=mask.astype(np.uint8)
  if row['automatic_gate']:
   overlay=patch.copy();overlay[mask]=(overlay[mask]*.5+np.array([0,230,0])*.5).astype(np.uint8);tile=Image.new('RGB',(391,425),'#18331f');tile.paste(Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)),(0,34));d=ImageDraw.Draw(tile);d.text((3,3),f"{index} {r['video'][:8]}:{r['frame']} {row['teacher_view'][:4]}",font=font,fill='white');tiles.append(tile)
  print('CROP_TEACHER',index,row['automatic_gate'],flush=True)
with (OUT/'teacher_combined_masks.npz').open('xb') as h:np.savez_compressed(h,**masks)
save('teacher_combined_result.json',dict(created=now(),records=rows,old_result_sha256=sha(OUT/'teacher_result.json'),masks_sha256=sha(OUT/'teacher_combined_masks.npz'),source_sha256=sha(__file__),no_reader_outcomes_seen=True))
for page in range((len(tiles)+15)//16):
 sheet=Image.new('RGB',(1564,1700),'#202020')
 for k,tile in enumerate(tiles[page*16:(page+1)*16]):sheet.paste(tile,((k%4)*391,(k//4)*425))
 sheet.save(OUT/'qa'/f'teacher_accepted_v2_{page}.png')
print('CROP_TEACHER_DONE',len(tiles),len({r['video'] for r in rows if r['automatic_gate']}),flush=True)
