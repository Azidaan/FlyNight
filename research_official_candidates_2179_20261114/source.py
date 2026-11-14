from official import *
import gzip,torch
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from PIL import ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=read(OUT/'protocol.json');v=p['video'];root=ROOT/'runs/z13_val20_matched_20261024/cache'/v;raw=np.load(root/'detections/raw_video.npz');db=zipfile.ZipFile(ROOT/p['labelzip']);gt=labels(db,v);clock={a['video_frame_index']:a for a in read(root/'alignment.json')['matches']};hist=json.load(gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz'));pred={};data=[];official={};source_rows=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False);model=SAM2ImagePredictor(net)
 for fr in p['frames']:
  a=clock[fr];r=gt[a['reference_index']];assert a['reference_name']==r['name'];t,k,others=masks(db,v,r,p['target']);rival,_,_=masks(db,v,r,p['rival']);official.update({str(fr)+'__far':t,str(fr)+'__near':rival,str(fr)+'__valid':k});im=cv2.imread(str(ROOT/f'research_mask_propagation_20261114/frame_{fr}.png'));rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);model.set_image(rgb);ids=np.flatnonzero((raw['frame_idx']==fr)&(raw['cls']==2));assigned={int(e['raw_id']):int(e['tid']) for e in hist['assignments'] if e['frame']==fr and e['emitted']};e=next((e for e in hist['events'] if e['frame']==fr),None);high=e['high_raw_ids'] if e is not None else None
  def bounds(m):
   ys,xs=np.where(m);return np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
  source_rows.append(dict(frame=fr,target_box=bounds(t).tolist(),rival_box=bounds(rival).tolist(),target_pixels=int(t.sum()),rival_pixels=int(rival.sum()),snapshot_exists=e is not None))
  for rid in ids:
   b=raw['xyxy'][rid];confidence=float(raw['conf'][rid]);out,score,_=model.predict(box=b.astype('float32'),multimask_output=False);mask=out[0].astype(bool);name=str(fr)+'__'+str(rid);pred[name]=mask;band='high' if confidence>=.25 else 'low' if confidence>.1 else 'discarded';y,x=np.indices(t.shape);rectangle=(x>=b[0])&(x<=b[2])&(y>=b[1])&(y<=b[3]);source=dict(frame=fr,raw_id=int(rid),assigned_tid=assigned.get(int(rid)),band=band,original_high_column=int(rid) in high if high is not None else None,confidence=confidence,box=b.astype(float).tolist(),predicted_quality=float(score[0]),target_mask=measure(mask,t,k,rival),rival_mask=measure(mask,rival,k,t),target_box_iou=iou(b,bounds(t)),rival_box_iou=iou(b,bounds(rival)),raw_box_target_pixels=int((rectangle&t).sum()),raw_box_rival_pixels=int((rectangle&rival).sum()));data.append(source)
  print('ORIGINAL_PROPOSALS',fr,len(ids),flush=True)
for n,z in [('candidate_masks.npz',pred),('official_masks.npz',official)]:
 with (OUT/n).open('xb') as f:np.savez_compressed(f,**z)
save('candidate_result.json',dict(records=data,source_frames=source_rows,source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);objects=[r for r in data if r['frame']==1165];best_high=max([r for r in objects if r['band']=='high'],key=lambda s:s['target_mask']['iou']);best=max(objects,key=lambda s:s['target_mask']['iou']);actual=next(r for r in objects if r['raw_id']==110260);sheet=Image.new('RGB',(1600,342),'#23262b');draw=ImageDraw.Draw(sheet)
for i,(name,s) in enumerate([('Официальная цель',None),('Выданная рамка 110260',actual),('Лучшая допустимая высокая',best_high),('Лучшая из всех исходных',best)]):
 im=cv2.cvtColor(cv2.imread(str(ROOT/'research_mask_propagation_20261114/frame_1165.png')),cv2.COLOR_BGR2RGB);far=official['1165__far'];m=far if s is None else pred['1165__'+str(s['raw_id'])];im[m]=(im[m]*.4+np.array([35,235,60])*.6).astype('uint8');co,_=cv2.findContours(far.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,co,-1,(255,214,24),1);patch=cv2.resize(im[190:450,900:1280],(400,274));sheet.paste(Image.fromarray(patch),(400*i,32));draw.text((400*i+4,6),name,font=font,fill='white')
 if s is not None:draw.text((400*i+4,310),f"ID {s['raw_id']}, IoU {s['target_mask']['iou']:.3f}",font=font,fill='white')
sheet.save(OUT/'actual_candidates.jpg',quality=91)
