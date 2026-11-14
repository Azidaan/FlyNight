from official import *
from assignment import edges,overlap_matrix
import torch
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
from tracking.gpu_lock import GpuLock
from PIL import ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');prev=ROOT/'research_return_cycle_20261114';res=read(prev/'result.json');dec=read(prev/'decisions.json')['records'];initial=np.load(prev/'outputs.npz');cases={c['id']:c for c in read(prev/'cases.json')['cases']};rows=[];mout={};costs=[];cert=[];details=[];z=zipfile.ZipFile(ROOT/p['labelzip']);cfg='configs/sam2.1/sam2.1_hiera_t.yaml';w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt');zero=np.zeros((720,1280),bool);pic=[]
def prop(folder,mode,box,truth):
 st=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 if mode=='runtime':vp.add_new_points_or_box(st,0,1,box=np.array(box,'float32'))
 else:vp.add_new_mask(st,0,1,truth)
 result={}
 for j,_,val in vp.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=4):
  result[j]=(val[0,0]>0).cpu().numpy();st['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
 return result
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);imnet=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for cid in sorted({b['case'] for b in res['error_attribution']}):
  c=cases[cid];v=c['video'];folder=(ROOT/c['frames'][0]['image']).parent;ls=labels(z,v);idx=c['index'];e0=next(r for r in dec if r['case']==cid);boxes=e0['past_boxes'];ids=c['ids'];past=[masks(z,v,ls[idx],tid)[0] for tid in ids];pr={}
  for k in (0,1):
   for mode in ['runtime','truepast']:pr[k,mode]=prop(folder,mode,boxes[k],past[k]) if boxes[k] is not None else {j:zero for j in range(5)}
   assert np.array_equal(pr[k,'runtime'][0],initial[f'{cid}__past__{k}'])
  for j in [1,2,4]:
   e=next(r for r in dec if r['case']==cid and r['frame']==j);im=cv2.imread(str(ROOT/c['frames'][j]['image']));run=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];ob=[(a,bb) for a,(bb,cf) in enumerate(zip(run.boxes.xyxy.cpu().numpy(),run.boxes.conf.cpu().numpy())) if cf>.1];rids=[a for a,b in ob];ms=[];imnet.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
   for _,b in ob:ms.append(imnet.predict(box=b,multimask_output=False)[0][0].astype(bool))
   g=[masks(z,v,ls[idx+j],tid) for tid in ids];truth=[x[0] for x in g];valid=g[0][1];aa={}
   for name,ps in [('runtime',[pr[k,'runtime'][j] for k in (0,1)]),('prior_labels_only',[pr[k,'truepast'][j] for k in (0,1)]),('query_labels_only',truth)]:
    s=overlap_matrix(ps,ms);en=edges(s,[0,1],rids);aa[name]=dict(en.pairs);costs.append(dict(case=cid,offset=j,mode=name,rids=rids,actions=list(en.pairs),scores=s.tolist()))
   assert aa['runtime']==dict(e['forward'])
   for i,tid in enumerate(ids):
    orig=next(x for x in res['records'] if x['case']==cid and x['frame']==j and x['slot']==i and x['mode']=='forward')
    gq=g[i];pool=[measure(m,*gq)['iou'] or 0 for m in ms];past_valid=masks(z,v,ls[idx],tid)[1];source=measure(pr[i,'runtime'][0],past[i],past_valid,past[1-i]);runtime_prior=measure(pr[i,'runtime'][j],*gq);corrected_prior=measure(pr[i,'truepast'][j],*gq);sc={}
    for n,items in aa.items():
     rid=items.get(i);m=ms[rids.index(rid)] if rid is not None else zero;io=measure(m,*gq)['iou'];st='deferred' if rid is None else 'correct' if (io or 0)>.5 else 'wrong';sc[n]=dict(action=rid,status=st,iou=io)
    assert sc['runtime']['status']==orig['status'] and sc['runtime']['action']==orig['raw_id'];mout[f'{cid}__{j}__{i}__original']=pr[i,'runtime'][j];mout[f'{cid}__{j}__{i}__truepast']=pr[i,'truepast'][j]
    if orig['status']=='wrong':stage='no_visible_reference' if not truth[i].any() else 'no_sufficient_current_mask' if max(pool,default=0)<=.5 else 'initial_mask_sufficient_intervention' if sc['prior_labels_only']['status']=='correct' else 'temporal_prediction_sufficient_intervention' if sc['query_labels_only']['status']=='correct' else 'competition_or_observation_not_resolved'
    else:stage='original_'+orig['status']
    row=dict(case=cid,video=v,frame=j,slot=i,split=c['split'],id=tid,status=orig['status'],stage=stage,initial_valid_overlap=source['iou'],prior_overlap=runtime_prior['iou'],truepast_prior_overlap=corrected_prior['iou'],initial_quality_failed=(source['iou'] or 0)<=.5,visible=bool(truth[i].any()),best_current_mask_iou=max(pool,default=0),scores=sc);rows.append(row)
    if cid=='b1d7b3ac-0bdb47dc_149' and i==0 and j==4:
     chosen=ms[rids.index(sc['runtime']['action'])];allmask=truth[i]|chosen;yy,xx=np.where(allmask);bb=[max(0,xx.min()-30),max(0,yy.min()-25),min(1280,xx.max()+31),min(720,yy.max()+26)]
     for title,mask in [('Цель по контуру',truth[i]),('Прогноз из прошлого',pr[i,'runtime'][j]),('Выбранное наблюдение',chosen),('Из верной прошлой маски',pr[i,'truepast'][j])]:
      img=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);img[mask]=(img[mask]*.45+np.array([43,231,70])*.55).astype('uint8');co,_=cv2.findContours(truth[i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,co,-1,(255,214,20),2);x,y,x2,y2=bb;pa=Image.fromarray(img[y:y2,x:x2]);pa.thumbnail((280,215));pic.append((title,pa))
  print('LOCALIZED_STAGE',cid,flush=True)
with (OUT/'interventions.npz').open('xb') as f:np.savez_compressed(f,**mout)
save('measurements.json',dict(records=rows,assignment_matrices=costs));sheet=Image.new('RGB',(1120,254),'#23282c');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for k,(title,img) in enumerate(pic):sheet.paste(img,(280*k+(280-img.width)//2,34+(215-img.height)//2));dr.text((280*k+4,7),title,font=font,fill='white')
sheet.save(OUT/'fresh_wrong_car.jpg',quality=92)
