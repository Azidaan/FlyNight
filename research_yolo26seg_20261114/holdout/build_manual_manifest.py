import cv2,datetime,hashlib,json,numpy as np
from pathlib import Path
ROOT=Path(r'E:\Dissertation');OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
cases=json.loads((OUT/'frozen_pairs.json').read_text(encoding='utf8'))['cases']
# Manually specified conservative physical cores, after original-pixel review;
# no segmentation, similarity, or assignment score was consulted.
polys={
 'B_day_shop':{
  'ref':{'target':[[[943,332],[960,329],[969,334],[966,345],[942,345]]], 'competitor':[[[914,339],[931,339],[933,347],[912,348]]]},
  'query':{'target':[[[960,329],[981,327],[987,334],[982,344],[957,345]]], 'competitor':[[[902,339],[923,336],[925,346],[904,350]]]}},
 'B_night_rain':{
  'ref':{'target':[[[936,396],[965,397],[970,410],[937,412]]], 'competitor':[[[723,369],[801,370],[808,414],[717,414]]]},
  'query':{'target':[[[963,397],[1006,399],[1010,414],[963,416]]], 'competitor':[[[743,371],[824,372],[831,418],[737,418]]]}},
 'B_dusk_queue':{
  'ref':{'target':[[[734,286],[805,286],[811,332],[734,333]]], 'competitor':[[[552,310],[679,309],[687,364],[548,366]]]},
  'query':{'target':[[[739,282],[812,283],[817,329],[737,330]]], 'competitor':[[[547,307],[674,307],[682,360],[544,364]]]}}
}
write('manual_polygons_frozen.json',dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),polygons=polys,outcomes_read=False,
    image_method='Sequential cv2.read from frame0; candidate_frames/ and old continuity/ are superseded seek previews and must not enter model computation.',
    reason='Conservative patches inside physically identifiable vehicle body, excluding obvious pole/neighbor/background; incomplete visible-object segmentation.',
    visual_review_inputs='frames/*_grid.png and frames/*_detail.png; sequential originals frames/{video}_{frame}.png',
    uncertainty='Union of fixed detector boxes for both objects, enlarged5pixels, minus visible cores. Contains uncertain vehicle/occluder/background pixels; it is not an object silhouette. Remaining scene is not pure background.'))
arrays={};events=[];checks=[];(OUT/'qa').mkdir(exist_ok=True)
for c in cases:
 v=c['video'];det=np.load(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz',allow_pickle=False)
 masks={};boxes={};images={}
 for time,frame in [('ref',c['reference_frame']),('query',c['query_frame'])]:
  path=OUT/f'frames/{v}_{frame}.png';img=cv2.imread(str(path));assert img is not None and img.shape==(720,1280,3);images[time]=path
  for obj in ['target','competitor']:
   rid=c['query_raw_id'] if time=='query' and obj=='target' else c[f'{obj}_{time}_raw_id']
   box=det['xyxy'][rid].astype(float).tolist();assert int(det['frame_idx'][rid])==frame;boxes[time,obj]=(rid,box)
   m=np.zeros((720,1280),np.uint8);cv2.fillPoly(m,[np.array(p,np.int32) for p in polys[c['id']][time][obj]],1)
   key=f'{c["id"]}__{time}__{obj}';arrays[key]=m.astype(bool);masks[time,obj]=key
  a=arrays[masks[time,'target']];b=arrays[masks[time,'competitor']];assert not (a&b).any()
  uncertain=np.zeros_like(a)
  for obj in ['target','competitor']:
   bb=boxes[time,obj][1];x1,y1=np.floor(np.array(bb[:2])-5).astype(int);x2,y2=np.ceil(np.array(bb[2:])+5).astype(int)
   uncertain[max(0,y1):min(720,y2),max(0,x1):min(1280,x2)]=True
  uncertain &=~(a|b);ukey=f'{c["id"]}__{time}__uncertain';arrays[ukey]=uncertain
  remainder=~(a|b|uncertain);arrays[f'{c["id"]}__{time}__remaining_scene']=remainder
  assert np.all(np.stack([a,b,uncertain,remainder]).sum(axis=0)==1)
  # Preserve full original image and show colored cores only; uncertainty has a
  # thin yellow contour so pixel evidence remains visible.
  overlay=img.copy()
  for mask,color in [(a,np.array([50,230,50])),(b,np.array([230,100,30]))]:overlay[mask]=(0.55*overlay[mask]+.45*color).astype(np.uint8)
  contours,_=cv2.findContours(uncertain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(0,220,220),1)
  x1,y1,x2,y2=c['detail_bounds'];crop=overlay[y1:y2,x1:x2];factor=min(4,1200/crop.shape[1]);crop=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(str(OUT/f'qa/{c["id"]}_{time}.png'),crop)
  checks.append(dict(case_id=c['id'],time=time,frame=frame,target_pixels=int(a.sum()),competitor_pixels=int(b.sum()),uncertain_pixels=int(uncertain.sum()),partition_disjoint_exhaustive=True,image_sha256=sha(path)))
 roles=[]
 for role,time,obj in [('target_ref','ref','target'),('competitor_ref','ref','competitor'),('query','query','target')]:
  rid,box=boxes[time,obj];other='competitor' if obj=='target' else 'target';frame=c['reference_frame'] if time=='ref' else c['query_frame']
  roles.append(dict(role=role,frame=frame,image=str(images[time].relative_to(ROOT)).replace('\\','/'),image_sha256=sha(images[time]),box=box,raw_id=rid,manual_key=masks[time,obj],other_manual_key=masks[time,other],uncertain_key=f'{c["id"]}__{time}__uncertain',physical_identity=c[f'{obj}_description'],physical_identity_status='visually_continuous_in_sequential_frames; GT only corroboration, no IoU ownership',gt_id_corroboration=c[f'{obj}_gt_id'],box_source='original_saved_YOLO26_detector_raw_video.npz',class_id=int(det['cls'][rid])))
 events.append(dict(id=c['id'],video=v,event_frame=c['event_frame'],roles=roles,masks='research_yolo26seg_20261114/holdout/manual_masks.npz',split_provenance=dict(source='research_identity_holdout_20261112/review/selected_events_B_RESERVED_v2.json',split='B_RESERVED',scope=c['scope']),additional_competitor_query=dict(frame=c['query_frame'],raw_id=c['competitor_query_raw_id'],box=boxes['query','competitor'][1],manual_key=masks['query','competitor']),manual_scope='Conservative visible cores, not full object masks; same fixed detector boxes in full/manual/auto.',identity_evidence='research_yolo26seg_20261114/holdout/sequential_clock_audit.json; corrected sequential visual continuity review'))
with (OUT/'manual_masks.npz').open('wb') as f:np.savez_compressed(f,**arrays)
manifest=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),events=events,masks_sha256=sha(OUT/'manual_masks.npz'),manual_polygons_sha256=sha(OUT/'manual_polygons_frozen.json'),outcomes_read=False,
             pairs=3,videos=3,scope='Outcome-blind new diagnostic pairs on reserved video frames; not full tracker improvement or correction of every source old_id failure.',
             source_selection='frozen_pairs.json and selection_disposition_25.json',input_correction='All manifest images sequentially decoded; seek previews were rejected before any model outcome.',
             comparison='Root freezes full/manual/auto at native and common_level0 MASA, identical ROI, class-car automatic maxIoU mapping; manual labels never used for automatic choice.',
             limitation='Small selected diagnostic sample, conservative core area differs across objects; missing auto mask remains a failure/unknown of extraction, not omitted. Manual cores cannot validate full-mask recall or full silhouette quality.')
write('manifest.json',manifest)
write('manual_mask_checks.json',dict(records=checks,outcomes_read=False,sha256=sha(OUT/'manual_masks.npz')))
write('absence_witness.json',dict(video='b23adb0d-72704b27',reference_frame=600,query_frame=606,images=[f'research_yolo26seg_20261114/holdout/frames/b23adb0d-72704b27_{f}.png' for f in [600,606]],panel='absence_witness_sequential.png',status='no_reliable_attributable_visible_target_core',note='Sequential review corrects the initial seek preview: a small red light remains at the left border in606. It cannot be confidently assigned to the previous vehicle from these two stills; do not claim total physical absence. No manual pair and no similarity test for this video.',outcomes_read=False))
print(json.dumps({'events':len(events),'roles':sum(len(e['roles']) for e in events),'masks_sha256':manifest['masks_sha256'],'manifest_sha256':sha(OUT/'manifest.json')}))
