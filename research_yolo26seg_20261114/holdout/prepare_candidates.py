import cv2
import datetime
import gzip
import hashlib
import json
import numpy as np
from pathlib import Path

ROOT=Path(r'E:\Dissertation'); OUT=Path(__file__).resolve().parent
def write(n,d): (OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'research_identity_holdout_20261112/review/selected_events_B_RESERVED_v2.json'
b=json.loads(source.read_text(encoding='utf8'))
protocol=dict(registered=datetime.datetime.now().astimezone().isoformat(),source=str(source.relative_to(ROOT)),source_sha256=sha(source),
 scope='Outcome-blind visual-content diagnostic on reserved B videos, not full tracking or project-unseen final evaluation.',
 selection='Review all 25 existing B proposals using original pixels and original detector/history boxes only. Choose at most one event per video, earliest event with two physically distinguishable cars and manually attributable visible cores in both reference and query. Prefer original fixed ROIs; retain failed eligibility and reasons. No YOLO26 segmentation or MASA outcomes consulted. If fewer than four eligible, keep honest smaller population.',
 identity='Read continuity in original raw frames and exact annotated anchors; dataset IDs only corroborate trajectory. IoU between rectangles never establishes pixel ownership. Unknown physical identity/absent target remains explicit and cannot be relabeled as failure or success.',
 acceptance=['Freeze selected cases, roles, ROIs, physical descriptions and manual polygons before reading segmentation/similarity outputs.', 'Full/manual/auto use the same original image, fixed ROI and association representation settings supplied by root; no new thresholds or weights.', 'Manual masks are conservative visible cores with separate uncertain area, not complete silhouettes. Candidate exclusions are preserved.', 'No GPU, tracker, downloads, shared catalog, plan or graph writes. All artifacts on E.'],
 roles=['target_ref','competitor_ref','query'],mask_schema='NPZ arrays named {case_id}__{role}__object and __uncertain, boolean HxW in original image coordinates; metadata explains physical object and fixed bbox. Additional competitor_query allowed.')
if not (OUT/'selection_protocol.json').exists():write('selection_protocol.json',protocol)
(OUT/'candidate_frames').mkdir(exist_ok=True);(OUT/'candidate_panels').mkdir(exist_ok=True)
records=[]
for video in b['videos']:
 source_events=[x for x in b['events'] if x['video']==video]
 archive=ROOT/source_events[0]['event_archive'];events={x['frame']:x for x in json.loads(gzip.decompress(archive.read_bytes()))['events']}
 detpath=ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/detections/raw_video.npz'
 with np.load(detpath,allow_pickle=False) as f: det={k:f[k].copy() for k in ['xyxy','cls','frame_idx']}
 cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{video}.mov'))
 frames={}
 for p in source_events:
  frame=p['frame'];q=p['proposal_keys'][0];row=events[frame]['rows'][q['row']];hist=row['history'][-1];ref=hist['frame'];rid=q['raw_id']
  assert row['tid']==q['old_id'] and int(det['frame_idx'][rid])==frame
  for f in [ref,frame]:
   if f not in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
    frames[f]=img;cv2.imwrite(str(OUT/f'candidate_frames/{video}_{f}.png'),img)
  refbox=det['xyxy'][hist['raw_id']].tolist();querybox=det['xyxy'][rid].tolist()
  crops=[]
  for f,box,label in [(ref,refbox,'reference'),(frame,querybox,'query')]:
   img=frames[f].copy();x1,y1,x2,y2=map(int,box)
   width=max(x2-x1,120);height=max(y2-y1,100)
   xx1=max(0,x1-width);xx2=min(img.shape[1],x2+width);yy1=max(0,y1-height);yy2=min(img.shape[0],y2+height)
   cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),1)
   crop=img[yy1:yy2,xx1:xx2];scale=min(600/crop.shape[1],360/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
   canvas=np.full((390,600,3),32,np.uint8);canvas[30:30+crop.shape[0],:crop.shape[1]]=crop
   cv2.putText(canvas,f'{video} {label} f{f} raw{hist["raw_id"] if label=="reference" else rid}',(5,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA);crops.append(canvas)
  panel=np.concatenate(crops,axis=1);name=f'{video}_{frame}.png';cv2.imwrite(str(OUT/'candidate_panels'/name),panel)
  records.append(dict(event_key=p['event_key'],video=video,frame=frame,reference_frame=ref,old_id=q['old_id'],raw_id=rid,reference_raw_id=hist['raw_id'],reference_bbox=refbox,query_bbox=querybox,panel=f'candidate_panels/{name}',frame_files=[f'candidate_frames/{video}_{ref}.png',f'candidate_frames/{video}_{frame}.png'],event_source=p['event_archive'],source_sha256=sha(archive)))
 cap.release()
write('candidate_manifest.json',dict(records=records,outcome_files_read=[],selection_protocol_sha256=sha(OUT/'selection_protocol.json')))
print(json.dumps({'candidates':len(records),'videos':b['videos'],'outcomes_read':False}))
