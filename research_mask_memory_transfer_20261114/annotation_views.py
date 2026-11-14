from common import *
import cv2,numpy as np
p=read(OUT/'design.json');ids=['00114999','00118240','00128803'];items=[]
for r,identity in zip(p['records'],ids):
 vid=r['video'];ap=ROOT/f'runs/z13_val20_matched_20261024/cache/{vid}/alignment.json';gp=ROOT/f'data/bdd100k/labels/box_track_20/val/{vid}.json';matches=read(ap)['matches'];labels=read(gp)
 for off in p['query_offsets']:
  fr=r['frame']+off;left=max((m for m in matches if m['video_frame_index']<=fr),key=lambda m:m['video_frame_index']);right=min((m for m in matches if m['video_frame_index']>=fr),key=lambda m:m['video_frame_index']);anchors=[]
  for m in [left,right]:
   row=labels[m['reference_index']];assert row['name']==m['reference_name'];box=next((x['box2d'] for x in row['labels'] if x['id']==identity and 'box2d' in x),None);anchors.append(dict(frame=m['video_frame_index'],annotation=m['reference_name'],box=box))
  have=[x for x in anchors if x['box'] is not None]
  if have:
   a,b=have[0],have[-1];v=(fr-a['frame'])/(b['frame']-a['frame']) if b['frame']!=a['frame'] else 0;box=np.array([(1-v)*a['box'][k]+v*b['box'][k] for k in ['x1','y1','x2','y2']]);cx,cy=(box[:2]+box[2:])/2
  else:cx=(r['box'][0]+r['box'][2])/2;cy=(r['box'][1]+r['box'][3])/2
  side=96 if vid.startswith('b23a') else 144;x=int(max(0,min(1280-side,round(cx-side/2))));y=int(max(0,min(720-side,round(cy-side/2))));im=cv2.imread(str(OUT/f'{vid}_{fr}.png'));assert im is not None;cv2.imwrite(str(OUT/f'annotation_view_{vid}_{fr}.png'),cv2.resize(im[y:y+side,x:x+side],(600,600)))
  items.append(dict(case=r['id'],video=vid,frame=fr,identity_guide=identity,annotation_only_crop=[x,y,x+side,y+side],anchors=anchors,alignment_sha256=sha(ap),labels_sha256=sha(gp)))
save('annotation_views.json',dict(records=items,scope='Ground-truth boxes on aligned annotated frames guide human localization only. Interpolated centers are not segmentation labels; queryboxes/crops/IDs never enter inference. Third target has no exact annotation on requestedframes and no distinguishable visible core in the right-edge image.'))
