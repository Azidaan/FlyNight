from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import sha,now
import numpy as np,cv2
source=np.load(HERE/'masks.npz',allow_pickle=False)
# Conservative image-only visible regions chosen before seeing any MASA score.
# These mark visible cores, not complete object segmentation.
polygons={
'1159':{'red':[[[1122,255],[1190,255],[1210,266],[1220,281],[1205,294],[1167,292],[1130,278]]], 'black':[[[941,268],[1020,265],[1040,280],[1020,309],[947,309],[916,322],[905,310],[920,283]]]},
'1164':{'red':[[[1122,254],[1189,255],[1210,265],[1219,280],[1208,288],[1166,280],[1140,269]]], 'black':[[[1003,269],[1078,264],[1086,276],[1068,311],[995,311],[964,317],[961,305],[987,284]],[[1170,307],[1263,305],[1275,327],[1261,350],[1170,342]]]},
'1165':{'red':[[[1122,254],[1189,255],[1210,265],[1219,280],[1208,288],[1166,280],[1140,269]]], 'black':[[[1010,270],[1086,264],[1091,276],[1072,311],[1002,311],[967,318],[963,305],[990,285]],[[1170,307],[1270,305],[1277,328],[1264,350],[1170,342]]]}
}
output={};records=[]
for frame,regions in polygons.items():
 occ=source[f'{frame}_occluder'];raw_red=source[f'{frame}_target_raw'];raw_black=source[f'{frame}_neighbor_raw'];cores={}
 for kind,ps in regions.items():
  mask=np.zeros(occ.shape,np.uint8)
  for poly in ps:cv2.fillPoly(mask,[np.asarray(poly,np.int32)],1)
  cores[kind]=mask.astype(bool)&~occ&(raw_red if kind=='red' else raw_black)
 overlap=cores['red']&cores['black'];cores['red']&=~overlap;cores['black']&=~overlap
 vehicle_union=raw_red|raw_black
 uncertain=vehicle_union&~(cores['red']|cores['black']|occ)
 background=~(vehicle_union|occ)
 for kind,mask in [*cores.items(),('foreground',occ),('background',background),('uncertain',uncertain),('red_sam',raw_red),('black_sam',raw_black)]:output[f'{frame}_{kind}']=mask
 image=cv2.imread(str(HERE.parent/'frames'/f'{frame}.png'));view=image.copy()
 for kind,color in [('red',(30,220,30)),('black',(220,80,20)),('foreground',(20,20,220)),('uncertain',(150,150,150))]:
  mask=output[f'{frame}_{kind}'];view[mask]=(.45*view[mask]+.55*np.asarray(color)).astype(np.uint8)
 cv2.imwrite(str(HERE/f'{frame}_core_review.png'),cv2.resize(view[240:420,880:1280],(1000,450),interpolation=cv2.INTER_NEAREST))
 records.append(dict(frame=int(frame),counts={k:int(output[f'{frame}_{k}'].sum()) for k in ['red','black','foreground','background','uncertain']},sam_overlap_pixels=int((raw_red&raw_black).sum()),all_classes_disjoint=True))
with (HERE/'masks_core.npz').open('xb') as f:np.savez_compressed(f,**output)
(HERE/'masks.json').write_text(json.dumps(dict(created=now(),file='masks_core.npz',sha256=sha(HERE/'masks_core.npz'),automatic_source='masks.npz',automatic_sha256=sha(HERE/'masks.npz'),keys='{frame}_red, {frame}_black, {frame}_foreground, {frame}_background, {frame}_uncertain',frames=[1159,1164,1165],semantic={'red':'Conservative visible core of far2179SUV; manual polygons intersect SAM, not full silhouette.','black':'Conservative visible core of near2445SUV; manual polygons intersect SAM, not full silhouette.','foreground':'SAM gray pedestrian/red pedestrian/cart union.','background':'Outside both raw SAM vehicle masks and foreground. May include unannotated distant objects, therefore remaining scene rather than pure road background.','uncertain':'Unassigned raw vehicle-mask region; must not silently treat as clean background.'},polygons=polygons,records=records,limitation='Raw SAM merges cars on1164/1165. This conservative visual-core diagnostic does not prove complete visible-object segmentation. Do not interpret black-only/core effects as sole neighbor cause while uncertain pixels remain uncontrolled. No MASA outcomes consulted.'),ensure_ascii=False,indent=2))
print('CORE_MASKS_READY',records)
