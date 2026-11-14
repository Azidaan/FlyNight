from common import *
import cv2,numpy as np
draft=read(OUT/'manual_query_cores.json');groups=draft['polygons_in_3x_detail'];groups['1183']['neighbor']=[];groups['1183']['occluder'].append([[1020,315],[1120,315],[1120,447],[1020,438]]);groups['1207']['neighbor']=[[[977,290],[1018,287],[1018,345],[971,357],[950,336]]]
arrays={}
for frame,regions in groups.items():
 truth=np.full((720,1280),-1,np.int8)
 for name,cls in [('background',0),('target',1),('neighbor',2),('occluder',3)]:
  for poly in regions[name]:
   m=cv2.fillPoly(np.zeros(truth.shape,'uint8'),[np.rint(np.array(poly)/3+[900,200]).astype('int32')],1).astype(bool);assert (truth[m]<0).all();truth[m]=cls
 if 'wheel' in regions:
  x,y,r=regions['wheel'];m=cv2.circle(np.zeros(truth.shape,'uint8'),(round(900+x/3),round(200+y/3)),round(r/3),1,-1).astype(bool);assert (truth[m]<0).all();truth[m]=1
 arrays[frame]=truth;im=cv2.imread(str(OUT/f'frame_{frame}.png'))
 for cls,col in [(0,[180,180,180]),(1,[40,240,30]),(2,[255,90,40]),(3,[30,30,245])]:
  m=truth==cls;im[m]=(im[m]*.4+np.array(col)*.6).astype('uint8')
 cv2.imwrite(str(OUT/f'reviewed_labels_{frame}.png'),cv2.resize(im[200:450,900:1280],(1140,750)))
with (OUT/'manual_query_cores_v2.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('annotation_review.json',dict(created=now(),polygons_in_3x_detail=groups,draft_sha256=sha(OUT/'manual_query_cores.npz'),reviewed_sha256=sha(OUT/'manual_query_cores_v2.npz'),changes='Before inference:1183draft neighbor region intersects walking person; remove it and annotate a safe jacket core asoccluder.1207neighborregion narrowed to clearcar pixels. Targetcoresunchanged.',review='Visual review by assistant, not an independent human annotation.',counts={k:{str(c):int((v==c).sum()) for c in [0,1,2,3]} for k,v in arrays.items()},scope='Partial cores; no complete-mask or groundtruth3D claim.'))
weights=ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt';assert sha(weights)=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
paths=[OUT/n for n in ['common.py','worker.py','design.json','selection.json','frames.json','manual_query_cores_v2.npz','annotation_review.json']]+[weights,ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz',ROOT/'tracking/gpu_lock.py']
paths += list((ROOT/'tools/sam2-runtime-20261027/sam2').rglob('*.py'))+list((ROOT/'tools/sam2-runtime-20261027/sam2/configs').rglob('*.yaml'))+[ROOT/r['path'] for r in read(OUT/'frames.json')['records']]
save('protocol.json',dict(registered=now(),design=read(OUT/'design.json'),inputs={rel(f):sha(f) for f in paths},weights=rel(weights),initial_mask='research_visible_part_2179_20261114/mask_review/masks_reviewed.npz',query_frames=[1165,1171,1183,1195,1207],query_labels='manual_query_cores_v2.npz',familiar_query_labels='research_visible_part_2179_20261114/mask_review/masks_reviewed.npz',precision='Float32, noautocast,TF32disabled, sameweights, apply_postprocessingFalse, no compilation, official thresholdlogit0',label_review='Newcores visually inspected beforepredictions; target is the stationary far SUV.1171frontglass/hood remainvisible while a nearer movingvehicle hides previous rearwindow patch; laterframes reveal the sameSUV rearbody andwheel.',scope='One existing video, later occlusionstates. Comparisons on knownpixels; outsidecoresunknown. Same first manualmask for allthree arms. No futureinputs supplied.'))
journal('Перед прогоном mask_propagation_20261114 закреплены ручные видимые области1171,1183,1195,1207 и проверено соответствие кадров1159/1165 архиву. При визуальном просмотре исправлены две смешанные отрицательные области: человек отделён от машины; целевые области не менялись. Источники: annotation_review.json, protocol.json. Модель ещё не запускалась. На1171 сосед скрывает прежний участок цели; позже открываются кузов и колесо, поэтому копирование прошлой части не равнозначно полной доступной маске.')
save('queue_run.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(OUT/n):sha(OUT/n) for n in ['worker.py','common.py','protocol.json']},jobs=[dict(entry=rel(OUT/'worker.py'),timeout=600)]));print('PROPAGATION_FROZEN')
