from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,time
from scipy.spatial import cKDTree,ConvexHull
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');f=read(OUT/'fit_result.json');assert sha(OUT/'predictions.npz')==f['predictions_sha256']
pred=np.load(OUT/'predictions.npz',allow_pickle=False);manual=np.load(OUT/'manual_test_masks.npz',allow_pickle=False)
points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);grid=np.stack(np.meshgrid(np.arange(391),np.arange(391)),axis=-1);nearest=cKDTree(points).query(grid.reshape(-1,2))[1].reshape(391,391);inside=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),points[ConvexHull(points).vertices],1).astype(bool)
rows=[];rasters={}
for r in [r for r in p['records'] if r['split']=='test']:
 pos=manual[r['id']+'__positive'].astype(bool);neg=manual[r['id']+'__negative'].astype(bool);assert not (pos&neg).any();known=(pos|neg)&inside
 for size in ['small','all']:
  for arm in p['arms']:
   estimate=pred[size+'__'+arm+'__'+r['id']].argmax(1)[nearest].astype(bool);tp=int((estimate&pos&inside).sum());fp=int((estimate&neg&inside).sum());fn=int((~estimate&pos&inside).sum());tn=int((~estimate&neg&inside).sum())
   rows.append(dict(id=r['id'],video=r['video'],frame=r['frame'],size=size,arm=arm,tp=tp,fp=fp,fn=fn,tn=tn,iou=tp/max(tp+fp+fn,1),recall=tp/max(tp+fn,1),known_positive=tp+fn,known_negative=fp+tn,known_fraction=float(known.sum()/inside.sum())))
   rasters[size+'__'+arm+'__'+r['id']]=estimate.astype(np.uint8)
summary={}
for size in ['small','all']:
 summary[size]={}
 for arm in p['arms']:
  videos={v:float(np.mean([r['iou'] for r in rows if r['video']==v and r['size']==size and r['arm']==arm])) for v in p['test_videos']}
  summary[size][arm]=dict(videos=videos,mean=float(np.mean(list(videos.values()))))
a=summary['all']['FlyVis34_G6'];b=summary['all']['Retina_history34_G6'];diff={v:a['videos'][v]-b['videos'][v] for v in p['test_videos']}
feasible=a['mean']>=.5 and min(a['videos'].values())>=.3;gain=float(np.mean(list(diff.values())));benefit=gain>0 and sum(x>0 for x in diff.values())>=3
save('result.json',dict(created=now(),records=rows,summary=summary,flyvis_difference=diff,mean_difference=gain,mask_feasibility=feasible,additional_flyvis_benefit=benefit,scientific_passed=feasible and benefit,manual_sha256=sha(OUT/'manual_test_masks.npz'),predictions_sha256=sha(OUT/'predictions.npz'),source_sha256=sha(__file__),metric='Target IoU only on manually certain positive/negative pixels inside eye support. Unknown boundaries excluded; not full contour IoU. Mean equally weighted by held-outvideo.',causal_limit='Different input representations with matched reader. Data-size comparison within this new binary task; no direct quantitative attribution from earlier three-class scene-specific probe.'))
with (OUT/'raster_predictions.npz').open('xb') as h:np.savez_compressed(h,**rasters)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);smallfont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for r in [r for r in p['records'] if r['split']=='test']:
 x1,y1,x2,y2=r['crop'];original=cv2.cvtColor(cv2.resize(cv2.imread(str(ROOT/r['image']))[y1:y2,x1:x2],(391,391)),cv2.COLOR_BGR2RGB)
 sheet=Image.new('RGB',(391*5,455),'#171b24');draw=ImageDraw.Draw(sheet);draw.text((8,3),r['id']+' | independent video | green = predicted target',font=font,fill='white')
 for k,arm in enumerate(['original','manual','FlyVis34_G6','Retina_history34_G6','Geometry6']):
  patch=original.copy()
  if arm=='manual':
   pos=manual[r['id']+'__positive'].astype(bool);neg=manual[r['id']+'__negative'].astype(bool);unknown=~(pos|neg);patch[pos]=(patch[pos]*.4+np.array([255,160,0])*.6).astype(np.uint8);patch[unknown]=(patch[unknown]*.45+np.array([160,160,160])*.55).astype(np.uint8)
  elif arm!='original':
   mask=rasters['all__'+arm+'__'+r['id']].astype(bool)&inside;patch[mask]=(patch[mask]*.4+np.array([50,230,80])*.6).astype(np.uint8);patch[~inside]=(patch[~inside]*.3).astype(np.uint8)
  sheet.paste(Image.fromarray(patch),(k*391,57));title={'original':'Original + detector ROI','manual':'Manual: orange core, gray unknown','FlyVis34_G6':'FlyVis + same ROI','Retina_history34_G6':'Same history without FlyVis','Geometry6':'ROI geometry only'}[arm]
  if arm not in ['original','manual']:title+=f" | IoU {next(x['iou'] for x in rows if x['id']==r['id'] and x['arm']==arm and x['size']=='all'):.3f}"
  draw.text((k*391+5,30),title,font=smallfont,fill='white')
  if arm=='original':
   b=r['box'];coords=[(b[0]-x1)*391/(x2-x1),57+(b[1]-y1)*391/(y2-y1),(b[2]-x1)*391/(x2-x1),57+(b[3]-y1)*391/(y2-y1)];draw.rectangle(coords,outline='#00ffff',width=2)
 sheet.save(OUT/'qa'/f"comparison_{r['id']}.png")
print(json.dumps(dict(summary=summary,feasible=feasible,benefit=benefit),ensure_ascii=False))
