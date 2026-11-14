from official import *
from scipy.spatial import cKDTree,ConvexHull
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');db=read(OUT/'dataset.json');z=np.load(OUT/'dataset.npz');pred=np.load(OUT/'predictions.npz');points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);yy,xx=np.indices((391,391));nearest=cKDTree(points).query(np.c_[xx.ravel(),yy.ravel()])[1].reshape(391,391);sup=cv2.fillConvexPoly(np.zeros((391,391),'uint8'),points[ConvexHull(points).vertices],1).astype(bool);val=[r for r in db['records'] if r['split']=='val'];rows=[]
def evaluate(m,k,t,inside):
 pos=k&(t==1);neg=k&(t!=1);n=neg&inside;tp=int((m&pos).sum());fp=int((m&neg).sum());fn=int((~m&pos).sum());return dict(iou=tp/(tp+fp+fn) if pos.any() else None,recall=tp/int(pos.sum()) if pos.any() else None,inside_fpr=int((n&m).sum())/int(n.sum()) if n.any() else None,tp=tp,fp=fp,fn=fn,inside_n=int(n.sum()),inside_fp=int((n&m).sum()),neighbor_fp=int((m&k&(t==2)).sum()),background_fp=int((m&k&(t==0)).sum()))
for q in val:
 key=q['id'];truth=z[key+'__truth'];k=(truth>=0)&sup;inside=z[key+'__inside']
 for mode in ['labels','proxy']:
  for arm in ['Fly','Retina','RGB','Geometry']:
   for seed in p['seeds']:
    source=f'{mode}__{arm}__{seed}__{key}';m=(pred[source].argmax(1)[nearest]==1);rows.append(dict(request=key,video=q['video'],physical=q['physical'],mode=mode,arm=arm,seed=seed,**evaluate(m,k,truth,inside)))
 rows.append(dict(request=key,video=q['video'],physical=q['physical'],mode='baseline',arm='Box',seed=None,**evaluate(inside,k,truth,inside)))
def mn(seq):
 clean=[x for x in seq if x is not None];return float(np.mean(clean)) if clean else None
means={};total={}
for (mode,arm) in sorted({(r['mode'],r['arm']) for r in rows}):
 name=mode+'__'+arm;sub=[r for r in rows if r['mode']==mode and r['arm']==arm];means[name]={}
 for v in sorted({r['video'] for r in sub}):
  vrows=[x for x in sub if x['video']==v];groups=sorted({x['physical'] for x in vrows});means[name][v]={met:mn([mn([x[met] for x in vrows if x['physical']==g]) for g in groups]) for met in ['iou','recall','inside_fpr']}
 total[name]={met:mn([d[met] for d in means[name].values()]) for met in ['iou','recall','inside_fpr']}
fly=means['labels__Fly'];primary=all(d['recall'] is not None and d['inside_fpr'] is not None and d['recall']>=.5 and d['inside_fpr']<=.1 for d in fly.values());wins=sum(d['iou']>max(means['labels__Retina'][v]['iou'],means['labels__Geometry'][v]['iou']) for v,d in fly.items());inc=len(fly)==3 and wins>=2 and total['labels__Fly']['iou']>max(total['labels__Retina']['iou'],total['labels__Geometry']['iou']);dense=total['labels__Fly']['inside_fpr']<total['proxy__Fly']['inside_fpr'] and total['labels__Fly']['iou']>=total['proxy__Fly']['iou']
save('result.json',dict(records=rows,per_video=means,video_equal=total,outperformed_controls_scenes=wins,quality_passed=bool(primary),increment_passed=bool(inc),dense_passed=bool(dense),scientific_passed=bool(primary and inc and dense),scope='All measurements on official masks within the convex eye footprint. Validation video count tiny. A car/truck/bus instance is first matched to the supplied candidate box. This is conditional segmentation; it cannot prove tracking identity.'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);imgs=zipfile.ZipFile(ROOT/p['imagezip'])
for v in sorted({q['video'] for q in val}):
 cand=[q for q in val if q['video']==v];q=next((q for q in cand if q['frame']==1165),cand[0]);key=q['id'];truth=z[key+'__truth'];x,y,x2,y2=q['eye'];im=cv2.imdecode(np.frombuffer(imgs.read(f'bdd100k/images/seg_track_20/val/{v}/'+q['name']),'uint8'),1);image=cv2.cvtColor(cv2.resize(im[y:y2,x:x2],(391,391)),cv2.COLOR_BGR2RGB);sheet=Image.new('RGB',(391*4,437),'#20252b');dr=ImageDraw.Draw(sheet)
 for i,(arm,name) in enumerate([('Truth','Официальный контур'),('Fly','Считыватель FlyVis'),('Retina','Тот же вход без FlyVis'),('Geometry','Только геометрия')]):
  classes=truth if i==0 else pred['labels__'+arm+'__42__'+key].argmax(1)[nearest];im=image.copy();pos=(classes==1)&sup;neighbor=(classes==2)&sup;im[pos]=(im[pos]*.45+np.array([31,231,60])*.55).astype('uint8');im[neighbor]=(im[neighbor]*.55+np.array([250,65,40])*.45).astype('uint8');co,_=cv2.findContours(((truth==1)&sup).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,co,-1,(251,214,30),1);sheet.paste(Image.fromarray(im),(391*i,30));dr.text((391*i+5,5),name,font=font,fill='white')
 sheet.save(OUT/(v+'_masks.jpg'),quality=91)
print('DENSE_READER',total,'gates',primary,inc,dense,flush=True)
