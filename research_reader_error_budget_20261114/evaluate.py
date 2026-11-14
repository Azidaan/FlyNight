from official import *
from scipy.spatial import cKDTree,ConvexHull
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');src=ROOT/'research_supervised_contour_reader_20261114';z=np.load(src/'dataset.npz');old=np.load(src/'predictions.npz');new=np.load(OUT/'predictions.npz');weights=np.load(OUT/'weights.npz');grid=read(OUT/'grid.json');val=[r for r in read(src/'dataset.json')['records'] if r['split']=='val'];points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);yy,xx=np.indices((391,391));nearest=cKDTree(points).query(np.c_[xx.ravel(),yy.ravel()])[1].reshape(391,391);sup=cv2.fillConvexPoly(np.zeros((391,391),'uint8'),points[ConvexHull(points).vertices],1).astype(bool);rows=[];worst=0;checks=0
for q in val:
 key=q['id'];t=z[key+'__truth'];k=(t>=0)&sup;inside=z[key+'__inside'];lab=z[key+'__labels'];p0=(t==1)&k;ne=(t!=1)&k;ni=ne&inside;hard=lab[nearest]>=0;oracle=next(x['max_iou'] for x in grid['oracle'] if x['key']==key)
 for arm in ['Fly','Retina','Geometry','RGB']:
  for seed in [42,43,44]:
   prefix=f'{arm}__{seed}';xi=(z[key+'__'+arm]-weights[prefix+'__mean'])/weights[prefix+'__sd'];v=np.maximum(xi@weights[prefix+'__0.weight'].T+weights[prefix+'__0.bias'],0)@weights[prefix+'__2.weight'].T+weights[prefix+'__2.bias'];diff=float(np.max(np.abs(v-new[prefix+'__'+key])));worst=max(worst,diff);assert diff<3e-4;checks+=1
   for mode in ['original','fractional']:
    a=old[f'labels__{arm}__{seed}__{key}'] if mode=='original' else new[prefix+'__'+key];m=a.argmax(1)[nearest]==1;tp=int((m&p0).sum());fp=int((m&ne).sum());fn=int((~m&p0).sum());iou0=tp/(tp+fp+fn);assert iou0<=oracle+1e-12;rows.append(dict(key=key,video=q['video'],physical=q['physical'],arm=arm,seed=seed,mode=mode,iou=iou0,recall=tp/int(p0.sum()),inside_fpr=int((m&ni).sum())/int(ni.sum()) if ni.any() else None,errors_on_unlabelled_cells=int((((m&ne)|(~m&p0))&~hard).sum()),errors_on_labelled_cells=int((((m&ne)|(~m&p0))&hard).sum()),target=tp+fn,tp=tp,fp=fp,fn=fn,oracle=oracle))
def mean(vals):
 s=[x for x in vals if x is not None];return float(np.mean(s)) if s else None
def agg(qq):
 physical=sorted({r['physical'] for r in qq});return {m:mean([mean([r[m] for r in qq if r['physical']==g]) for g in physical]) for m in ['iou','recall','inside_fpr','oracle']}
vv=sorted({r['video'] for r in val});stats={}
for mode in ['original','fractional']:
 for arm in ['Fly','Retina','Geometry','RGB']:
  rr=[r for r in rows if r['mode']==mode and r['arm']==arm];vid={v:agg([r for r in rr if r['video']==v]) for v in vv};stats[mode+'__'+arm]=dict(per_video=vid,mean={m:mean([g[m] for g in vid.values()]) for m in ['iou','recall','inside_fpr','oracle']})
of=stats['original__Fly'];ff=stats['fractional__Fly'];gain=ff['mean']['iou']>of['mean']['iou'] and sum(ff['per_video'][v]['iou']>of['per_video'][v]['iou'] for v in vv)>=2;fpr=ff['mean']['inside_fpr']<of['mean']['inside_fpr'] and ff['mean']['recall']>=of['mean']['recall'];increment=ff['mean']['iou']>max(stats['fractional__'+n]['mean']['iou'] for n in ['Geometry','Retina']);save('result.json',dict(scientific_passed=bool(gain and fpr and increment),gain=bool(gain),fpr_with_recall=bool(fpr),fly_increment=bool(increment),stats=stats,rows=rows,scope='Only frozen stored representations, supervised readout. Oracle mask per cell sees query labels so never used in decisions.'))
save('audit.json',dict(passed=True,independent_model_outputs=checks,max_abs=worst,grid_optimality_certificates=len(grid['oracle']),result_sha256=sha(OUT/'result.json')))
q=next(x for x in val if x['frame']==1165);key=q['id'];t=z[key+'__truth'];pr=read(src/'protocol.json');imgs=zipfile.ZipFile(ROOT/pr['imagezip']);im=cv2.imdecode(np.frombuffer(imgs.read('bdd100k/images/seg_track_20/val/'+q['video']+'/'+q['name']),'uint8'),1);x,y,x2,y2=q['eye'];im=cv2.cvtColor(cv2.resize(im[y:y2,x:x2],(391,391)),cv2.COLOR_BGR2RGB);sheet=Image.new('RGB',(1564,429),'#22282c');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for i,(title,m) in enumerate([('Разметка',(t==1)&sup),('Предел сетки',new['oracle__'+key][nearest]&sup),('Исходный FlyVis',(old['labels__Fly__42__'+key].argmax(1)[nearest]==1)&sup),('Долевые метки FlyVis',(new['Fly__42__'+key].argmax(1)[nearest]==1)&sup)]):
 a=im.copy();a[m]=(a[m]*.45+np.array([40,233,61])*.55).astype('uint8');co,_=cv2.findContours(((t==1)&sup).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(a,co,-1,(255,210,25),1);sheet.paste(Image.fromarray(a),(391*i,32));dr.text((391*i+5,6),title,font=font,fill='white')
sheet.save(OUT/'reader_1165.jpg',quality=91)
