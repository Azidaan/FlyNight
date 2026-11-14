from common import *
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json');choice=read(OUT/'choices.json');res=read(OUT/'result.json');ds=np.load(OUT/'descriptors.npz');maps=np.load(OUT/'maps.npz');selected=np.load(OUT/'selected_masks.npz');A=ROOT/'research_mask_propagation_20261114';B=ROOT/'research_mask_memory_transfer_20261114';ap=np.load(A/'predictions.npz');am=np.load(A/'memory_predictions.npz');bp=np.load(B/'predictions.npz');ai=np.load(ROOT/read(A/'protocol.json')['initial_mask'])['1159_target'].astype(bool);bi=np.load(B/'initial_masks.npz');at=np.load(A/'evaluation_truth.npz');bt=np.load(B/'query_cores.npz');names=p['candidates'];verified=[];descriptor_error=0.;font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
def get(c,fr):
 return [ap['sam__'+str(fr)],am['initial_memory_only__'+str(fr)],ap['flow__'+str(fr)],ap['static__'+str(fr)]] if c['id']=='old2179' else [bp[c['id']+'__'+n+'__'+str(fr)] for n in names]
def values(image,depth,ordered,mask):
 y,x=np.where(mask);sample=depth[y,x];mid=(np.searchsorted(ordered,sample,'left')+np.searchsorted(ordered,sample,'right'))/(2*len(ordered));quant=[.1,.25,.5,.75,.9];return dict(RGB=np.quantile(image[y,x]/255,quant,axis=0).flatten(),Depth=np.quantile(mid,quant),Y=np.quantile(y/(depth.shape[0]-1),quant))
def ranks(cost):
 return np.array([1+sum(v<x for v in cost)+.5*(sum(v==x for v in cost)-1) for x in cost])
for c in p['cases']:
 cid=c['id'];ref=ai if cid=='old2179' else bi[cid];sheet=Image.new('RGB',(1200,330*len(c['queries'])),'#20232a');draw=ImageDraw.Draw(sheet)
 for i,fr in enumerate([c['reference']]+c['queries']):
  im=cv2.cvtColor(cv2.imread(str(ROOT/c['image_dir']/c['image_pattern'].format(frame=fr))),cv2.COLOR_BGR2RGB);depth=maps[cid+'__'+str(fr)];ordered=np.sort(depth.ravel());masks=[ref] if i==0 else get(c,fr);costs={k:[] for k in ['RGB','Depth','Y']}
  for j,mask in enumerate(masks):
   if not mask.any():
    for k in costs:costs[k].append(np.inf)
    continue
   data=values(im,depth,ordered,mask);prefix=cid+'__reference__' if i==0 else cid+'__'+str(fr)+'__'+names[j]+'__'
   for k,datax in data.items():
    err=float(np.max(np.abs(datax-ds[prefix+k])));descriptor_error=max(descriptor_error,err);assert err<1e-12,(prefix+k,err)
    if i:costs[k].append(float(np.abs(datax-first[k]).mean()))
   if not i:first=data
  if not i:continue
  chosen=next(v for v in choice['records'] if v['case']==cid and v['frame']==fr);valid=np.isfinite(costs['RGB']);computed={k:np.array(v) for k,v in costs.items()};computed['RGB+Depth']=ranks(costs['RGB'])+ranks(costs['Depth']);computed['RGB+Y']=ranks(costs['RGB'])+ranks(costs['Y']);truth=at[str(fr)] if cid=='old2179' else bt[c['video']+'__'+str(fr)]
  for key,v in computed.items():
   idx=int(np.argmin(np.where(valid,v,np.inf)));assert chosen['selected'][key]==names[idx];mask=masks[idx];assert np.array_equal(selected[cid+'__'+str(fr)+'__'+key],mask);known=truth>=0;cm=np.bincount((truth[known]==1).astype(int)*2+mask[known].astype(int),minlength=4);row=next(x for x in res['records'] if x['case']==cid and x['frame']==fr and x['method']==key);assert [row['tp'],row['fp'],row['fn']]==[cm[3],cm[1],cm[2]];verified.append(dict(case=cid,frame=fr,selector=key,chosen=names[idx]))
  if cid=='old2179':x,y,rr,b=900,200,1280,450
  else:x,y,rr,b=next(w['annotation_only_crop'] for w in read(B/'annotation_views.json')['records'] if w['video']==c['video'] and w['frame']==fr)
  for col,(label,mask) in enumerate([('Кадр',None),('Цвет',selected[cid+'__'+str(fr)+'__RGB']),('Цвет + глубина',selected[cid+'__'+str(fr)+'__RGB+Depth']),('Цвет + высота',selected[cid+'__'+str(fr)+'__RGB+Y'])]):
   a=im.copy()
   if mask is not None:a[mask]=(a[mask]*.4+np.array([30,235,50])*.6).astype('uint8')
   patch=a[y:b,x:rr];scale=min(300/patch.shape[1],295/patch.shape[0]);sz=(round(patch.shape[1]*scale),round(patch.shape[0]*scale));sheet.paste(Image.fromarray(cv2.resize(patch,sz)),(col*300,(i-1)*330+30));draw.text((col*300+4,(i-1)*330+5),str(fr)+' · '+label,font=font,fill='white')
 sheet.save(OUT/(cid+'_choices.png'))
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('audit.json',dict(passed=True,descriptor_max_absolute_error=descriptor_error,choices_reconstructed=len(verified),checks=verified,method='Independent fullmap sorted lookup of percentile ranks, explicit tie counts, binary confusion table and equality of chosen masks.',source_unchanged=True,map_sha256=sha(OUT/'maps.npz'),artifacts={n:sha(OUT/n) for n in ['result.json','choices.json','descriptors.npz','selected_masks.npz']}));print('DEPTH_CHOICE_AUDITED',len(verified),descriptor_error)
