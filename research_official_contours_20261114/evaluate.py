from official import *
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');db=read(OUT/'cases.json');pred=np.load(OUT/'predictions.npz');z=zipfile.ZipFile(ROOT/p['labelzip']);rows=[];cache={};absent=[]
for c in db['cases']:
 v=c['video'];cid=c['id']
 if v not in cache:cache[v]=labels(z,v)
 for j in [1,2,4]:
  r=cache[v][c['index']+j];t,k,other=masks(z,v,r,c['target'])
  for s in p['arms']:
   m=pred[f'{cid}__{s}__{j}'];rows.append(dict(case=cid,video=v,offset=j,frame_index=c['index']+j,source=s,**measure(m,t,k,other)))
means={}
for v in p['holdout']:
 means[v]={}
 for s in p['arms']:
  obs=[r['iou'] for r in rows if r['video']==v and r['source']==s and r['iou'] is not None];means[v][s]=float(np.mean(obs)) if obs else None
usable=[v for v,m in means.items() if m['box_first'] is not None];assert usable;avg={s:float(np.mean([means[v][s] for v in usable])) for s in p['arms']};rng=np.random.default_rng(42);idx=rng.integers(len(usable),size=(10000,len(usable)));diff={}
for s in ['flow_same_box_initial','static_same_box_initial','box_ordinary','yolo_first','manual_full_first']:
 a=np.array([means[v]['box_first']-means[v][s] for v in usable]);ci=np.quantile(a[idx].mean(1),[.025,.975]);diff[s]=dict(delta=float(a.mean()),low=float(ci[0]),high=float(ci[1]),wins=int((a>0).sum()),ties=int((a==0).sum()))
hard=next(r for r in read(OUT/'old_result.json')['records'] if r['frame']==1171 and r['method']=='box');oldpass=hard['iou']>=.5 and hard['precision'] is not None and hard['precision']>=.90;win=all(diff[s]['low']>0 for s in ['flow_same_box_initial','static_same_box_initial']) and len(usable)==len(p['holdout']);quality=avg['box_first']>=.5 and len(usable)==len(p['holdout'])
save('result.json',dict(records=rows,means=means,video_equal_means=avg,paired_intervals=diff,n_video=len(usable),n_cases=len(db['cases']),visible_queries=sum(r['source']=='box_first' and r['iou'] is not None for r in rows),without_positive=sum(r['source']=='box_first' and r['iou'] is None for r in rows),old_passed=bool(oldpass),control_passed=bool(win),quality_passed=bool(quality),scientific_passed=bool(oldpass and win and quality),scope='Unknown/ignore/crowd excluded. Full visible contours where labeled; target not annotated may include complete occlusion. No HOTA or global identity metric.'))
# Predetermined visual sampling: first case, hardest and best eligible box query, identified explicitly.
boxrows=[r for r in rows if r['source']=='box_first' and r['iou'] is not None];ranked=sorted(boxrows,key=lambda r:(r['iou'],r['case'],r['offset']));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for tag,rr in [('first',boxrows[0]),('worst',ranked[0]),('best',ranked[-1])]:
 c=next(c for c in db['cases'] if c['id']==rr['case']);j=rr['offset'];r=cache[c['video']][rr['frame_index']];t,k,other=masks(z,c['video'],r,c['target']);im=cv2.imread(str(ROOT/c['frames'][j]['image']));yy,xx=np.where(t);b=c['past_box'];x=max(0,int(min(xx.min(),b[0])-50));y=max(0,int(min(yy.min(),b[1])-50));x2=min(im.shape[1],int(max(xx.max(),b[2])+50));y2=min(im.shape[0],int(max(yy.max(),b[3])+50));sheet=Image.new('RGB',(420*4,340),'#20252b');dr=ImageDraw.Draw(sheet)
 for i,(method,title) in enumerate([('truth','Официальная маска'),('box_first','Рамка в SAM 2'),('flow_same_box_initial','Перенос движением'),('yolo_first','Маска YOLO в SAM 2')]):
  vis=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=t if method=='truth' else pred[f"{c['id']}__{method}__{j}"];vis[m]=(vis[m]*.4+np.array([40,235,60])*.6).astype('uint8');cs,_=cv2.findContours(t.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(vis,cs,-1,(255,217,20),1);panel=Image.fromarray(vis[y:y2,x:x2]);panel.thumbnail((420,266));sheet.paste(panel,(i*420,34));dr.text((i*420+4,6),title,font=font,fill='white')
 dr.text((6,308),f"{tag}: {c['video']}, кадр {rr['frame_index']+1}. Зелёный — прогноз, жёлтый контур — разметка.",font=font,fill='white');sheet.save(OUT/f'comparison_{tag}.jpg',quality=91)
print('OFFICIAL_MEANS',avg,'OLD',hard,'GATES',oldpass,win,quality,flush=True)
