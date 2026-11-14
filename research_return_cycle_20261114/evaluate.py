from official import *
from collections import Counter
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');rr=read(OUT/'decisions.json')['records'];obs=np.load(OUT/'outputs.npz');returns=np.load(OUT/'returns.npz');z=zipfile.ZipFile(ROOT/p['labelzip']);cache={};rows=[];bad=[];comparison={};casepack=read(OUT/'cases.json');cases={c['id']:c for c in casepack['cases']};checked=0;oldactions=[]
for r in rr:
 cid,v,j=r['case'],r['video'],r['frame'];rids=r['return_ids'];past=[obs[f'{cid}__past__{i}'] for i in [0,1]];data=[returns[f'{cid}__{j}__{rid}'] for rid in rids];A=np.asarray(r['return_scores'])
 for i,x in enumerate(past):
  for k,y in enumerate(data):
   n=np.count_nonzero(x&y);u=np.count_nonzero(x)+np.count_nonzero(y)-n;assert abs(A[i,k]-(n/u if u else 0.))<1e-12;checked+=1
 independently=[(i,rid) for i,rid in r['forward'] if (a:=A[i,rids.index(rid)])>.5 and np.all(a>np.delete(A[i],rids.index(rid))) and np.all(a>np.delete(A[:,rids.index(rid)],i))]
 assert set(independently)=={tuple(x) for x in r['confirmed']}
 if r['split']=='old':
  oldactions.append(dict(frame=j,forward=r['forward'],confirmed=r['confirmed'],reverse_overlap=r['return_scores']));continue
 if v not in cache:cache[v]=labels(z,v)
 frame=cache[v][r['index']];priorframe=cache[v][cases[cid]['index']];allobj=[o for o in frame['labels'] if not any((o['attributes'] or {}).get(k,False) for k in ['ignore','crowd'])];priorobj=[o for o in priorframe['labels'] if not any((o['attributes'] or {}).get(k,False) for k in ['ignore','crowd'])]
 for i,identity in enumerate(r['ids']):
  t,k,other=masks(z,v,frame,identity);role=next((o for o in frame['labels'] if o['id']==identity),None);status=[]
  for mode in ['forward','confirmed']:
   rid=dict(r[mode]).get(i);m=obs[f'{cid}__{j}__obs__{rid}'] if rid is not None else np.zeros(t.shape,bool);q=measure(m,t,k,other);s='deferred' if rid is None else 'correct' if (q['iou'] or 0)>.5 else 'wrong';row=dict(case=cid,video=v,split=r['split'],frame=j,slot=i,id=identity,mode=mode,raw_id=rid,status=s,identity_record_present=role is not None,visible=bool(t.any()),iou=q['iou'],tp=q['tp'],fp=q['fp'],fn=q['fn']);rows.append(row);status.append(s)
   if mode=='forward' and s=='wrong':
    def rank(mask,objects,valid):
     mm=M.encode(np.asfortranarray((mask&valid).astype('uint8')));rs=M.iou([mm],[o['rle'] for o in objects],[0]*len(objects))[0] if objects else [];return sorted([dict(id=o['id'],category=o['category'],overlap=float(n),in_past=any(x['id']==o['id'] for x in priorobj)) for o,n in zip(objects,rs)],key=lambda a:(-a['overlap'],a['id']))[:3]
    matches=rank(m,allobj,k);prior_id=rank(past[i],priorobj,np.ones(k.shape,bool));bad.append(dict(**row,dominant=matches,prior_match=prior_id,rejected_by_cycle=(i,rid) not in independently,assigned_neighbour=(matches and matches[0]['overlap']>.5 and matches[0]['id']==r['ids'][1-i]),gt_attributes=None if role is None else role['attributes']))
  assert status!=['deferred','correct'] and status!=['deferred','wrong']
for split in ['diagnostic','fresh']:
 row=[q for q in rows if q['split']==split];vs=sorted({q['video'] for q in row});stats={}
 for mode in ['forward','confirmed']:
  rs=[q for q in row if q['mode']==mode];stats[mode]=dict(correct=sum(q['status']=='correct' for q in rs),wrong=sum(q['status']=='wrong' for q in rs),deferred=sum(q['status']=='deferred' for q in rs),wrong_when_absent=sum(q['status']=='wrong' and not q['visible'] for q in rs),total=len(rs),videos=len(vs))
 pair=lambda m,v:sum(x['status']=='wrong' and x['mode']==m and x['video']==v for x in row)/sum(x['mode']==m and x['video']==v for x in row)
 delta=np.array([pair('forward',v)-pair('confirmed',v) for v in vs]);lo,hi=np.quantile(delta[np.random.default_rng(42).integers(len(vs),size=(10000,len(vs)))].mean(1),[.025,.975]);comparison[split]=dict(stats=stats,paired_error_reduction=float(delta.mean()),interval=[float(lo),float(hi)])
fr=comparison['fresh'];s=fr['stats'];oldpass=next(x for x in oldactions if x['frame']==1171)['confirmed']==[[0,111062]] and next(x for x in oldactions if x['frame']==1165)['confirmed']==[];safe=s['confirmed']['correct']==s['forward']['correct'];gain=fr['interval'][0]>0 and s['confirmed']['wrong']<s['forward']['wrong'];save('result.json',dict(scientific_passed=bool(oldpass and safe and gain),old_passed=oldpass,correct_preserved=safe,fresh_improvement=gain,comparison=comparison,old=oldactions,records=rows,error_attribution=bad,causal_scope='Changing only reverse consistency removed or retained each forward choice. No physical identity inferred solely from absence of annotation.'))
save('audit.json',dict(passed=True,reverse_overlaps_recomputed=checked,duplicate_or_new_actions=0,result_hash=sha(OUT/'result.json')))
challenge=[b for b in bad if b['split']=='diagnostic' and b['assigned_neighbour']]
if challenge:
 q=challenge[0];cid=q['case'];j=q['frame'];r=next(x for x in rr if x['case']==cid and x['frame']==j);c=cases[cid];im=cv2.cvtColor(cv2.imread(str(ROOT/c['frames'][j]['image'])),cv2.COLOR_BGR2RGB);t,k,_=masks(z,c['video'],cache[c['video']][r['index']],q['id']);m=obs[f"{cid}__{j}__obs__{q['raw_id']}"];neighbor,_,_=masks(z,c['video'],cache[c['video']][r['index']],r['ids'][1-q['slot']]);ys,xs=np.where(t|neighbor|m);xx=max(0,int(xs.min())-35);yy=max(0,int(ys.min())-35);x2=min(1280,int(xs.max())+36);y2=min(720,int(ys.max())+36);sheet=Image.new('RGB',(1000,415),'#232930');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 for h,(ma,title) in enumerate([(t,'Цель по разметке'),(m,'Кого выбрало прямое сравнение')]):
  img=im.copy();img[ma]=(img[ma]*.45+np.array([31,230,67])*.55).astype('uint8');cs,_=cv2.findContours(t.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,cs,-1,(255,221,41),2);patch=Image.fromarray(img[yy:y2,xx:x2]);patch.thumbnail((499,325));sheet.paste(patch,(500*h+(500-patch.width)//2,40+(325-patch.height)//2));dr.text((500*h+7,10),title,font=font,fill='white')
 dr.text((8,380),'Обратная проверка: '+('отклонила неверный выбор' if q['rejected_by_cycle'] else 'тоже подтвердила соседа'),font=font,fill='white');sheet.save(OUT/'wrong_neighbour.jpg',quality=92)
