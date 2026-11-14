from official import *
p=read(OUT/'protocol.json');lab=zipfile.ZipFile(ROOT/p['labelzip']);imgs=zipfile.ZipFile(ROOT/p['imagezip']);cases=[dict(c,split='familiar') for c in p['old_cases']];skips=[]
for v in p['new_videos']:
 d=labels(lab,v)
 for idx in (49,149):
  r=d[idx];cars=[o for o in r['labels'] if eligible(o) and M.area(o['rle'])>0];opts=[(max([iou(bbox(o),bbox(a)) for a in cars if a['id']!=o['id']]+[0.]),int(M.area(o['rle'])),o['id'],o) for o in cars if M.area(o['rle'])>=512]
  if len(cars)<2 or not opts or any(idx+j not in d for j in range(5)):skips.append([v,idx,'no eligible pair or incomplete clip']);continue
  _,_,tid,target=sorted(opts,key=lambda a:(-a[0],-a[1],a[2]))[0];rival=sorted([a for a in cars if a['id']!=tid],key=lambda a:(-iou(bbox(target),bbox(a)),-int(M.area(a['rle'])),a['id']))[0];cid=v+'_'+str(idx);folder=OUT/'frames'/cid;folder.mkdir(parents=True,exist_ok=False);frames=[]
  for j in range(5):
   rr=d[idx+j];dest=folder/f'{j:05d}.jpg';dest.write_bytes(imgs.read('bdd100k/images/seg_track_20/val/'+v+'/'+rr['name']));frames.append(dict(index=idx+j,image=rel(dest),sha256=sha(dest),name=rr['name']))
  cases.append(dict(id=cid,video=v,index=idx,ids=[tid,rival['id']],frames=frames,split='fresh'))
save('cases.json',dict(cases=cases,skips=skips));print('PREPARED',len(cases))
