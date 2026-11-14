from official import *
from collections import defaultdict,Counter
p=read(OUT/'protocol.json');z=zipfile.ZipFile(ROOT/p['labelzip']);imgs=zipfile.ZipFile(ROOT/p['imagezip']);cases=[];scan=[];all_events=[]
def ok(o):return o['category']=='car' and not any((o.get('attributes') or {}).get(k,False) for k in ['ignore','ignored','crowd','iscrowd']) and M.area(o['rle'])>0
for v in p['videos']:
 db=labels(z,v);by=defaultdict(list);cache={}
 for idx,r in db.items():
  cache[idx]={o['id']:o for o in r['labels'] if ok(o)}
  for o in cache[idx].values():by[o['id']].append(idx)
 events=[]
 for tid,indices in by.items():
  for a,b in zip(sorted(indices),sorted(indices)[1:]):
   if b<=a+1 or b>a+8 or M.area(cache[a][tid]['rle'])<512:continue
   if not all(k in cache for k in range(a,b+1)):continue
   other=[o for n,o in cache[a].items() if n!=tid];other=sorted(other,key=lambda o:(-iou(bbox(cache[a][tid]),bbox(o)),-int(M.area(o['rle'])),o['id']))
   if not other:continue
   events.append(dict(video=v,prior=a,return_index=b,target=tid,rival=other[0]['id'],reference_area=int(M.area(cache[a][tid]['rle'])),past_overlap=iou(bbox(cache[a][tid]),bbox(other[0])),intervening=b-a-1))
 all_events.extend(events);chosen=sorted(events,key=lambda e:(e['return_index'],e['prior'],e['target'],e['rival']))[:1];scan.append(dict(video=v,episodes=len(events),chosen=bool(chosen)))
 for e in chosen:
  a=e['prior'];b=e['return_index'];cid=v+'_'+str(a)+'_'+str(b);folder=OUT/'frames'/cid;folder.mkdir(parents=True,exist_ok=False);fs=[]
  for idx in range(a,b+1):
   r=db[idx];f=folder/f'{idx-a:05d}.jpg';f.write_bytes(imgs.read('bdd100k/images/seg_track_20/val/'+v+'/'+r['name']));fs.append(dict(index=idx,name=r['name'],image=rel(f),sha256=sha(f)))
  cases.append(dict(**e,id=cid,frames=fs,past_boxes=[bbox(cache[a][id0]).tolist() for id0 in [e['target'],e['rival']]]))
save('cases.json',dict(cases=cases,availability=scan,candidate_events=all_events));print('ANNOTATION_GAPS',len(cases),len(all_events),flush=True)
