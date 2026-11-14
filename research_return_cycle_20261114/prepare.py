from official import *
pr=read(OUT/'protocol.json');old=read(ROOT/'research_depth_identity_competition_20261114/pairs.json');lab=zipfile.ZipFile(ROOT/pr['labelzip']);imgs=zipfile.ZipFile(ROOT/pr['imagezip']);cases=[];skips=[]
for c in old['cases']:cases.append(dict(c,split='diagnostic'))
for video in pr['fresh_videos']:
 ls=labels(lab,video)
 for idx in (49,149):
  r=ls[idx];cars=[x for x in r['labels'] if eligible(x) and M.area(x['rle'])>0];targets=[x for x in cars if M.area(x['rle'])>=512]
  if len(cars)<2 or not targets or any(idx+j not in ls for j in range(5)):skips.append([video,idx,'insufficient pair or clip']);continue
  target=sorted(targets,key=lambda o:(-max(iou(bbox(o),bbox(a)) for a in cars if a['id']!=o['id']),-int(M.area(o['rle'])),o['id']))[0];rival=sorted([x for x in cars if x['id']!=target['id']],key=lambda a:(-iou(bbox(target),bbox(a)),-int(M.area(a['rle'])),a['id']))[0]
  cid=video+'_'+str(idx);folder=OUT/'frames'/cid;folder.mkdir(parents=True,exist_ok=False);fs=[]
  for j in range(5):
   rj=ls[idx+j];dest=folder/f'{j:05d}.jpg';dest.write_bytes(imgs.read('bdd100k/images/seg_track_20/val/'+video+'/'+rj['name']));fs.append(dict(image=rel(dest),sha256=sha(dest),name=rj['name'],index=idx+j))
  cases.append(dict(id=cid,video=video,index=idx,split='fresh',ids=[target['id'],rival['id']],past_truth_boxes=[bbox(target).tolist(),bbox(rival).tolist()],frames=fs))
save('cases.json',dict(cases=cases,skips=skips))
