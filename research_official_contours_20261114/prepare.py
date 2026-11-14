from official import *
p=read(OUT/'protocol.json');z=zipfile.ZipFile(ROOT/p['labelzip']);imgs=zipfile.ZipFile(ROOT/p['imagezip']);cases=[];skip=[];oracle={};old=p['old'];v=old['video'];rows=labels(z,v);align=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')['matches'];clock={e['video_frame_index']:e for e in align};source=np.load(ROOT/'research_early_target_parts_20261114/prior_parts.npz')['visible'];ref=rows[clock[1141]['reference_index']];assert ref['name']==clock[1141]['reference_name']
bind=sorted([dict(identity=o['id'],covered=int((rle(o)&source).sum()),area=int(rle(o).sum()),box=bbox(o).tolist()) for o in ref['labels'] if eligible(o)],key=lambda b:(-b['covered'],b['identity']));assert len(bind)>=2 and bind[0]['covered']>=.8*source.sum() and bind[1]['covered']<.1*source.sum(),'Ambiguous ID; stop before query evaluation'
target=bind[0]['identity'];oldrows=[];frozen=np.load(ROOT/'research_auto_prior_masks_20261114/predictions.npz')
for fr in old['queries']:
 if fr not in clock:continue
 a=clock[fr];r=rows[a['reference_index']];assert r['name']==a['reference_name'];t,k,other=masks(z,v,r,target)
 for method in ('manual','yolo','box'):
  x=frozen[f'old2179__{method}__{fr}'];oldrows.append(dict(frame=fr,method=method,**measure(x,t,k,other)))
 if fr==1171:
  im=cv2.imread(str(ROOT/'research_mask_propagation_20261114/frame_1171.png'));panels=[]
  for method in ['truth','manual','yolo','box']:
   m=t if method=='truth' else frozen[f'old2179__{method}__{fr}'];c=im.copy();c[m]=(c[m]*.4+np.array([60,239,50])*.6).astype('uint8');co,_=cv2.findContours(t.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(c,co,-1,(0,200,255),1);panels.append(cv2.resize(c[200:450,900:1280],(420,276)))
  from PIL import ImageDraw,ImageFont
  out=Image.new('RGB',(1680,320),'#22252a');draw=ImageDraw.Draw(out);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
  for i,(img,name) in enumerate(zip(panels,['Полная разметка BDD100K','Ручные прошлые участки','Прошлая маска YOLO','Прошлая рамка SAM 2'])):out.paste(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),(420*i,36));draw.text((420*i+4,7),name,font=font,fill='white')
  out.save(OUT/'official_1171.jpg',quality=92)
save('old_result.json',dict(target=target,past_covered=bind[:4],past_known_pixels=int(source.sum()),exact_frames=sorted({r['frame'] for r in oldrows}),excluded_unaligned=[f for f in old['queries'] if f not in clock],records=oldrows,source='MOTS stable instance ID bound at past frame1141 only; box-track ID is a separate namespace. Skipped all query indices without exact saved image alignment.'))
for v in p['holdout']:
 lbl=labels(z,v)
 for idx in [49,149]:
  r=lbl.get(idx);cars=[o for o in (r or {}).get('labels',[]) if eligible(o)];options=[]
  for o in cars:
   area=int(M.area(o['rle']));comp=max([iou(bbox(o),bbox(other)) for other in cars if other['id']!=o['id']]+[0.])
   if area>=512:options.append((comp,area,o['id'],o))
  if not options or not all(idx+j in lbl for j in range(5)):skip.append(dict(video=v,index=idx,reason='no eligible visible car or missing official frame'));continue
  collision,area,target,o=sorted(options,key=lambda a:(-a[0],-a[1],a[2]))[0];cid=v+'_'+str(idx);folder=OUT/'frames'/cid;folder.mkdir(parents=True,exist_ok=False);lst=[]
  for j in range(5):
   rj=lbl[idx+j];member='bdd100k/images/seg_track_20/val/'+v+'/'+rj['name'];dest=folder/f'{j:05d}.jpg';dest.write_bytes(imgs.read(member));lst.append(dict(index=idx+j,name=rj['name'],image=rel(dest),sha256=sha(dest)))
  t,k,car=masks(z,v,r,target);oracle[cid]=t;cases.append(dict(id=cid,video=v,index=idx,target=target,past_box=bbox(o).tolist(),past_visible_pixels=area,past_competitor_bbox_iou=collision,frames=lst,folder=rel(folder)))
assert len({c['video'] for c in cases})>=8
with (OUT/'initial_official.npz').open('xb') as f:np.savez_compressed(f,**oracle)
save('cases.json',dict(cases=cases,skips=skip,initial_sha256=sha(OUT/'initial_official.npz'),images_sha256_checked=True));print('OFFICIAL_PREPARED',len(cases),len(skip),len(oldrows),flush=True)
