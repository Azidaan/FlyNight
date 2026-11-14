from official import *
from collections import defaultdict
p=read(OUT/'protocol.json');cases=read(OUT/'cases.json')['cases'];res=read(OUT/'result.json');z=zipfile.ZipFile(ROOT/p['labelzip']);images=zipfile.ZipFile(ROOT/p['imagezip']);pd=np.load(OUT/'predictions.npz');old=read(OUT/'old_result.json');cache={};table={};bindings={};nchecks=0
def bittruth(v,r,id):
 pix=raster(z,v,r);k=(pix[:,:,1]&3)==0;colors=pix[:,:,2].astype(int)*256+pix[:,:,3];o=next((o for o in r['labels'] if o['id']==id),None);mask=np.zeros(k.shape,bool)
 if o:
  original=rle(o);assert original.any();codes,counts=np.unique(colors[original],return_counts=True);code=int(codes[counts.argmax()]);assert code and counts.max()==original.sum(),(v,id)
  mask=(colors==code)&(pix[:,:,0]==3);assert np.array_equal(mask,original),(v,r['name'],id)
  if (v,id) in bindings:assert bindings[v,id]==code
  bindings[v,id]=code
 return mask&k,k
def check(m,t,k,r):
 q=np.bincount(2*t[k].astype(int)+m[k],minlength=4);tn,fp,fn,tp=map(int,q);assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn']);assert r['iou']==(tp/(tp+fp+fn) if tp+fn else None)
for c in cases:
 v=c['video'];cid=c['id']
 if v not in cache:cache[v]=labels(z,v)
 for j in [1,2,4]:
  rr=cache[v][c['index']+j];t,k=bittruth(v,rr,c['target'])
  for r in [r for r in res['records'] if r['case']==cid and r['offset']==j]:
   check(pd[f"{cid}__{r['source']}__{j}"],t,k,r);nchecks+=1
for name in p['arms']:
 vu={}
 for v in p['holdout']:
  vals=[r['iou'] for r in res['records'] if r['video']==v and r['source']==name and r['iou'] is not None]
  if vals:vu[v]=sum(vals)/len(vals)
 assert abs(sum(vu.values())/len(vu)-res['video_equal_means'][name])<1e-13
v=p['old']['video'];gt=labels(z,v);align=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')['matches'];truth=np.load(ROOT/'research_mask_propagation_20261114/evaluation_truth.npz');prior=np.load(ROOT/'research_auto_prior_masks_20261114/predictions.npz');discrep=[];font=ImageFont=None
from PIL import ImageFont,ImageDraw
imgpan=[];reviewframes=[]
for fr in [1141,1159,1165,1171]:
 a=next(x for x in align if x['video_frame_index']==fr);g=gt[a['reference_index']];n=g['name'];mot=(ROOT/'data/bdd100k/images/track/val'/v/n).read_bytes();mots=images.read('bdd100k/images/seg_track_20/val/'+v+'/'+n);assert mot==mots
 t,k=bittruth(v,g,old['target']);pix=raster(z,v,g);codes=pix[:,:,2].astype(int)*256+pix[:,:,3];neighbors={o['id']:rle(o) for o in g['labels'] if o['category']=='car'}
 if fr in [1165,1171]:
  manual=truth[str(fr)];mp=manual==1;mn=(manual>=0)&(manual!=1);counts={key:int((mp&m).sum()) for key,m in neighbors.items() if (mp&m).any()};m=prior['old2179__yolo__'+str(fr)];fp=m&mn;discrep.append(dict(frame=fr,source_name=n,manual_positive=int(mp.sum()),manual_positive_official_ids=counts,manual_negative_over_official_target=int((mn&t).sum()),manual_yolo_false_positive=int(fp.sum()),of_those_official_ignore=int((fp&~k).sum()),of_those_official_target=int((fp&t).sum()),of_those_official_other=int((fp&k&~t).sum())))
  for rr in [x for x in old['records'] if x['frame']==fr]:check(prior['old2179__'+rr['method']+'__'+str(fr)],t,k,rr)
 else:mp=(np.load(ROOT/'research_early_target_parts_20261114/prior_parts.npz')['visible'] if fr==1141 else np.load(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz')['1159_target']).astype(bool)
 im=cv2.imdecode(np.frombuffer(mots,'uint8'),1);panel=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);rival=neighbors.get('528449',np.zeros_like(t));panel[t]=(panel[t]*.44+np.array([20,230,50])*.56).astype('uint8');panel[rival]=(panel[rival]*.44+np.array([245,40,35])*.56).astype('uint8');cc,_=cv2.findContours(mp.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,cc,-1,(255,215,15),2);imgpan.append(cv2.resize(panel[190:440,900:1280],(490,322)));reviewframes.append(fr)
sheet=Image.new('RGB',(980,750),'#23262b');d=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for i,pan in enumerate(imgpan):x=(i%2)*490;y=(i//2)*351;sheet.paste(Image.fromarray(pan),(x,y+26));d.text((x+5,y+2),'Кадр '+str(reviewframes[i]),font=font,fill='white')
d.text((6,712),'Зелёный — официальная дальняя машина; красный — сосед; жёлтый — прежний ручной участок.',font=font,fill='white');sheet.save(OUT/'ownership_review.jpg',quality=92)
save('ownership_audit.json',dict(records=discrep,MOT_and_MOTS_jpegs_exact=reviewframes,stable_id=old['target'],alternative_id='528449',scope='Independent dataset annotation conflicts with prior manual labels on1165; labels at unaligned frames remain unadjudicated. Sparse negative vs official ignored regions is a scoring convention, not evidence that ignored objects are absent.'))
save('audit.json',dict(passed=True,heldout_tables=nchecks,old_tables=6,group_means=len(p['arms']),matched_mask_encodings='Each visible query object: RLE against exact RGBA identity raster; counts recomputed from RGBA using confusion matrix.',official_instance_bindings={v+'__'+i:c for (v,i),c in bindings.items()},result_sha256=sha(OUT/'result.json'),consistent=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
