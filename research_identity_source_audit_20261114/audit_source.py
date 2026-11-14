from official import *
import gzip,lap
from scipy.optimize import linear_sum_assignment
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');v=p['video'];z=zipfile.ZipFile(ROOT/p['labelzip']);imgzip=zipfile.ZipFile(ROOT/p['imagezip']);seg=labels(z,v);base=ROOT/'runs/z13_val20_matched_20261024/cache'/v;native=ROOT/'runs/domain_matched_native_20261109';raw=np.load(base/'detections/raw_video.npz');GT=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');clock=read(base/'alignment.json')['matches'];outputs=np.load(native/'rollouts/C34'/(v+'.npz'))['outputs'];events=json.load(gzip.open(native/'rollouts/C34'/(v+'.events.json.gz')));limit=read(native/'protocol.json')['native_config']['match_thresh']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def solve(mat):
 nr,nc=mat.shape;_,xs,_=lap.lapjv(mat,extend_cost=True,cost_limit=limit);pairs=sorted((ix,int(j)) for ix,j in enumerate(xs) if j>=0);cost=np.zeros((nr+nc,nr+nc));cost[:nr,:nc]=mat;cost[:nr,nc:]=limit/2;cost[nr:,:nc]=limit/2;ii,jj=linear_sum_assignment(cost);obj=sum(float(mat[i,j]) for i,j in pairs)+(nr+nc-2*len(pairs))*limit/2;assert abs(float(cost[ii,jj].sum())-obj)<1e-8;return pairs
timeline=[];snapshots=[];sources=[];frmap={r['video_frame_index']:r for r in clock};elements=events['assignments'];rangeidx=range(480,563)
for fr in rangeidx:
 record={}
 for tid in [832,773]:
  raw_ass=[q for q in elements if q['frame']==fr and q['tid']==tid];assert len(raw_ass)<=1
  emit=outputs[(outputs[:,0]==fr)&(outputs[:,5]==tid)];assert len(emit)<=1 and bool(len(emit))==bool(raw_ass and raw_ass[0]['emitted'])
  if raw_ass:rid=raw_ass[0]['raw_id'];assert int(raw['frame_idx'][rid])==fr;record[str(tid)]=dict(raw_id=rid,public=bool(len(emit)),detector_box=raw['xyxy'][rid].tolist(),detector_class=int(raw['cls'][rid]),conf=float(raw['conf'][rid]))
 timeline.append(dict(frame=fr,owners=record))
for e in events['events']:
 fr=e['frame']
 if not 528<=fr<=546:continue
 tids=[r['tid'] for r in e['rows']];cols=e['high_raw_ids'];M0=np.array(e['C'],'float32');M1=np.array(e['corrected'],'float32');pp=solve(M1);assert pp==sorted(map(tuple,e['actual_first']));pp0=solve(M0);assert pp0==sorted(map(tuple,e['native_first']))
 involved=[];sel={tids[ii]:cols[jj] for ii,jj in pp}
 for tid in [832,773]:
  if tid not in tids:continue
  i=tids.index(tid);chosen=sel.get(tid);assigned=next((q for q in elements if q['frame']==fr and q['tid']==tid),None);opts=[]
  for j,col in enumerate(cols):
   if col in [sel.get(832),sel.get(773),31468]:opts.append(dict(raw_id=col,original=float(M0[i,j]),corrected=float(M1[i,j]),eligible=bool(M1[i,j]<limit),local_alteration=float(M1[i,j]-M0[i,j])))
  involved.append(dict(tid=tid,state=e['rows'][i]['state'],first=chosen,assigned_any_stage=assigned['raw_id'] if assigned else None,observed_options=opts))
 special=None
 if fr==540:
  keep=[j for j,t in enumerate(tids) if t!=773];removed=solve(M1[keep]);special=dict(remove773_only={str(tids[keep[i]]):cols[j] for i,j in removed}.get('832'),original_target_cost=float(M0[tids.index(832),cols.index(31468)]),rival_cost=float(M1[tids.index(773),cols.index(31468)]))
 snapshots.append(dict(frame=fr,shape=list(M1.shape),n_changed=int((M1!=M0).sum()),rows=involved,counterfactual=special))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);sheet=Image.new('RGB',(1000,690),'#23262b');dr=ImageDraw.Draw(sheet)
for idx,fr in enumerate(p['indices']):
 a=frmap[fr];r=seg[a['reference_index']];rr=GT[a['reference_index']];assert r['name']==rr['name']==a['reference_name'];g=next(o for o in rr['labels'] if o['id']=='00118997');bb=bbox(g);past=next(x['owners'] for x in timeline if x['frame']==fr);cands=sorted([dict(id=o['id'],category=o['category'],attributes=o['attributes'],bbox_overlap=iou(bbox(o),bb),box=bbox(o).tolist(),mask_pixels=int(M.area(o['rle']))) for o in r['labels']],key=lambda x:(-x['bbox_overlap'],x['id']));cands=[x for x in cands if x['bbox_overlap']>0][:4]
 member='bdd100k/images/seg_track_20/val/'+v+'/'+r['name'];blob=imgzip.read(member);assert blob==(ROOT/'data/bdd100k/images/track/val'/v/r['name']).read_bytes()
 im=cv2.cvtColor(cv2.imdecode(np.frombuffer(blob,'uint8'),1),cv2.COLOR_BGR2RGB);best=next(o for o in r['labels'] if o['id']==cands[0]['id']);ms=rle(best);im[ms]=(im[ms]*.45+np.array([33,235,60])*.55).astype('uint8')
 for tid,x in past.items():
  b=np.rint(x['detector_box']).astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(255,216,20) if tid=='832' else (255,86,255),1)
 px=(idx%2)*500;py=(idx//2)*324;sheet.paste(Image.fromarray(cv2.resize(im[334:452,488:708],(500,268))),(px,py+46));dr.text((px+3,py+3),f"{fr} · MOT: машина 00118997",font=font,fill='white');dr.text((px+3,py+24),f"MOTS: {best['category']} {best['id']}",font=font,fill='white');sources.append(dict(frame=fr,annotation=rr['name'],mot=dict(id=g['id'],category=g['category'],box=bb.tolist()),mots=cands,owners=past,jpeg_byte_identical=True))
dr.text((6,650),'Зелёный — официальный контур MOTS. Рамки назначений: 832 — жёлтая, 773 — розовая.',font=font,fill='white');sheet.save(OUT/'owner_sequence.jpg',quality=92)
available={s.split('/')[-2] for s in z.namelist() if '/bitmasks/val/' in s and s.endswith('.png')};hold=read(ROOT/'research_identity_holdout_20261112/recovery/summary_A.json')['new_proposals'];coverage=[dict(event=r['event_key'],video=r['video'],frame=r['frame'],has_video=r['video'] in available) for r in hold]
save('result.json',dict(timeline=timeline,snapshots=snapshots,annotations=sources,prior_unknown_coverage=coverage,solver_limit=limit,missing_snapshots=[f for f in range(534,541) if f not in {s['frame'] for s in snapshots}],matrices_replayed=len(snapshots),scientific_passed=False,scope='Two independent annotation products disagree. Establishes actual assignment path, not a unique physical identity through occlusion.'))
print('OWNERS_REPLAY',len(snapshots),[(r['frame'],r['mots'][0]['id']) for r in sources],coverage,flush=True)
