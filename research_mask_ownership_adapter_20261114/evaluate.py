from official import *
from collections import Counter
from PIL import ImageDraw,ImageFont
import itertools
p=read(OUT/'protocol.json');d=read(OUT/'decision.json')['records'];db=zipfile.ZipFile(ROOT/p['labelzip']);cache={};cases={c['id']:c for c in read(ROOT/'research_depth_identity_competition_20261114/pairs.json')['cases']};obs=np.load(OUT/'observations.npz');prior=np.load(OUT/'prior.npz');old=ROOT/'research_official_candidates_2179_20261114';go=np.load(old/'official_masks.npz');rows=[];old_detail=[];independent=0
def independently_assign(matrix,ids,obs_ids,reserved):
 # Independent from optimization: strict mutual maxima cannot collide.
 M=np.asarray(matrix);out=[]
 for i,tid in enumerate(ids):
  for j,rid in enumerate(obs_ids):
   if M[i,j]>.5 and sum(z>=M[i,j] for z in M[i])==1 and M[i,j]==max(M[i]) and sum(z>=M[i,j] for z in M[:,j])==1 and M[i,j]==max(M[:,j]) and reserved.get(rid,tid)==tid:out.append([tid,rid])
 return sorted(out)
for q in d:
 cid,v,fr=q['case'],q['video'],q['frame'];ids=q['owners'];rids=[r['id'] for r in q['items']];reserved=dict(q['reserved']);ms={i:obs[f'{cid}__{fr}__{i}'] for i in rids}
 if cid=='old2179':
  truths=[go[f'{fr}__{role}'] for role in ['far','near']];valid=go[f'{fr}__valid']
 else:
  if v not in cache:cache[v]=labels(db,v)
  c=cases[cid];r=cache[v][c['index']+fr];g=[masks(db,v,r,t) for t in c['ids']];truths=[a[0] for a in g];valid=g[0][1]
 for i,own in enumerate(ids):
  pre=prior[f'{cid}__{fr}__{i}'];rp=pre.ravel();mat=[]
  for rid,m in ms.items():
   mp=m.ravel();inter=np.count_nonzero(rp&mp);u=np.count_nonzero(rp)+np.count_nonzero(mp)-inter;mat.append(inter/u if u else 0.)
  assert np.allclose(mat,q['actions']['mask']['scores'][i],rtol=0,atol=1e-12)
 for name,a in q['actions'].items():
  assert sorted(a['pairs'])==independently_assign(a['scores'],ids,rids,reserved);independent+=1
  assert len(set(y for x,y in a['pairs']))==len(a['pairs'])
  pairs=dict(a['pairs']);original=dict(q['protected']);composed=original.copy()
  for own,rid in a['pairs']:
   assert rid not in original or original[rid]==own
   if own not in original.values():composed[rid]=own
  assert set(original.items())<=set(composed.items()) and len(composed)==len(set(composed.values())),(fr,name,composed)
  for i,own in enumerate(ids):
   t=truths[i];o=truths[1-i];rid=pairs.get(own);m=ms[rid] if rid is not None else np.zeros(valid.shape,bool);me=measure(m,t,valid,o);ne=measure(m,o,valid,t);status='deferred' if rid is None else 'correct' if me['iou'] is not None and me['iou']>.5 else 'wrong';available=[s for s,x in ms.items() if (measure(x,t,valid,o)['iou'] or 0)>.5]
   row=dict(case=cid,video=v,frame=fr,slot=i,owner=own,method=name,action=rid,visible=bool(t.any()),correct_observations=available,status=status,target_iou=me['iou'],other_iou=ne['iou'],past_reference=(pre_flag:=q['past_boxes'][i] is not None),best_mask_score=max(a['scores'][i],default=0),physical_neighbour_selected=(ne['iou'] or 0)>.5)
   if cid=='old2179':old_detail.append(dict(row,original_observations=len(original),composed_observations=len(composed),composed_owner=[k for k,x in composed.items() if x==own],same_original_pairs=True))
   else:rows.append(row)
videos=sorted({x['video'] for x in rows});methods=['past_box','predicted_box','mask'];stats={}
for n in methods:
 rr=[x for x in rows if x['method']==n];stats[n]=dict(actions=len(rr),correct=sum(x['status']=='correct' for x in rr),wrong=sum(x['status']=='wrong' for x in rr),deferred=sum(x['status']=='deferred' for x in rr),absent=sum(not x['visible'] for x in rr),wrong_absent=sum(not x['visible'] and x['status']=='wrong' for x in rr),neighbour_swaps=sum(x['physical_neighbour_selected'] for x in rr),by_video={v:sum(x['status']=='correct' for x in rr if x['video']==v) for v in videos})
out={}
for n in methods[:2]:
 dif=np.array([np.mean([x['status']=='correct' for x in rows if x['method']=='mask' and x['video']==v])-np.mean([x['status']=='correct' for x in rows if x['method']==n and x['video']==v]) for v in videos]);rng=np.random.default_rng(42);lo,hi=np.quantile(dif[rng.integers(len(dif),size=(10000,len(dif)))].mean(1),[.025,.975]);out[n]=dict(mean=float(dif.mean()),interval=[float(lo),float(hi)],winning_videos=int((dif>0).sum()),total=len(videos))
od=[q for q in old_detail if q['method']=='mask' and q['owner']==2179];oldpass=all((q['action'] is None and q['composed_owner']==[] if q['frame']==1165 else q['action']==111062 and q['composed_owner']==[111062] and q['status']=='correct') for q in od) and len(od)==2
transfer=all(stats['mask']['wrong']<=stats[n]['wrong'] and out[n]['winning_videos']>len(videos)/2 and out[n]['interval'][0]>0 for n in methods[:2]);reasons=Counter('missing_initial' if not x['past_reference'] else 'absent' if not x['visible'] else 'no_correct_detection' if not x['correct_observations'] else 'missed_or_wrong_despite_available' for x in rows if x['method']=='mask' and x['status']!='correct')
save('result.json',dict(scientific_passed=bool(oldpass and transfer and stats['mask']['wrong_absent']==0),old_passed=oldpass,transfer_passed=transfer,stats=stats,comparisons=out,old=old_detail,transfer_rows=rows,failure_partition=reasons))
save('audit.json',dict(passed=True,independent_mutual_optima=independent,independent_overlap_matrices=len(d),original_ownership_preserved=True,decisions_sha256=sha(OUT/'decision.json'),result_sha256=sha(OUT/'result.json')))
sheet=Image.new('RGB',(940,360),'#242a30');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
for k,q in enumerate(od):
 fr=q['frame'];im=cv2.cvtColor(cv2.imread(str(ROOT/f'research_mask_propagation_20261114/frame_{fr}.png')),cv2.COLOR_BGR2RGB);t=go[f'{fr}__far'];r=go[f'{fr}__near'];m=obs[f'old2179__{fr}__{q["action"]}'] if q['action'] is not None else np.zeros(t.shape,bool);im[m]=(im[m]*.45+np.array([32,225,87])*.55).astype('uint8')
 for mask,col in [(r,(40,168,255)),(t,(255,219,38))]:
  co,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,co,-1,col,2)
 sheet.paste(Image.fromarray(cv2.resize(im[185:445,855:1280],(470,287))),(470*k,31));draw.text((470*k+6,6),str(fr)+(': подождать' if q['action'] is None else ': вернуть номер 2179'),font=font,fill='white')
draw.text((8,328),'Жёлтый — цель, голубой — сосед, зелёный — маска назначенной рамки.',font=font,fill='white');sheet.save(OUT/'recovery_2179.jpg',quality=92)
