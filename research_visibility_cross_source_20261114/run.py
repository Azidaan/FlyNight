from official import *
from PIL import ImageDraw,ImageFont
p=read(OUT/'protocol.json');parent=ROOT/'research_return_cycle_20261114';rr=read(parent/'result.json');cases={c['id']:c for c in read(parent/'cases.json')['cases']};masks_saved=np.load(parent/'outputs.npz');z=zipfile.ZipFile(ROOT/p['labelzip']);cache={};mot={};results=[];cols=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def eligible_track(o):return o['category'] in ['car','truck','bus'] and not any((o.get('attributes') or {}).get(k,False) for k in ['ignore','ignored','crowd','iscrowd'])
def enclosed(b,m):
 y,x=np.where(m);return float(np.mean((x>=b[0])&(x<=b[2])&(y>=b[1])&(y<=b[3]))) if len(y) else 0
def rank(b,objects):return sorted([dict(id=o['id'],iou=iou(b,bbox(o)),box=bbox(o).tolist(),category=o['category']) for o in objects],key=lambda a:(-a['iou'],a['id']))
def tight(m):
 yy,xx=np.where(m);return np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
def bind(b,objs):
 rr=rank(b,objs);good=[q for q in rr if q['iou']>=.5];return (good[0]['id'] if len(good)==1 else None),rr[:3],len(good)
for q in rr['error_attribution']:
 cid=q['case'];c=cases[cid];v=c['video'];j=q['frame'];idx=c['index'];i=q['slot'];name=q['id'];rid=q['raw_id'];ridx=idx+j;date=c['frames'][j]['name']
 if v not in cache:cache[v]=labels(z,v);mot[v]={r['frameIndex']:r for r in read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json')}
 seg=cache[v];mf=mot[v];past=seg[idx];query=seg[ridx];mpast=mf[idx];mq=mf[ridx];assert past['name']==mpast['name']==c['frames'][0]['name'] and query['name']==mq['name']==date
 for k in [0,j]:assert (ROOT/c['frames'][k]['image']).read_bytes()==(ROOT/'data/bdd100k/images/track/val'/v/c['frames'][k]['name']).read_bytes()
 source=next(o for o in past['labels'] if o['id']==name);m0=masks_saved[f'{cid}__past__{i}'];m=masks_saved[f'{cid}__{j}__obs__{rid}'];mo=[o for o in mq['labels'] if eligible_track(o)];po=[o for o in mpast['labels'] if eligible_track(o)];bound,rpast,nb=bind(bbox(source),po);pb,manualrank,na=bind(tight(m0),po) if m0.any() else (None,[],0)
 best=q['dominant'][0] if q['dominant'] else None;to=next((o for o in query['labels'] if best and o['id']==best['id']),None);mb,msrank,nm=bind(tight(m),mo);mseg,seg_rank,nseg=bind(bbox(to),mo) if to else (None,[],0);bothpast=bound is not None and bound==pb;bothquery=mb is not None and mb==mseg
 primary=next((o for o in mo if bound is not None and o['id']==bound),None);target=next((o for o in query['labels'] if o['id']==name),None)
 flags=dict(byte_identical=True,matching_binding_at_reference=bothpast,matching_binding_at_query=bothquery,segmentation_nominal_present=target is not None,box_nominal_present=primary is not None,seg_best_id=best['id'] if best else None,seg_best_iou=best['overlap'] if best else None,clear_different_seg=bool(best and best['id']!=name and best['overlap']>.5),clear_same_seg=bool(best and best['id']==name and best['overlap']>.5))
 if flags['clear_different_seg'] and bothpast and bothquery and bound!=mb:verdict='independent_products_support_different_owner'
 elif flags['clear_different_seg'] and bothpast and bothquery and bound==mb:verdict='products_disagree_on_continuity'
 elif target and best and best['id']==name and best['overlap']<=.5:verdict='closest_seg_target_fails_quality_not_proved_swap'
 elif flags['clear_different_seg']:verdict='different_seg_id_second_source_ambiguous'
 else:verdict='mixture_or_inadequate_coverage_no_owner_proof'
 results.append(dict(case=cid,video=v,frame=j,slot=i,split=q['split'],seg_target=name,mask_raw=rid,verdict=verdict,flags=flags,past_box_track_id=bound,past_automatic_id=pb,query_mask_track_id=mb,query_seg_track_id=mseg,independent_rankings=dict(past_official=rpast,past_auto=manualrank,query_auto=msrank,query_seg=seg_rank),unique_counts=[nb,na,nm,nseg],expected_box_containment=enclosed(bbox(primary),m) if primary else None))
 if (verdict not in [s[0] for s in cols]) and len(cols)<4:
  im=cv2.cvtColor(cv2.imread(str(ROOT/c['frames'][j]['image'])),cv2.COLOR_BGR2RGB);im[m]=(im[m]*.45+np.array([35,231,76])*.55).astype('uint8');b=tight(m)
  if primary:
   a=bbox(primary).astype(int);cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),(250,206,20),2);b=np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])]
  x,y,x2,y2=b.astype(int);patch=Image.fromarray(im[max(0,y-25):min(720,y2+26),max(0,x-25):min(1280,x2+26)]);patch.thumbnail((415,185));cols.append((verdict,patch,cid,j))
cats={k:sum(r['verdict']==k for r in results) for k in sorted({r['verdict'] for r in results})};assert len(results)==13
save('result.json',dict(scientific_passed=all(r['verdict']!='different_seg_id_second_source_ambiguous' and r['verdict']!='mixture_or_inadequate_coverage_no_owner_proof' for r in results),records=results,counts=cats,scope='Reference bindings require exactly one MOT vehicle at IoU>=.5 both for past official MOTS box and automatic mask bounds. Current independent bindings likewise require agreement between auto bounds and official dominant MOTS box. No human truth inferred from a missing ID. Same BDD data family, different annotation products not independent annotator sample.'))
save('audit.json',dict(passed=True,paired_image_byte_checks=2*len(results),all_forward_errors=len(results),case_keys_unique=len({(r['case'],r['frame'],r['slot']) for r in results})==len(results),result_sha256=sha(OUT/'result.json')))
labels_ru={'independent_products_support_different_owner':'Обе разметки: другая машина','products_disagree_on_continuity':'Разметки расходятся по номеру','closest_seg_target_fails_quality_not_proved_swap':'Цель ближе других, контур неточен','different_seg_id_second_source_ambiguous':'Другой контур, рамки неоднозначны','mixture_or_inadequate_coverage_no_owner_proof':'Смешанная или неполная маска'};sheet=Image.new('RGB',(850,480),'#22282c');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for pos,(k,patch,cid,j) in enumerate(cols):
 xx=425*(pos%2);yy=240*(pos//2);sheet.paste(patch,(xx+(425-patch.width)//2,yy+30+(185-patch.height)//2));dr.text((xx+4,yy+4),labels_ru[k],font=font,fill='white');dr.text((xx+4,yy+216),cid+' +'+str(j),font=font,fill='white')
sheet.save(OUT/'agreement_examples.jpg',quality=91)
