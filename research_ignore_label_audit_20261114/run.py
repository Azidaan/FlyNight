from official import *
from collections import Counter
p=read(OUT/'protocol.json');files=p['files'];z=zipfile.ZipFile(ROOT/p['labelzip']);cache={};records=[];impacts={};all_cases={};errors=[]
def flagged(o):return bool(o and any((o.get('attributes') or {}).get(k,False) for k in ('ignored','ignore','crowd','iscrowd')))
def label_at(split,v,idx):
 key=(split,v)
 if key not in cache:cache[key]={g['frameIndex']:g for g in json.loads(z.read(PREFIX+f'rles/{split}/'+v+'.json'))['frames']}
 return cache[key][idx]
def row(split,v,idx,identity):
 r=label_at(split,v,idx);o=next((x for x in r['labels'] if x['id']==identity),None);attr=(o.get('attributes') or {}) if o else {};return dict(video=v,index=idx,id=identity,record_present=o is not None,legacy_eligible=o is not None and not any(attr.get(k,False) for k in ['ignore','crowd']),excluded=flagged(o),attributes=attr)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for key in ['missing','cycle']:
 db=read(ROOT/files[key])['cases'];all_cases[key]={c['id']:c for c in db}
 for c in db:
  for i,id0 in enumerate(c['ids']):
   for j in [0,1,2,4]:
    meta=row('val',c['video'],c['index']+j,id0);records.append(dict(dataset=key,case=c['id'],split=c['split'],slot=i,offset=j,**meta))
for q in read(ROOT/files['reader'])['records']:records.append(dict(dataset='reader',case=q['id'],slot=None,offset=0,split=q['split'],**row(q['split'],q['video'],q['index'],q['official_id'])))
for c in read(ROOT/files['single'])['cases']:
 for j in [0,1,2,4]:records.append(dict(dataset='single',case=c['id'],slot=0,offset=j,split='secondary',**row('val',c['video'],c['index']+j,c['target'])))
for r in records:
 if not r['excluded']:continue
 split=r['split'] if r['dataset']=='reader' else 'val';f=label_at(split,r['video'],r['index']);o=next(x for x in f['labels'] if x['id']==r['id']);pix=np.asarray(Image.open(io.BytesIO(z.read(PREFIX+f"bitmasks/{split}/{r['video']}/"+f['name'].replace('.jpg','.png')))));target=rle(o);r.update(instance_area=int(target.sum()),invalid_pixels=int((target&((pix[:,:,1]&3)!=0)).sum()),valid_pixels=int((target&((pix[:,:,1]&3)==0)).sum()),flag_histogram=dict((str(i),int(n)) for i,n in zip(*np.unique(pix[:,:,1][target],return_counts=True))))
 rr=r['attributes'];assert flagged(o);expected=int(rr.get('ignored',False) or rr.get('ignore',False)) + (int(bool(rr.get('crowd',False) or rr.get('iscrowd',False)))<<1)
 if target.any() and (r['valid_pixels']!=0 or any((int(g)&3)!=expected for g in r['flag_histogram'])):errors.append(dict(case=r['case'],id=r['id'],expected=expected,hist=r['flag_histogram']))
mapping={(r['dataset'],r['case'],r['slot'],r['offset']):r for r in records};ds=read(ROOT/files['scores'])['records'];dr=read(ROOT/files['roi'])['records']
for sp in ['familiar','fresh']:
 ar=[q for q in ds if q['split']==sp];keep=[q for q in ar if not mapping['missing',q['case'],q['slot'],0]['excluded']];impacts['box_availability_'+sp]=dict(previous_denominator=len(ar),corrected_denominator=len(keep),excluded=[(q['case'],q['slot']) for q in ar if q not in keep],old_full=sum(q['full'] is not None for q in ar),valid_full=sum(q['full'] is not None for q in keep),old_tiling_gain=sum(q['new_from_tiles'] for q in ar),valid_tiling_gain=sum(q['new_from_tiles'] for q in keep),definition='Same original selected instances, no replacement and no re-training.')
 cr=read(ROOT/files['cycle_result'])['records']
for sp in ['diagnostic','fresh']:
 for name in ['forward','confirmed']:
  ar=[x for x in cr if x['split']==sp and x['mode']==name];keep=[x for x in ar if not mapping['cycle',x['case'],x['slot'],0]['excluded'] and not mapping['cycle',x['case'],x['slot'],x['frame']]['excluded']];impacts[sp+'_'+name]=dict(original=dict(Counter(x['status'] for x in ar)),corrected=dict(Counter(x['status'] for x in keep)),n_excluded=len(ar)-len(keep))
reader=[q for q in records if q['dataset']=='reader'];impacts['reader']=dict(training_excluded=[r['case'] for r in reader if r['split']=='train' and r['excluded']],val_excluded=[r['case'] for r in reader if r['split']=='val' and r['excluded']])
oldreader=read(ROOT/files['reader_result'])['records'];fval=[r['case'] for r in reader if r['split']=='val' and r['excluded']];impacts['reader']['affected_report_rows']=sum(r['request'] in fval for r in oldreader)
for c in read(ROOT/files['roi'])['records']:
 if c['arm']!='full':continue
 assert (c['official_pixels']==0)==(mapping['missing',c['case'],c['slot'],0]['excluded'] or not mapping['missing',c['case'],c['slot'],0]['record_present']),c
excluded=set((r['video'],r['index'],r['id']) for r in records if r['excluded']);misspelled=set((r['video'],r['index'],r['id']) for r in records if r['excluded'] and r['legacy_eligible']);save('result.json',dict(scientific_passed=bool(misspelled and not errors),records=records,impacts=impacts,unique_flagged=len(excluded),unique_ignored_missed_by_legacy=len(misspelled),flag_disagreements=errors,scope='This audit follows exactly the original selected identities and frames, without changing source datasets or silently selecting replacement pairs. No guarantee about untested parts of the project.'))
save('audit.json',dict(passed=True,selection_records=len(records),unique_frames_checked=len({(x['video'],x['index']) for x in records}),explicit_metadata_pixel_discrepancies=errors,result_sha256=sha(OUT/'result.json')))
