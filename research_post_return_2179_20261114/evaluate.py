from official import *
from collections import Counter
p=read(OUT/'protocol.json');q=read(OUT/'actions.json');a=np.load(OUT/'trajectory_masks.npz');z=zipfile.ZipFile(ROOT/p['labelzip']);db=labels(z,p['video']);rows=q['records'];n=0
for r,tr in zip(rows,q['matrices']):
 fr=r['frame'];assert fr==tr['frame'] and r['method']==tr['method'];g=masks(z,p['video'],db[r['reference_index']],p['official_ids'][0]);t,k,o=g
 for suffix,col in [('chosen','mask_iou'),('prior','prior_iou')]:
  m=a[f"{fr}__{r['method']}__{suffix}"];counts=np.bincount(t[k].astype(int)*2+m[k],minlength=4);tn,fp,fn,tp=counts;io=tp/(tp+fp+fn) if tp+fn else None;assert (io is None and r[col] is None) or (io is not None and abs(r[col]-io)<1e-12);n+=1
 base=dict(tr['original']);w=dict(tr['final']);assert all(w[k]==v for k,v in base.items()) and len(w)==len(set(w.values()))
 if r['action'] is not None:assert w[r['action']]==2179
stats={}
for name in ['original','confirmed']:
 gs=[r for r in rows if r['frame']>1171 and r['method']==name];counts=Counter(r['status'] for r in gs);vis=sum(r['visible'] for r in gs);current=sum(bool(r['truth_adequate_observations']) for r in gs);reserved=sum(bool(r['truth_adequate_observations']) and all(rid in r['adequate_reserved'] for rid in r['truth_adequate_observations']) for r in gs);stats[name]=dict(frames=len(gs),visible=vis,correct=counts['correct'],wrong=counts['wrong'],wait=counts['wait'],unknown=counts['unknown_ignore'],available=current,all_adequate_occupied=reserved)
old={r['frame']:r for r in rows if r['method']=='original'};known=old[1165]['status']=='wait' and old[1171]['status']=='correct' and old[1171]['action']==111062
g=stats['confirmed'];extended=g['visible']>=2 and g['correct']==g['visible'] and g['wrong']==0 and all(r['status']=='wait' for r in rows if r['frame']>1171 and r['method']=='confirmed' and not r['visible']);inc=g['correct']>stats['original']['correct'] and g['wrong']<=stats['original']['wrong'];save('result.json',dict(scientific_passed=bool(known and extended and inc),old_reproduced=bool(known),continuity=bool(extended),update_increment=bool(inc),stats=stats,rows=rows,scope='Committed-frame shadow decisions from archived observations, not running modified feedback through a tracker. Only exact aligned annotated raw frames. Initial true ID bound retrospectively; no claim during intervening unannotated frames.'))
save('audit.json',dict(passed=True,independent_mask_counts=n,all_original_owners_preserved=True,result_sha256=sha(OUT/'result.json'),source_inputs_unchanged=q['source_unchanged']))
