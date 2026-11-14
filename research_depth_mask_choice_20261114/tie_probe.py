from common import *
import numpy as np
from scipy.stats import rankdata
if sys.argv[1]=='prepare':
 save('tie_protocol.json',dict(registered=now(),parent_sha256=sha(OUT/'protocol.json'),reason='On nighttime1093 all four RGB+depth rank sums equal5. Ordinary SAM is picked by list order even though RGB favors optical flow. Check a deterministic tie policy and whether a true target candidate exists on old1171.',rules=['Keep exact two-feature summed ranks. Only among tied minima choose the candidate with the lower raw RGB distance; remaining ties keep original order. No epsilon or fitted parameter. Apply symmetrically to RGB+Depth and RGB+Y.','Evaluate all frames/controls, preserve original failed primary. Report oracle range at1171 and forced-choice edge outputs.'],inputs={rel(OUT/n):sha(OUT/n) for n in ['choices.json','result.json','audit.json','descriptors.npz','protocol.json','tie_probe.py']},scope='Post-result bug diagnosis and candidate repair on explored queries; no independent validation or production promotion.'))
 journal('После основного depth_mask_choice_20261114 обнаружена точная причина выбора на1093: все четыре суммы рангов цвета и глубины равны5, поэтому сработал произвольный порядок кандидатов. Зарегистрировано диагностическое разрешение только точных ничьих по исходному цветовому расстоянию для обеих пар признаков, без новых весов или порогов. Прежний отрицательный результат сохраняется. Источник: tie_protocol.json.')
 raise SystemExit
pp=read(OUT/'tie_protocol.json');choices=read(OUT/'choices.json')['records'];r=read(OUT/'result.json');names=read(OUT/'protocol.json')['candidates'];rows=[]
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
for x in choices:
 v={k:np.array([np.inf if z is None else z for z in a]) for k,a in x['cost'].items()}
 for other in ['Depth','Y']:
  summed=rankdata(v['RGB'])+rankdata(v[other]);summed[~np.isfinite(v['RGB'])]=np.inf;ties=np.flatnonzero(summed==summed.min());j=min(ties,key=lambda i:(v['RGB'][i],i));name=names[j];original=x['selected']['RGB+'+other];candidate=next(z for z in r['records'] if z['case']==x['case'] and z['frame']==x['frame'] and z['method']==name);originalscore=next(z for z in r['records'] if z['case']==x['case'] and z['frame']==x['frame'] and z['method']=='RGB+'+other);rows.append(dict(case=x['case'],frame=x['frame'],feature=other,tied=[names[i] for i in ties],rank_sums=[float(z) if np.isfinite(z) else None for z in summed],original=original,repaired=name,original_iou=originalscore['iou'],iou=candidate['iou'],fpr=candidate['fpr'],predicted_pixels=candidate['predicted_pixels']))
means={}
for other in ['Depth','Y']:
 means[other]={}
 for c in read(OUT/'protocol.json')['cases']:
  z=[v['iou'] for v in rows if v['case']==c['id'] and v['feature']==other and v['iou'] is not None];means[other][c['id']]=float(np.mean(z)) if z else None
old=[z for z in r['records'] if z['case']=='old2179' and z['frame']==1171 and z['method'] in names];save('tie_result.json',dict(records=rows,means_by_video=means,video_equal_means={k:float(np.mean([v for v in d.values() if v is not None])) for k,d in means.items()},old1171_candidate_iou={z['method']:z['iou'] for z in old},source_unchanged=all(sha(ROOT/n)==h for n,h in pp['inputs'].items()),scope='No candidate masks or depthmodel were changed; same-frame exploratory repair only.'))
print(json.dumps(dict(means=means,changed=[(z['case'],z['frame'],z['feature'],z['original'],z['repaired']) for z in rows if z['original']!=z['repaired']])),flush=True)
