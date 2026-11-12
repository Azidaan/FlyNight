from common import *
import numpy as np
p=verify();flow=read(OUT/'correspondence.json');members=read(OUT/'box_membership.json');rows=[]
for case,f,m in zip(p['cases'],flow['records'],members['records']):
    assert case['case']['old_id']==f['old_id']==m['old_id']
    q=np.array(f['originals']['C34']);det=np.array(case['case']['current_box']);gt=np.array(m['current_candidate_GT_box']);inside=lambda bb:((q>=bb[:2])&(q<=bb[2:])).all(1)
    a,b=inside(det),inside(gt)
    rows.append(dict(old_id=f['old_id'],total=len(q),detector_box=det,annotated_candidate_box=gt,outside_detector=int((~a).sum()),outside_annotation=int((~b).sum()),outside_both=int((~a&~b).sum()),inside_detector_outside_annotation=int((a&~b).sum()),outside_detector_inside_annotation=int((~a&b).sum()),inside_both=int((a&b).sum())))
r=next(x for x in rows if x['old_id']==574);assert r['outside_both']==22 and r['inside_detector_outside_annotation']==28 and r['outside_annotation']==50
save('detector_box_audit.json',dict(passed=True,created=now(),records=rows,sources={n:sha(OUT/n) for n in ['protocol.json','correspondence.json','box_membership.json','detector_box_audit.py']},scope='Exact set membership. Distinguishes full-template extent from detector/annotation discrepancy; neither is declared a causal explanation of score without a counterfactual.'))
print(json.dumps(cast(r),ensure_ascii=False))
