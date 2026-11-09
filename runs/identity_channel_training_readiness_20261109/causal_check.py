"""Check warm-last really equals the latest native public observation."""
import audit as a
from audit import *
p=read(OUT/'protocol.json');verify(p);r=read(OUT/'results.json');files=[Path(__file__),OUT/'protocol.json',OUT/'results.json'];write('causal_check_protocol.json',dict(passed=True,scope='Strengthen the same data gate: warm observation index3 must equal last native public observation before query. No new labels, maps or scores.',sources={f.relative_to(ROOT).as_posix():sha(f) for f in files}))
checks=[];eligible=[]
for video in sorted({e['video'] for e in r['edges'] if e['cohort']=='strict_last' and e['supported']}):
    with np.load(ROOT/p['videos'][video]['native'],allow_pickle=False) as z:outputs=z['outputs']
    for e in r['edges']:
        if e['video']!=video or e['cohort']!='strict_last' or not e['supported']:continue
        past=outputs[(outputs[:,5]==e['old'])&(outputs[:,0]<e['frame'])];assert len(past);last=int(past[:,0].max());reference=e['sequence'][e['reference_relative']][0];matches=last==reference;checks.append(dict(video=video,uid=e['uid'],raw_id=e['raw_id'],last_public_frame=last,reference_frame=reference,matches=matches,label=e['label']))
        if matches:eligible.append(e)
classes={video:{e['label'] for e in eligible if e['video']!=video} for video in sorted({e['video'] for e in eligible})};ready=bool(classes) and all(s=={'own','foreign'} for s in classes.values());verify(p)
write('causal_reference_audit.json',dict(passed=True,warm_last_edges=len(checks),actual_last_edges=len(eligible),own=sum(e['label']=='own' for e in eligible),foreign=sum(e['label']=='foreign' for e in eligible),videos=len(classes),mismatches=[c for c in checks if not c['matches']],checks=checks,actual_last_LOVO_ready=ready,zero_new_GT_labels=True,zero_signal_computations=True));print('Actual-last references',len(eligible),len(checks),'ready',ready,flush=True)
