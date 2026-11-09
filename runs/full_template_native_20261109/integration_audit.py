"""Exact algebra of the frozen joint head; no new fit, map, threshold or rollout."""
import sys
from common import *
start('cpu',timeout=600)
import gzip,collections
import numpy as np
mode=sys.argv[1]
if mode=='register':
    assert not (OUT/'integration_audit_protocol.json').exists()
    files=[Path(__file__),OUT/'deployment.json',OUT/'id_fate.json',OUT/'comparison_report.json',
        OUT/'boundary_audit.json',ROOT/'tracking/channel_identity_head.py',ROOT/'experiments/useful_literature.json']
    files += list((OUT/'rollouts/C34').glob('*.events.json.gz'))
    write('integration_audit_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
        question='Does the frozen joint head reject directly own links because of its channel term or because it also changed geometry coefficients/intercept?',
        timing='Post-result explanatory audit after negative full20video pilot. No independent confirmation or new reader-selection gate.',
        math='zC=bC+wC_geo*g+wC_channels*c = zG7+((bC-bG)+(wC_geo-wG_geo)*g)+wC_channels*c. Normalize g by exactly the deployed training stats. Reproduce BOTH logged zC and zG7 at every scored record. Undefined evidence must remain the exactG7fallback.',
        fixed_counterfactual='For interpretation ONLY compute zG7+wC_channels*c at all frozen213loggedfeaturevectors. No new coefficient fit or decision threshold. Do not call these counterfactual signs actual IDs. If own flips persist, preserving geometry alone is insufficient; if they disappear, changed geometry/intercept are necessary for those particular flips, not proof the channel term generalizes.',
        hypotheses=[dict(rank=1,status='known model architecture; contribution unmeasured',cause='Joint optimization changed seven geometry coefficients and free intercept, possibly reversing a good geometry decision.'),
            dict(rank=2,status='strongest alternative',cause='The channel correspondence itself is adverse on new instances; freezing geometry will not remove the false refusal.'),
            dict(rank=3,status='separate measured mechanism',cause='Global competition and duplicate removal explain lost outputs after positive heads, but not the two negative own-head decisions.')],
        primary_basis='Frozen ridge logistic objective and optimizer in tracking/channel_identity_head.py; Lin,Weng,Keerthi2008 https://jmlr.org/papers/volume9/lin08b/lin08b.pdf already verified/saved. This source supports the optimization method, not identity or transfer. The algebraic attribution and fixed counterfactual are ours.',
        next_action='Use the decomposition to choose one concrete reader-integration correction or reject that correction. No new channel/threshold/regularization variants on91pairs; no inference or full metric repeats.',
        sources={x.relative_to(ROOT).as_posix():sha(x) for x in files}))
    print('Frozen-head integration algebra registered',flush=True)
elif mode=='run':
    p=read(OUT/'integration_audit_protocol.json');verify(p)
    model=read(OUT/'deployment.json');gc=np.array(model['models']['G7']['theta']);cc=np.array(model['models']['C34']['theta'])
    mean=np.array(model['geometry_normalization']['mean']);scale=np.array(model['geometry_normalization']['scale'])
    fates={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in read(OUT/'id_fate.json')['records'] if r['arm']=='C34' and r['scored']}
    records=[];max_error=0.;counts=collections.Counter()
    for path in sorted((OUT/'rollouts/C34').glob('*.events.json.gz')):
        video=path.name.split('.events.')[0]
        with gzip.open(path,'rt',encoding='utf8') as f:events=json.load(f)['events']
        for event in events:
            for row in event['proposals']:
                if row['reason'] not in ['head_accept','head_refuse']:continue
                fate=fates[(video,event['frame'],row['old_id'],row['raw_id'])]
                g=(np.array(row['geometry'])-mean)/scale;zg=float(g@gc[:-1]+gc[-1])
                max_error=max(max_error,abs(zg-row['geometry_logit']))
                rec=dict(video=video,frame=event['frame'],old_id=row['old_id'],raw_id=row['raw_id'],kind=fate['kind'],
                    geometry_logit=zg,actual_logit=row['logit'],head_accept=row['accepted'],
                    evidence_valid=row.get('evidence_valid',False),global_selected=fate['global_selected'],
                    old_ID_emitted=fate['old_id_emitted_on_candidate'])
                if row.get('geometry_fallback',False):
                    assert abs(zg-row['logit'])<=1e-12;rec['fallback']=True;counts['geometry_fallback']+=1
                else:
                    c=np.array(row['channels']);assert c.shape==(34,)
                    drift=float(g@(cc[:7]-gc[:7])+cc[-1]-gc[-1]);channel=float(c@cc[7:-1])
                    reconstructed=zg+drift+channel;counterfactual=zg+channel
                    max_error=max(max_error,abs(reconstructed-row['logit']))
                    rec.update(changed_geometry_and_intercept=drift,channel_term=channel,
                        reconstructed_logit=reconstructed,frozen_geometry_same_channel_weights_logit=counterfactual,
                        frozen_geometry_counterfactual_accept=counterfactual>0)
                    counts['joint_head_scored']+=1
                    if fate['kind']=='own' and zg>0 and row['logit']<=0:
                        counts['direct_own_reversed_from_G7']+=1
                        counts['direct_own_reversal_removed_without_geometry_drift']+=int(counterfactual>0)
                        counts['direct_own_reversal_persists_without_geometry_drift']+=int(counterfactual<=0)
                    if fate['kind']=='foreign':
                        counts['direct_foreign']+=1
                        counts['direct_foreign_counterfactual_accepted']+=int(counterfactual>0)
                records.append(rec)
    assert len(records)==213 and max_error<1e-12
    verify(p);write('integration_audit.json',dict(passed=True,records=records,counts=counts,max_reconstruction_error=max_error,
        protocol_sha256=sha(OUT/'integration_audit_protocol.json'),zero_fits_maps_or_rollouts=True,
        limits='Fixed-vector decomposition only. Keeping old geometry while borrowing joint channel weights is not a newly trained/deployed verifier. Neither signs nor additive terms prove causal future ID benefit or correct channel mechanism.'))
    print('INTEGRATION',json.dumps(cast(counts)),max_error,flush=True)
else:raise ValueError(mode)
