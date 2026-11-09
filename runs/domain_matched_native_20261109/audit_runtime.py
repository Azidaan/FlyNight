"""Independently check the newly deployed decisions and actual cost matrices.

No fit, retina, core, tracker rollout or official metric is repeated. The source
is registered separately because the already-running native passport is frozen.
"""
from evaluation_common import *
start(1800)
import gzip,math,collections
import numpy as np
from ultralytics.trackers.utils import matching
p=read(OUT/'protocol.json');phase=read(OUT/'audit_protocol.json');verify(phase)
assert all(read(OUT/(a+'_complete.json'))['passed'] for a in p['arms'])
model=read(OUT/'deployment.json');assert sha(OUT/'deployment.json')==sha(ROOT/'research_domain_matched_reader_20261109/deployment.json')
assert model['independent_gradient_max']<1e-7 and read(ROOT/'research_domain_matched_reader_20261109/audit.json')['passed']
mean=np.array(model['geometry_normalization']['mean']);scale=np.array(model['geometry_normalization']['scale']);assert (scale>0).all()
def logit(arm,g,c=None):
    if arm=='G_DOMAIN':return 1.
    z=(g-mean)/scale
    if arm!='G7':z=np.r_[z,c]
    theta=np.asarray(model['models'][arm]['theta']);assert len(theta)==len(z)+1
    return float(sum(float(a)*float(b) for a,b in zip(z,theta[:-1]))+theta[-1])
def geometry(q,frame):
    a=q['reference_box'];b=q['current_box'];size=[a[2]-a[0],a[3]-a[1]]
    return np.array([q['original_cost'],frame-q['reference_frame'],math.log((b[2]-b[0])/size[0]),math.log((b[3]-b[1])/size[1]),
        ((b[0]+b[2])-(a[0]+a[2]))/(2*size[0]),((b[1]+b[3])-(a[1]+a[3]))/(2*size[1]),math.log1p(q['old_cells'])])
zero=read(OUT/'zero_complete.json');assert zero['passed'] and zero['new_full_wrapper_witnesses']==2 and zero['old_all20_both_zero_controls_reused']
for video in p['zero_witness_videos']:
    r=read(OUT/'rollouts/zero_empty'/(video+'.receipt.json'));assert r['passed'] and r['exact_full_raw_and_JSON'] and not r['core_imported']
    assert r['execution_sha256']==sha(OUT/'execution_protocol.json')
totals={};receipts=[];max_geometry=max_logit=max_channel_mean=0.;precision_cases=[]
for arm in p['arms']:
    counts=collections.Counter()
    for video in p['videos']:
        receipt=read(OUT/'rollouts'/arm/(video+'.receipt.json'));assert receipt['passed'] and receipt['execution_sha256']==sha(OUT/'execution_protocol.json')
        assert arm=='C34' or not receipt['core_imported']
        for name,digest in receipt['artifacts'].items():assert sha(ROOT/name)==digest,name
        for name,digest in receipt['feature_files'].items():assert sha(ROOT/name)==digest,name
        with gzip.open(OUT/'rollouts'/arm/(video+'.events.json.gz'),'rt',encoding='utf8') as f:data=json.load(f)
        assert len(data['events'])==receipt['events'];counts['events']+=len(data['events'])
        for event in data['events']:
            original=np.asarray(event['C'],dtype=np.float32);corrected=np.asarray(event['corrected'],dtype=np.float32)
            assert original.shape==corrected.shape and np.isfinite(original).all() and np.isfinite(corrected).all()
            expected=original.copy()
            for i,j in event['votes']:expected[i,j]-=p['alpha']
            assert np.array_equal(expected,corrected)
            before,ut,_=matching.linear_assignment(original.copy(),thresh=p['native_config']['match_thresh'])
            after,_,_=matching.linear_assignment(corrected.copy(),thresh=p['native_config']['match_thresh'])
            assert np.asarray(before).reshape(-1,2).tolist()==event['native_first']
            assert np.asarray(after).reshape(-1,2).tolist()==event['actual_first']
            permitted=[]
            for q in event['proposals']:
                counts['proposals']+=1
                if q['reason'] not in ['head_accept','head_refuse']:
                    assert not q['accepted'];continue
                counts['scored']+=1;row=event['rows'][q['row']]
                assert q['row'] in ut and row['state']==2 and row['cls']==2 and len(q['candidate_columns'])==2
                assert len({h['frame'] for h in row['history']})>=2 and event['frame']-q['reference_frame'] in (5,6) and q['old_cells']>=3
                g=geometry(q,event['frame']);error=float(abs(g-np.asarray(q['geometry'])).max());assert error<1e-12;max_geometry=max(max_geometry,error)
                oldscore=logit('G_DOMAIN' if arm=='G_DOMAIN' else 'G7',g);assert abs(oldscore-q['geometry_logit'])<1e-10
                if arm in ['G7','G_DOMAIN'] or not q.get('evidence_valid'):score=oldscore
                else:
                    score=logit(arm,g,np.asarray(q['channels']))
                    winners=[i for i,v in enumerate(q['scores']) if v is not None];assert winners
                    winner=max(winners,key=lambda i:q['scores'][i]);assert winner==q['winner']
                    error=abs(float(np.mean(q['channels']))-q['scores'][winner]);assert error<1e-12;max_channel_mean=max(max_channel_mean,error)
                error=abs(score-q['logit']);assert error<1e-10;max_logit=max(max_logit,error)
                assert q['accepted']==(score>0) and q['reason']==('head_accept' if score>0 else 'head_refuse')
                if score>0:
                    permitted.append([q['row'],q['column']]);counts['accepted']+=1
                    value=float(corrected[q['row'],q['column']])
                    if value>p['native_config']['match_thresh']:
                        precision_cases.append(dict(arm=arm,video=video,frame=event['frame'],old_id=q['old_id'],raw_id=q['raw_id'],original_cost=q['original_cost'],corrected_cost=value))
            assert permitted==event['votes'];counts['global_matrices']+=1
        receipts.append(dict(arm=arm,video=video,receipt_sha256=sha(OUT/'rollouts'/arm/(video+'.receipt.json'))))
        print('AUDITED_RUNTIME',arm,video,counts['scored'],flush=True)
    totals[arm]=dict(counts)
verify(phase)
write('runtime_audit.json',dict(passed=True,totals=totals,geometry_max_error=max_geometry,logit_max_error=max_logit,
    channel_mean_max_error=max_channel_mean,accepted_above_float32_threshold=precision_cases,receipts=receipts,
    zero_witnesses=2,old_zero_controls_reused=40,training_audit_reused=True,no_refit=True,no_core=True,no_tracker_rollout=True,
    audit_protocol_sha256=sha(OUT/'audit_protocol.json'),goal_achieved=False))
status('runtime_audit_complete');print('RUNTIME_AUDIT_PASSED',max_logit,'threshold_cases',len(precision_cases),flush=True)
