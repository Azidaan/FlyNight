"""Read final primary artifacts and verify process identities after bookkeeping."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    out=RUN/'masa_agent_review';out.mkdir(exist_ok=True)
    names=['masa_query_fp32_recovery1','masa_audit_fp32_recovery1','masa_final_checked_record_recovery1','masa_action_exposure']
    controllers={}
    for name in names:
        p=RUN/name/'controller_result.json';c=service.read(p)
        assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        controllers[name]=dict(source=str(p.relative_to(ROOT)),sha256=service.sha(p),actual_all_owned_ended=True)
    a=service.read(RUN/'masa_audit_fp32_recovery1/report.json')
    s=service.read(RUN/'masa_final_checked_record_recovery1/scientific_summary.json')
    assert a['complete'] and a['passed'] and len(a['checked'])==20
    assert len(a['feature_checks'])==9 and s['paired']==a['paired']
    assert max(x['max_difference'] for x in a['feature_checks'])<1e-5
    assert not a['main_acceptance'] and not s['main_acceptance']
    p=service.read(RUN/'masa_query_fp32_recovery1/protocol.json')
    assert {x['video'] for x in p['sources']}=={x['video'] for x in a['checked']}
    for x in p['sources']:
        marker=service.read(RUN/'masa_query_fp32_recovery1/videos'/x['video']/'tracking_completed.json')
        assert marker['native_zero_exact']
        assert all(service.sha(ROOT/f)==h for f,h in marker['sha256'].items())
    t=service.read(ROOT/'experiments/tracking_hypotheses.json')
    t10=next(x for x in t['methods'] if x['id']=='T10')
    assert t10['status']=='completed' and service.sha(ROOT/t10['result']['source'])==t10['result']['sha256']
    graph=RUN/'t10_completed_graph_20261029_201933/audit.json';g=service.read(graph)
    assert g['passed'] and g['other_hypotheses_and_legacy_byte_identical']
    assert g['deleted_files']==g['moved_files']==0
    assert (g['hypothesis_node_count'],g['hypothesis_edge_count'],g['root_rays'])==(24,23,4)
    assert not g['external_hypothesis_edges']
    card=next(x for x in service.read(ROOT/'experiments/project_history.json')['experiments'] if x['id']=='reid_masa_detic_v1')
    assert card['state']=='inconclusive' and not card.get('confirmed_results')
    assert s['pooled']['native']==s['pooled']['zero']
    queue=service.read(RUN/'queue.launch.json');guard=service.read(RUN/'guard.launch.json')
    assert service.matching(queue['pid'],queue['created']) and service.matching(guard['pid'],guard['created'])
    report=dict(complete=True,passed=True,controllers=controllers,scientific_source=t10['result'],graph_source=str(graph.relative_to(ROOT)),
        main_acceptance=False,card_state=card['state'],confirmed_positive_comparisons=0,
        queue_identity_verified=queue,guard_identity_verified=guard,
        full_output_sha_verified=True,all_20_keys_verified=True,
        next_action='Finished MASA-Detic comparison must not be repeated or retuned on these historical val videos. Prepare one distinct memory-write intervention with a prospective protocol and a separate dev gate; require agreement for changing the scientific plan. No accepted B2 audit or runnable MGOR/ScaleTrack model exists.',updated=service.now())
    service.write(out/'verified.json',report)
    print('VERIFIED20/5;fullaudit;noacceptedpositivecomparison;T10completed;allstagePIDended;queue/guardalive')

if __name__=='__main__':main()
