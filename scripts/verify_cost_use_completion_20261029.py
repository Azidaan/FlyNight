"""Verify saved cost-use evidence without repeating tracking, metrics, or tests."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    out=RUN/'cost_use_agent_review';controllers={}
    for n in ['cost_use_preparation','cost_use_dev','cost_use_dev_audit','cost_use_val_preparation_recovery1','cost_use_val','cost_use_val_audit','cost_use_val_record']:
        f=RUN/n/'controller_result.json';c=service.read(f)
        assert c['exit_code']==0 and c['all_owned_ended'],n
        assert not any(service.matching(k,v) for k,v in c['owned'].items()),n
        controllers[n]=dict(source=str(f.relative_to(ROOT)),sha256=service.sha(f),actual_all_owned_ended=True,owned_count=len(c['owned']))
    a=service.read(RUN/'cost_use_val_audit/report.json');r=service.read(RUN/'cost_use_val/report.json')
    p=service.read(RUN/'cost_use_val/protocol.json');rec=service.read(RUN/'cost_use_val_record/report.json')
    assert a['complete'] and a['passed'] and a['audit_complete'] and rec['complete'] and rec['checks_passed']
    for key in ['primary_comparisons','native_comparisons','extractor_descriptive','pooled','main_acceptance']:
        assert a[key]==r[key],key
    assert set(a['primary_comparisons'])=={'r50_old','r50_shift','detic_old','detic_shift'}
    assert set(a['native_comparisons'])=={'r50_centered','detic_centered','r50_shift'}
    assert a['main_acceptance']==all(x['ci_bonferroni_9875'][0]>0 for x in a['primary_comparisons'].values())
    assert len(a['checked'])==20 and len({x['video'] for x in a['checked']})==20
    assert {x['video'] for x in a['checked']}=={s['video'] for s in p['sources']}
    assert all(service.sha(ROOT/f)==h for f,h in p['sha256'].items())
    for x in a['checked']:
        assert x['passed'] and x['raw_json_cost_memory_exact']
        assert x['inputs']['protocol_sha256']==service.sha(RUN/'cost_use_val/protocol.json')
        assert all(service.sha(ROOT/f)==h for f,h in x['sha256'].items())
        old=RUN/'masa_query_fp32_recovery1/videos'/x['video']/'report.json'
        assert x['inputs']['original_video_report_sha256']==service.sha(old)
    old=service.read(RUN/'masa_final_checked_record_recovery1/scientific_summary.json')
    for n,k in {'native':'native','r50_last':'r50_1333','detic_last':'detic_1333'}.items():assert a['pooled'][n]==old['pooled'][k]
    assert a['encoder_inferences']==0 and a['feature_max_difference_reused']==old['feature_max_difference']
    h=service.read(ROOT/'experiments/project_history.json');card=next(c for c in h['experiments'] if c['id']=='query_cost_use_v1')
    assert card['state']==('completed_positive' if a['main_acceptance'] else 'inconclusive')
    assert card['acceptance'][0]['met'] is True and card['acceptance'][1]['met']==a['main_acceptance']
    expected={(g,k) for g in ['primary_comparisons','native_comparisons'] for k,v in a[g].items() if v['ci_global7'][0]>0}
    actual=set()
    for c in card['confirmed_results']:
        g,k=c['evidence']['locator'];assert (g,k) in expected
        assert c['evidence']['sha256']==service.sha(RUN/'cost_use_val_audit/report.json');actual.add((g,k))
    assert actual==expected and rec['confirmed_result_count']==len(expected)
    graph=service.read(RUN/'cost_use_val_record/graph_audit.json')
    assert graph['passed'] and graph['all_hypotheses_and_legacy_byte_identical']
    assert (graph['hypothesis_node_count'],graph['hypothesis_edge_count'],graph['root_rays'])==(24,23,4)
    assert graph['deleted_files']==graph['moved_files']==0
    native=a['pooled']['native']['HOTA']
    proof=dict(complete=True,passed=True,controllers=controllers,all20_keys_and_output_sha_verified=True,all_input_sha_verified=True,
        references_exact_reuse=True,primary_comparisons=a['primary_comparisons'],native_comparisons=a['native_comparisons'],
        extractor_descriptive=a['extractor_descriptive'],pooled=a['pooled'],main_acceptance=a['main_acceptance'],
        confirmed_positive_comparisons=len(expected),relative_hota_percent={n:100*(m['HOTA']/native-1) for n,m in a['pooled'].items()},
        graph_source='runs/continuation_20261029/cost_use_val_record/graph_audit.json',updated=service.now())
    service.write(out/'verified.json',proof)
    print(proof)

if __name__=='__main__':main()
