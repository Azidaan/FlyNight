"""Повторно прочитать все выходы и проверить границы вмешательств и парные числа."""
from pathlib import Path
from datetime import datetime
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching
from scripts.hota_error_budget_20261027 import interventions

def audit():
    folder=RUN/'error_budget/recovery1';p=read(folder/'protocol.json');report=read(folder/'report.json');controller=read(folder/'controller_result.json')
    assert controller['exit_code']==0 and controller['all_owned_ended'] and not controller['timed_out']
    old_deadline=datetime.fromisoformat(read(folder.parent/'execution_deadline.json')['deadline'])
    new_deadline=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline'])
    assert new_deadline<=old_deadline==datetime.fromisoformat(p['original_deadline'])
    assert not any(matching(k,v) for k,v in controller['owned'].items())
    assert report['complete'] and report['diagnostic_only'] and not report['maximum_or_upper_bound'] and not report['real_model_improvement']
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    assert set(p['groups']['development20']).isdisjoint(p['groups']['transfer20'])
    # Проверяется биекция числовых ID и исходных смысловых идентичностей во всех кадрах/ветвях.
    forward={};reverse={};count=0;geometry_rows=0
    for source in p['sources']:
        video=source['video'];base=read(ROOT/source['baseline']);truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
        outputs={arm:read(folder/arm/(video+'.json')) for arm in p['arms']}
        assert all(len(out)==len(base) for out in outputs.values())
        assert report['per_video']['baseline'][video]==source['baseline_metrics']
        assert sha(ROOT/source['baseline'])==report['proof'][video]['baseline_sha256']
        for index,(row,gt) in enumerate(zip(base,truth)):
            expected=interventions(row['labels'],gt['labels'],p['matching_iou'])
            for arm,out in outputs.items():
                actual=out[index];assert {k:v for k,v in actual.items() if k!='labels'}=={k:v for k,v in row.items() if k!='labels'}
                labels=actual['labels'];assert len({x['id'] for x in labels})==len(labels)
                assert len(labels)==len(expected[arm])
                for got,want in zip(labels,expected[arm]):
                    if arm in ['identity','identity_geometry','identity_missing']:
                        number=int(got['id']);name=want['id']
                        assert name not in forward or forward[name]==number
                        assert number not in reverse or reverse[number]==name
                        forward[name]=number;reverse[number]=name
                        assert {k:v for k,v in got.items() if k!='id'}=={k:v for k,v in want.items() if k!='id'}
                    else:assert got==want
                    count+=1
                if arm=='geometry':geometry_rows+=1
    independent={};rng=np.random.default_rng(91027)
    means={group:float(np.mean([report['per_video']['baseline'][v]['HOTA'] for v in vids])) for group,vids in p['groups'].items()}
    for group,vids in p['groups'].items():
        independent[group]={}
        comparisons={arm:(arm,'baseline') for arm in p['arms']}
        comparisons.update(missing_given_identity=('identity_missing','identity'),geometry_given_identity=('identity_geometry','identity'))
        for name,(a,b) in comparisons.items():
            d=np.array([report['per_video'][a][v]['HOTA']-report['per_video'][b][v]['HOTA'] for v in vids])
            expected=report['paired'][group][name]
            assert abs(float(d.mean())-expected['mean_per_video'])<1e-10
            assert int((d>0).sum())==expected['better'] and len(d)==expected['n']==20
            boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
            assert (ci[0]>0)==(expected['ci95'][0]>0)
            independent[group][name]=dict(mean_per_video=float(d.mean()),ci95=ci)
    proof=dict(passed=True,complete_videos=40,source_hashes_verified=True,numeric_identity_bijection_verified=True,
        all_output_fields_verified=True,geometry_only_boxes=True,identity_only_ids=True,unique_ids_every_frame=True,
        reused_baseline_metrics_exact=True,checked_labels=count,geometry_checked_frames=geometry_rows,
        independent_seed=91027,independent_pairs=independent,baseline_mean_per_video=means,
        controller_all_owned_ended=True,original_budget_not_extended=True,
        report_sha256=sha(folder/'report.json'),source_sha256=sha(__file__))
    write(folder/'independent_audit.json',proof)
    print(json.dumps({k:proof[k] for k in ['passed','complete_videos','checked_labels','baseline_mean_per_video']},ensure_ascii=False))
    return proof

if __name__=='__main__':audit()
