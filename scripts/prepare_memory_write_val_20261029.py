"""Freeze full matched memory evaluation after the accepted development gate."""
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np
    checks=RUN/'memory_write_val_preparation';out=RUN/'memory_write_val'
    proofs={}
    for name in ['memory_write_preparation','memory_write_dev','memory_write_dev_audit']:
        c=service.read(RUN/name/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        proofs[name]=dict(controller_sha256=service.sha(RUN/name/'controller_result.json'),actual_all_owned_ended=True)
    dev=service.read(RUN/'memory_write_dev/report.json');audit=service.read(RUN/'memory_write_dev_audit/report.json')
    assert dev['complete'] and audit['passed'] and audit['full_raw_json_cost_memory_metrics_replay']
    assert dev['metrics']==audit['metrics'] and len(dev['variants'])==8
    mark=service.read(RUN/'memory_write_dev/tracking_completed.json')
    assert all(service.sha(ROOT/f)==h for f,h in mark['sha256'].items())
    gate=service.read(RUN/'memory_write_canonical_input_coverage.json');assert gate['all_complete']
    old=service.read(RUN/'masa_query_fp32_recovery1/protocol.json')
    oldaudit=service.read(RUN/'masa_audit_fp32_recovery1/report.json');assert oldaudit['complete'] and oldaudit['passed']
    assert len(oldaudit['feature_checks'])==9 and max(x['max_difference'] for x in oldaudit['feature_checks'])<1e-5
    files=['tracking/query_memory_write.py','tracking/query_appearance_comparison.py','tracking/lazy_appearance_association.py',
        'tracking/frame_masa_appearance.py','tracking/offline_association.py','tracking/xywh_bytetrack.py',
        'scripts/memory_write_val_20261029.py','scripts/audit_memory_write_val_20261029.py','scripts/record_memory_write_val_20261029.py',
        'scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py',
        'runs/continuation_20261029/memory_write_authorization.json',
        'runs/continuation_20261029/memory_write_dev/protocol.json',
        'runs/continuation_20261029/memory_write_dev/report.json','runs/continuation_20261029/memory_write_dev_audit/report.json',
        'runs/continuation_20261029/memory_write_canonical_input_coverage.json',
        'runs/continuation_20261029/masa_audit_fp32_recovery1/report.json',
        'runs/continuation_20261029/masa_final_checked_record_recovery1/scientific_summary.json']
    for source in old['sources']:
        prior=RUN/'masa_query_fp32_recovery1/videos'/source['video']
        original=service.read(prior/'report.json')
        assert all(service.sha(ROOT/f)==h for f,h in original['sha256'].items())
        files.extend(original['sha256'])
        files.extend([str((prior/'report.json').relative_to(ROOT)),source['detections'],source['baseline'],source['alignment']])
        # Both encoder ledgers must cover the same exact predictor-box keys.
        z=np.load(prior/'vectors.npz')
        for frame in service.read(prior/'tracking_completed.json')['query_frames']:
            assert {tuple(b.tolist()) for b in z[f'{frame}_r50_1333_boxes']}=={tuple(b.tolist()) for b in z[f'{frame}_detic_1333_boxes']}
            assert np.isfinite(z[f'{frame}_r50_1333_vectors']).all() and np.isfinite(z[f'{frame}_detic_1333_vectors']).all()
    assert shutil.disk_usage(ROOT).free>6_000_000_000
    p=dict(id='query_memory_write_v1',created=service.now(),budget_seconds=86400,
        sources=old['sources'],config=old['config'],appearance_weight=old['appearance_weight'],beta=.9,
        new_variants=['r50_mean','r50_cautious','detic_mean','detic_cautious'],
        development_proofs=proofs,first_full_dev_seconds=dev['seconds'],first_dev_cpu_audit_seconds=audit['seconds'],
        interventions=service.read(RUN/'memory_write_dev/protocol.json')['policies'],
        predictor='Identical saved GPU FP32 normalized appearance vectors used bit for bit; only memory writing and resulting causal assignments differ. The new-policy memory/association has always run on CPU, including accepted devGPU. No encoder or completed last-policy inference repeated. Native CPU executes solely as the new zero-shadow-state control, not as a rescored scientific reference.',
        references='Original complete last R501333/Detic1333/native results, outputs and metrics reused by SHA. All first-query input vectors already have independent CPU spot checks0/10/19 within1e-5; exact unchanged inputs justify reuse of these checks, not a new full pixel replay.',
        missing_input_rule='All canonical query detections score>=.1 checked in all20; zero missing. If a new policy requests a missing vector, fail rather than approximate. Any recovery must be distinct, preserve this source and compute only genuinely absent inputs.',
        primary='Four comparisons: cautious-last and cautious-mean separately for R501333/Detic1333; paired video bootstrap20000 seed1;95% and Bonferroni98.75% intervals;mainpositive iff all4 corrected lower bounds>0. No tuning/quality-based variant selection.',
        audit='Separate CPU process replays all20/new4/zero/native-shadow-reference raw,JSON,cost events,memory write vectors,full metrics and bootstrap exactly. SHA checkpoints only after full videos; changed input rejected. New tracker outputs saved before metrics; no reexecution if those outputs already exist.',
        limitations='Same20historicalval already used, not new independent final test. Frozen detector/extractors, no training. Component adaptation of confidence-dependent write, not full Deep OC-SORT, not FlyVis core. EMA beta.9 fixed before dev; score is not identity truth; slow updates can preserve mistakes. Shared input preparation is not complete independent MMDetection implementation.',
        disk='5GB live reserve and 2*first_saved_bytes*19+5GB<free after first full video; binding.',
        time='Forecast only; stage time caps cancelled; save30.10 07:30,ownedstop07:39:30,hardstop07:40 remain.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    assert not (out/'protocol.json').exists();service.write(out/'protocol.json',p)
    for name in ['memory_write_val_audit','memory_write_val_record']:
        service.write(RUN/name/'protocol.json',dict(budget_seconds=86400,parent_sha256=service.sha(out/'protocol.json'),created=service.now()))
    service.write(checks/'dev_verified.json',dict(complete=True,passed=True,proofs=proofs,technical_gate_only=True,
        dev_seconds=dev['seconds'],audit_seconds=audit['seconds'],all_current_query_inputs_available=True,
        exact_gpu_fp32_inputs_reused=True,updated=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');card=next(c for c in h['experiments'] if c['id']==p['id'])
    card['state']='ready';card['acceptance'][0]['met']=True
    card['acceptance'][0]['evidence']='runs/continuation_20261029/memory_write_val_preparation/dev_verified.json'
    card['evidence'].append(dict(kind='file',ref=str((out/'protocol.json').relative_to(ROOT)),claim='Полная проверка до val; старые выходы и точные GPU-векторы переиспользуются'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Полный dev и независимый повтор приняты; фиксация нового val без повторения извлекателя',
        'runs/continuation_20261029/memory_write_val_preparation/dev_verified.json: dev8вариантов/1213кадров107,516с,CPU22,859с,код0/деревьяфактическизакончены. Это техническийгейт,качествоодноговидеонеиспользованодлявыбора/подбора. Все20canonicalqueryROIестьвстарыхFP32векторах:новогоGPU/повторногоencoderинференсаненужно. Начальнаяпроверкасырыхboxключейнекорректнодопускаларасхождениеиз-заxyxy→xywh→xyxyfloat32;провереныточныеObservationTrack.xyxyключи(0пропусков),старыйотчётсохранён. Протоколmemory_write_val/protocol.jsonДОновыхполитикval:4новыхветви,старыеlast/native/20иCPUfeaturechecksпереиспользуютсяпоSHA;памятькаквdevсчиталасьнаCPU. Полныйнезависимыйповторвсехновыхвыходов/памяти/метрик,4primary/95%иBonferroni98,75%;историческийvalненовыйитоговыйтест. Диск5GB/срок30.10 07:40.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_memory_write.py','tests/test_query_appearance_comparison.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as output:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,name,script in [('110','memory_write_val','scripts/memory_write_val_20261029.py'),('120','memory_write_val_audit','scripts/audit_memory_write_val_20261029.py'),('125','memory_write_val_record','scripts/record_memory_write_val_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=str((RUN/name).relative_to(ROOT)),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
