"""Freeze full cost-use validation only after a complete accepted CPU dev replay."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'cost_use_val_preparation';folder=RUN/'cost_use_val'
    for n in ['cost_use_preparation','cost_use_dev','cost_use_dev_audit']:
        c=service.read(RUN/n/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'cost_use_dev_audit/report.json');assert a['passed'] and a['full_raw_json_events_memory_metrics_replay'] and a['zero_raw_state_exact']
    dev=service.read(RUN/'cost_use_dev/protocol.json')
    assert set(a['metrics'])==set(dev['new_variants']) and a['technical_gate_only']
    old=service.read(RUN/'memory_write_val/protocol.json');p=dict(old)
    p.update(id=dev['id'],created=service.now(),new_variants=dev['new_variants'],beta=None,neutral_distance=.5,
        action=dev['action'],strongest_counterargument=dev['strongest_counterargument'],primary=dev['prospective_primary'],
        partial_confirmation_rule=dev['partial_confirmation_rule'],
        predictor='Only3new CPU cost trackers using EXACT old audited GPU FP32 vectors. Memory last unchanged; original geometry/query/weight unchanged. Old native/last scientific outputs and metrics reused by SHA. Native CPU only zero-state control. No encoder inference/decode/training/new data.',
        limitations='Same20historicalval after sequential exploratory studies, not independent final test. Common shift may account for improvement; centered-old alone cannot identify useful appearance content. Not full Deep SORT/BoT-SORT/FlyVis core. Old9encoderchecks reused by SHA, no new pixel replay. beta unused compatibility field, no EMA.',
        secondary_native='Three planned new-native comparisons reported with95%,Bonferroni98.75%,and family7 ci_global7; not primary acceptance. No partial confirmation unless family7lower>0.',
        extractor_descriptive='Detic-centered minus R50-centered, ordinary95% only, descriptive, not eligible for confirmed_results.')
    files=['scripts/cost_use_val_20261029.py','scripts/audit_cost_use_val_20261029.py','scripts/record_cost_use_val_20261029.py',
        'tracking/query_cost_use.py','scripts/memory_write_val_20261029.py',
        'runs/continuation_20261029/cost_use_dev/protocol.json','runs/continuation_20261029/cost_use_dev_audit/report.json',
        'runs/continuation_20261029/cost_use_authorization.json',
        'runs/continuation_20261029/masa_audit_fp32_recovery1/report.json',
        'runs/continuation_20261029/masa_final_checked_record_recovery1/scientific_summary.json',
        'runs/continuation_20261029/memory_write_canonical_input_coverage.json',
        'runs/continuation_20261029/memory_write_source_protocol_verified.json']
    for s in p['sources']:
        f=RUN/'masa_query_fp32_recovery1/videos'/s['video']/'report.json';r=service.read(f)
        assert r['passed'] and all(service.sha(ROOT/k)==h for k,h in r['sha256'].items())
        files.extend([str(f.relative_to(ROOT)),'data/bdd100k/labels/box_track_20/val/'+s['video']+'.json'])
    p['sha256']={**old['sha256'],**dev['sha256'],**{f:service.sha(ROOT/f) for f in files}}
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    for n in ['cost_use_val_audit','cost_use_val_record']:
        service.write(RUN/n/'protocol.json',dict(budget_seconds=86400,parent_sha256=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']==p['id'])
    card['acceptance'][0].update(met=True,evidence='runs/continuation_20261029/cost_use_dev_audit/report.json')
    card['evidence'].append(dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Полная проверка сигнала против общего сдвига до новых val выходов'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Полный протокол применения признаков после технического dev',
        'cost_use_dev_audit/report.json:первый фиксированный1213кадров dev/3новых/zero/native-stateexact/всеraw/JSON/cost/write/метрикиповторены. Качество не использовано для настройки. cost_use_val/protocol.json ДОval:3новыхтрекера centeredR50/centeredDetic/общийshift-only; last-память неизменна. Старые20native/last и ТОЧНЫЕ GPUFP32векторыпоSHA без повторногоencoder. Четыре primarycentered-old/shift×2encoder/20kseed1/95%иBonf98,75%,mainвсе4lower>0. Три вторичных native,частныеподтверждениятолькосдополнительнойfamily7lower>0;извлекателиописательно. Полный20CPUповторвсехвыходов/памяти/метрик/paired. Диск5GB/готовыеtracking_completedнепересчитывать/метрикитольковосстановить. Исторические20неновыйитоговыйтест. Save30.10 07:30/stop07:40.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_cost_use.py','tests/test_obsidian_fundamental_overview.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,n,s in [('340','cost_use_val','scripts/cost_use_val_20261029.py'),('350','cost_use_val_audit','scripts/audit_cost_use_val_20261029.py'),('360','cost_use_val_record','scripts/record_cost_use_val_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{n}.json',dict(name=n,folder=str((RUN/n).relative_to(ROOT)),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(RUN/n/'protocol.json')))

if __name__=='__main__':main()
