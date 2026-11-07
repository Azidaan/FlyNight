"""Freeze the next full tracker comparison after the independent dev gate."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'assignment_write_val_preparation';folder=RUN/'assignment_write_val'
    for n in ['assignment_write_preparation','assignment_write_dev','assignment_write_dev_audit']:
        c=service.read(RUN/n/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'assignment_write_dev_audit/report.json');assert a['passed'] and a['full_raw_json_events_memory_metrics_replay']
    assert a['counts']['r50_guarded']['rejected']>0 and a['counts']['detic_guarded']['rejected']>0
    old=service.read(RUN/'memory_write_val/protocol.json');p=dict(old)
    p['id']='query_assignment_write_v1';p['created']=service.now()
    p['new_variants']=['r50_guarded','detic_guarded'];p['beta']=.9 # unused by admitted-write tracker; no EMA
    p['interventions']=service.read(RUN/'assignment_write_dev/protocol.json')['action']
    p['predictor']='Only write admission differs. Exact old GPU FP32 vectors, old last/mean/native outcomes reused by SHA; new admitted-last policies use CPU as in the independently accepted dev. Native CPU only the new shadow-state control.'
    p['primary']='Four guarded-last/guarded-mean comparisons in two encoders;paired20000seed1/95%andBonferroni98.75%;all4correctedlower>0. No parameter selection from dev/val.'
    p['limitations']='Same historical20val, not new independent test. Mutual geometric eligibility is not identity truth; rejecting writes can preserve an initially wrong or stale memory. Component adaptation, not complete KeepTrack, not FlyVis core. No new thresholds/training/encoder inference. Dev detector has seen train.'
    files=['scripts/assignment_write_val_20261029.py','scripts/audit_assignment_write_val_20261029.py','scripts/record_assignment_write_val_20261029.py',
        'tracking/query_assignment_write.py','runs/continuation_20261029/assignment_write_dev/protocol.json','runs/continuation_20261029/assignment_write_dev_audit/report.json',
        'runs/continuation_20261029/memory_write_val_audit/report.json']
    for source in p['sources']:
        f=RUN/'memory_write_val/videos'/source['video']/'report.json'
        r=service.read(f);assert r['passed'] and all(service.sha(ROOT/k)==h for k,h in r['sha256'].items())
        files.append(str(f.relative_to(ROOT)))
        files.append('data/bdd100k/labels/box_track_20/val/'+source['video']+'.json')
    p['sha256']={**old['sha256'],**{f:service.sha(ROOT/f) for f in files}}
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    for n in ['assignment_write_val_audit','assignment_write_val_record']:
        service.write(RUN/n/'protocol.json',dict(budget_seconds=86400,parent_sha256=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']==p['id'])
    card['acceptance'][0].update(met=True,evidence='runs/continuation_20261029/assignment_write_dev_audit/report.json')
    card['evidence'].append(dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Полная проверка допуска записи после независимого dev, до новых val-выходов'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Допуск записи прошёл dev; фиксируется полная отдельная проверка',
        'assignment_write_dev_audit/report.json:1213кадров/2новыхguarded/zero-state/всеraw/JSON/cost/write/метрикиточно;1474записи/88отказоввкаждомизвлекателе,devCPU'+str(a['seconds'])+'с. Это техническаяготовность,непольза/невыборпокачеству. Доvalassignment_write_val/protocol.json:двановыхтрекераguarded,готовыеnative/last/meanпоSHAпереиспользованы;GPUвекторыточнопрежние,никакогоencoder/GPU/обучения. 4primaryguarded-last/mean×2encoder/20kseed1/95%иBonferroni98,75%,mainвсе4нижние>0. СохранениевыходовДОметрик/полныйнезависимыйCPUповтор/диск5GB/общийсрок30.10 07:40. Следующее решение по первичным актам, историческийvalненовыйитоговыйтест. Этоотдельныйопыт,предыдущийотрицательныйEMAнепереписывается.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_assignment_write.py','tests/test_obsidian_fundamental_overview.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as output:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,name,script in [('180','assignment_write_val','scripts/assignment_write_val_20261029.py'),('190','assignment_write_val_audit','scripts/audit_assignment_write_val_20261029.py'),('195','assignment_write_val_record','scripts/record_assignment_write_val_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=str((RUN/name).relative_to(ROOT)),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
