"""Freeze bounded write comparison only after accepted exact development replay."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'bounded_write_val_preparation';folder=RUN/'bounded_write_val'
    for n in ['bounded_write_preparation','bounded_write_dev','bounded_write_dev_audit']:
        c=service.read(RUN/n/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'bounded_write_dev_audit/report.json');assert a['passed'] and a['full_raw_json_events_memory_metrics_replay']
    assert all(a['counts'][n]['accepted']>0 for n in ['r50_bounded','r50_periodic','detic_bounded','detic_periodic'])
    old=service.read(RUN/'memory_write_val/protocol.json');p=dict(old);p['id']='query_bounded_write_v1';p['created']=service.now()
    dev=service.read(RUN/'bounded_write_dev/protocol.json');p['new_variants']=dev['new_variants'];p['beta']=.9 # engine compatibility only, unused; neither policy averages
    p['interventions']=dev['action'];p['age']=dev['age']
    p['predictor']='Exact old GPU FP32 vectors. Only write admission differs; new4 on CPU as in accepted dev. Old native/last/unbounded guarded outcomes reused by SHA; no image inference/training/old scientific run repeated. Native CPU solely new zero-state control.'
    p['primary']='Four bounded-last and bounded-periodic comparisons across two encoders, paired20000seed1,95%andBonferroni98.75%,main iff all4correctedlower>0. No age sweep or quality selection.'
    p['mechanism_secondary']='Bounded-unbounded guarded in two encoders reported separately with95%and conservative98.75%intervals; not additional acceptance criteria, not identity truth. This matched change addresses write delay but forced replacement also changes the observed history.'
    p['limitations']='Same20historicalval after previous exploratory comparisons, not independent final test. Forced updates may overwrite correct identity; geometric uniqueness is not identity truth. Age chosen as original max_frames_lost=30, no new numeric hyperparameter. New classes do not average; beta field is unused engine compatibility. Component adaptation, not full KeepTrack or FlyVis core.'
    files=['scripts/bounded_write_val_20261029.py','scripts/audit_bounded_write_val_20261029.py','scripts/record_bounded_write_val_20261029.py',
        'tracking/query_bounded_write.py','tracking/query_assignment_write.py','scripts/assignment_write_dev_20261029.py',
        'runs/continuation_20261029/bounded_write_dev/protocol.json','runs/continuation_20261029/bounded_write_dev_audit/report.json',
        'runs/continuation_20261029/assignment_write_val_audit/report.json']
    for s in p['sources']:
        f=RUN/'assignment_write_val/videos'/s['video']/'report.json';r=service.read(f)
        assert r['passed'] and all(service.sha(ROOT/k)==h for k,h in r['sha256'].items())
        files.extend([str(f.relative_to(ROOT)),'data/bdd100k/labels/box_track_20/val/'+s['video']+'.json'])
    p['sha256']={**old['sha256'],**{f:service.sha(ROOT/f) for f in files}}
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    for n in ['bounded_write_val_audit','bounded_write_val_record']:
        service.write(RUN/n/'protocol.json',dict(budget_seconds=86400,parent_sha256=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');card=next(x for x in h['experiments'] if x['id']==p['id'])
    card['acceptance'][0].update(met=True,evidence='runs/continuation_20261029/bounded_write_dev_audit/report.json')
    card['evidence'].append(dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Полный протокол ограничения записи и равного периодического контроля до val'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Ограничение задержки записи: полный протокол после независимого dev',
        'bounded_write_dev_audit/report.json:полныйфиксированныйdev/4новыхварианта/нулевоесостояниеточно/всеraw/JSON/events/write/метрикиповторены. Срокобновленияmax_frames_lost30изисходноготрекера,неподборчисла;невременнойбюджетэтапа. Новыйbounded_write_val/protocol.jsonДОval:4новыхbounded/periodic×2encoder,готовыеlast/native/guardedпоSHAиТОЧНЫЕGPUвекторынеповторяются. 4primarybounded-last/periodic/20kseed1/95%иBonferroni98,75%;secondarybounded-guardedнеприёмка/неистиннаяличность. ПолныйотдельныйCPUповтор/диск5GB/срок30.10 07:40. Выборправкипослепрошлыхрезультатовделаетисторическийvalисследовательским,не независимымитоговымтестом. Никакогоencoder/обучения/новыхданных.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_bounded_write.py','tests/test_obsidian_fundamental_overview.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as output:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,n,s in [('250','bounded_write_val','scripts/bounded_write_val_20261029.py'),('260','bounded_write_val_audit','scripts/audit_bounded_write_val_20261029.py'),('265','bounded_write_val_record','scripts/record_bounded_write_val_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{n}.json',dict(name=n,folder=str((RUN/n).relative_to(ROOT)),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(RUN/n/'protocol.json')))

if __name__=='__main__':main()
