"""Prospective lifetime-bounded write action with a matched periodic control."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'bounded_write_preparation';folder=RUN/'bounded_write_dev'
    c=service.read(RUN/'assignment_write_agent_record/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    old=service.read(RUN/'memory_write_dev/protocol.json')
    files=['tracking/query_bounded_write.py','tracking/query_assignment_write.py','tests/test_query_bounded_write.py',
        'scripts/bounded_write_dev_20261029.py','scripts/audit_bounded_write_dev_20261029.py','scripts/assignment_write_dev_20261029.py',
        'runs/continuation_20261029/memory_write_authorization.json','runs/continuation_20261029/memory_write_dev/protocol.json',
        'runs/continuation_20261029/assignment_write_mechanism_review/report.json',
        'runs/continuation_20261029/assignment_write_val_audit/report.json',old['source']['detections'],old['source']['alignment'],
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
    inputs=service.read(RUN/'memory_write_dev/tracking_completed.json')['sha256']
    assert all(service.sha(ROOT/f)==h for f,h in inputs.items());files.extend(inputs)
    p=dict(id='query_bounded_write_v1',created=service.now(),budget_seconds=86400,source=old['source'],config=old['config'],appearance_weight=old['appearance_weight'],
        new_variants=['r50_bounded','r50_periodic','detic_bounded','detic_periodic'],
        action='Bounded admits a unique first-pair write as before OR forces replacement by the currently assigned real query observation once age>=the unchanged tracker.max_frames_lost. Periodic ignores uniqueness and replaces only after that same age. Both initialize from the first assigned observation. No averaging, no assignment cancellation, no new numeric threshold; expiry is considered after real assignment, not after predicted-only frames.',
        age='Uses actual installed max_frames_lost=args.track_buffer,30 in retained config, chosen by original tracker lifetime rather than quality optimization. Does not promise a strict read-age bound during missing observations.',
        strongest_counterargument='Forced refresh can overwrite a correct memory with a wrong identity; periodic age and unique geometry are not identity labels. Bounded can still be worse than last.',
        basis='Next single write-policy implementation within direct approved memory branch and AGENTS6.6/9.3, after measured stale unbounded memory; not a new human instruction naming this formula.',
        dev_gate='One existing fixed first fit20video and four new write variants; same exact cached GPU FP32 inputs; full separate CPU raw/JSON/cost/write/metric replay and zero-state. No encoder inference/training/quality-based selection.',
        prospective_val='Separate protocol only after passed dev. Four primary bounded-last and bounded-periodic across two encoders;paired20000seed1/95%andBonferroni98.75%;main iff all4correctedlower>0. References last and completed vectors reused. Historical20 not independent final test. No age sweep.',
        prelaunch_failure='runs/continuation_20261029/bounded_write_prelaunch_failure/failure.json:wrong attribute max_time_lost, source saved; max_frames_lost verified before registration. No scientific run was started.',
        deadline='Save30.10 07:30/ownedstop07:39:30/hardstop07:40,stage-timecaps cancelled,disk5GBbinding.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    assert not (folder/'protocol.json').exists();service.write(folder/'protocol.json',p)
    service.write(RUN/'bounded_write_dev_audit/protocol.json',dict(budget_seconds=86400,parent_sha=service.sha(folder/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');assert not any(x['id']==p['id'] for x in h['experiments'])
    h['experiments'].append(dict(id=p['id'],title='Свежесть памяти — ограниченный допуск записи против периодического обновления',area='reid',state='ready',started='2026-10-29',
        question='Поможет ли ограничение задержки записи исходным сроком жизни трека при тех же признаках и назначениях?',
        hypothesis='Ограниченный допуск может избежать накопления устаревших наблюдений; причиной ухудшения давность ещё не доказана.',
        expected='Сначала полный технический dev и независимый повтор, затем отдельный полный протокол bounded против last/periodic.',
        acceptance=[dict(criterion=p['dev_gate'],met=None),dict(criterion=p['prospective_val'],met=None)],
        cannot_claim='Подготовка не доказывает пользу. Давность и качество различаются в разных траекториях, причинный вклад возраста пока не установлен. Принудительная запись может сохранить чужую личность. Порог возраста не подбирается: взят исходный max_frames_lost. Это отдельное правило памяти, не полный KeepTrack/ядроFlyVis. Новый val не запускался.',
        evidence=[dict(kind='file',ref=str((folder/'protocol.json').relative_to(ROOT)),claim='Одно действие и равный периодический контроль до нового dev')]))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Следующая адресная правка: срок обновления памяти равен исходному сроку жизни трека',
        'НеограниченныйдопускзаписихужеlastпослеBonferroniвобоихencoder. assignment_write_mechanism_review/report.json:средняячитаемаядавностьвспорныхсопоставленияхR50/Deticс8,986/8,975до55,090/55,107кадров,максимум66→613;эторазныенаборысобытий,непричинностьдоказанная. НовоедействиепоAGENTS6.6/9.3вразрешённойпамяти:дополнитьguardedпринудительнойзаписьюпослеmax_frames_lostприполученииреальногонаблюдения;сильныйконтрольтолькопериодическисосрокомтемже. Установленноеmax_frames_lost=args.track_buffer30,неподборчислапокачеству. Новыхпорогов/обучения/encoderнет;тесамыепризнаки/детекции/query/.1/геометрия. Ошибкаимениполяmax_time_lostвCPUunitдозаморозкисохраненавbounded_write_prelaunch_failure,исправлена1passed3,29с. Новыйпротоколbounded_write_dev/protocol.jsonДОdev:4новыхварианта/полныйфиксированныйfit20dev/CPUповтор. Valотдельнопослетехническогогейта,4primarybounded-last/periodic×2encoder/95%/Bonferroni98,75%. Неподбиратьвозрастпопрошлымval;этоисторическиеданные,неновыйитоговыйтест.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_bounded_write.py','tests/test_query_assignment_write.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as output:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,name,script in [('220','bounded_write_dev','scripts/bounded_write_dev_20261029.py'),('230','bounded_write_dev_audit','scripts/audit_bounded_write_dev_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=str((RUN/name).relative_to(ROOT)),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
