"""Next cheap write-policy action in the authorized memory branch, not val tuning."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    checks=RUN/'assignment_write_preparation';out=RUN/'assignment_write_dev'
    c=service.read(RUN/'memory_write_agent_record/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    previous=service.read(RUN/'memory_write_dev/protocol.json')
    inputs=service.read(RUN/'memory_write_dev/tracking_completed.json')['sha256']
    assert all(service.sha(ROOT/f)==h for f,h in inputs.items())
    files=['tracking/query_assignment_write.py','tests/test_query_assignment_write.py',
        'scripts/assignment_write_dev_20261029.py','scripts/audit_assignment_write_dev_20261029.py',
        'tracking/query_appearance_comparison.py','tracking/lazy_appearance_association.py','tracking/frame_masa_appearance.py','tracking/xywh_bytetrack.py',
        'runs/continuation_20261029/memory_write_authorization.json',
        'runs/continuation_20261029/memory_write_first_divergence/report.json',
        'runs/continuation_20261029/memory_write_dev/protocol.json',
        previous['source']['detections'],previous['source']['alignment'],*inputs]
    p=dict(id='query_assignment_write_v1',created=service.now(),budget_seconds=86400,
        source=previous['source'],config=previous['config'],appearance_weight=previous['appearance_weight'],
        basis='Agent-selected next write-policy action under direct approved memory experiment and AGENTS6.6/9.3; not an additional direct human order naming KeepTrack.',
        action='Keep the last admitted real query observation unchanged. After initial seeding, admit a new write only when its track has exactly one eligible first-association detection and that detection has exactly one eligible track, using the unchanged original cost/match_thresh before appearance. Do not alter assignments, geometry, detections, query times, features or the action cost rule. Second-stage/ambiguous writes rejected; no EMA and no new numeric threshold.',
        source_mechanism='https://raw.githubusercontent.com/visionml/pytracking/master/pytracking/tracker/keep_track/keep_track.py; update_flag excludes uncertain/not_found. Our one-to-one admission proxy is a component adaptation, not full KeepTrack or the same uncertainty estimator.',
        strongest_counterargument='Unique geometric eligibility does not prove identity. Rejected writes can retain a wrong seed or stale appearance; this action may make the failure worse.',
        preceding_diagnostic='One read-only earliest-divergence check:24of40pairs differ; all24 first changed outputs occur at query, recorded original cost/eligibility unchanged, appearance cost differs. It localizes the intervention path but does not identify truth or prove pollution.',
        dev_gate='One previously fixed first fit20video, two new guarded trackers plus native-shadow/zero, exact CPU replay of all raw/JSON/cost/write decisions/metrics. Exact existing GPU FP32 dev vectors reused, no encoder inference, no new data/training. No quality-based parameter selection.',
        prospective_val='Only after passed dev and actual PID end, freeze separate full20val sources/reuse/protocol. Guarded-last and guarded-mean in two encoders:4primary/paired20000seed1/ordinary95% and Bonferroni98.75%,all4 correctedlower>0. Full independent replay; historical20 not new final test. Current run is readiness only.',
        deadline='Save30.10 07:30;allownedstop07:39:30;hardstop07:40;stage-time gates cancelled, disk5GBbinding.',
        sha256={f:service.sha(ROOT/f) for f in sorted(set(files))})
    assert not (out/'protocol.json').exists();service.write(out/'protocol.json',p)
    service.write(RUN/'assignment_write_dev_audit/protocol.json',dict(budget_seconds=86400,parent_sha=service.sha(out/'protocol.json'),created=service.now()))
    h=service.read(ROOT/'experiments/project_history.json');assert not any(x['id']==p['id'] for x in h['experiments'])
    h['experiments'].append(dict(id=p['id'],title='Запись памяти только после однозначного назначения — без усреднения',area='reid',state='ready',started='2026-10-29',
        question='Поможет ли отделение назначения от допуска записи при прежних детекциях и признаках?',
        hypothesis='Отказ от записи при неоднозначном назначении может защитить последнее принятое наблюдение; загрязнение личности пока не установлено.',
        expected='Сначала полная техническая проверка dev и независимый CPUповтор, затем отдельно замороженная полная проверка guarded против last/mean.',
        acceptance=[dict(criterion=p['dev_gate'],met=None),dict(criterion=p['prospective_val'],met=None)],
        cannot_claim='Подготовка не доказывает пользу. Однозначность геометрии не подтверждает личность; отказ от записи может сохранять ошибку или устаревшую внешность. Это отдельный компонент, не полный KeepTrack, не ядро FlyVis. Новый val не запускался.',
        evidence=[dict(kind='file',ref=str((out/'protocol.json').relative_to(ROOT)),claim='Следующее действие зарегистрировано до dev, без подбора по историческому val')]))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'Следующее действие в разрешённой проверке памяти: отделить назначение от записи',
        'Осторожная EMAхужеlastпослеBonferroniсобоимиизвлекателями(memory_write_val_audit/report.json),версиянепродвигается. ПоAGENTS6.6/9.3выбранаследующаязаменаузлазаписи,неподборbeta:новаязаписьтолькодляединственнойвзаимнодопустимойпарыпервогоназначенияпопрежнемуoriginalcost/match_thresh;иначесохранитьпоследнююпринятую. Новыхчисленныхпорогов/обучения/encoderинференсанет. KeepTrackпервичныйисточникразделяетобновлениеприпотере/неопределённости,этокомпонентнаяадаптация,неполныйметод. Сильнаяальтернатива—заморозитсяошибочнаяилиустаревшаяпамять. Однадиагностикапервогорасхожденияготова:24из40партрекеров,24из24содинаковымиoriginalcost/eligibleиноразнойappearance;непроверкаGT/невиновникдоказанный. Источник memory_write_first_divergence/report.json. Новыйпротоколassignment_write_dev/protocol.jsonДОdev:теже2encoder/тежеGPUвекторы/фиксированныйdev1213кадров,полныйCPUповтор. Valтолькопослепроверкииотдельнойзаморозки,4primary/95%/Bonferroni98,75%;готовыеlast/meanнеповторяются. Общийсрок30.10 07:40неизменён.')
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_query_assignment_write.py','-q']]):
        with (checks/f'check_{i}.log').open('w',encoding='utf-8') as outlog:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=outlog,stderr=subprocess.STDOUT,timeout=420).returncode
        assert r==0,(i,r)
    service.write(checks/'report.json',dict(complete=True,passed=True,updated=service.now()))
    for index,name,script in [('150','assignment_write_dev','scripts/assignment_write_dev_20261029.py'),('160','assignment_write_dev_audit','scripts/audit_assignment_write_dev_20261029.py')]:
        service.write(RUN/f'ready_jobs/{index}_{name}.json',dict(name=name,folder=str((RUN/name).relative_to(ROOT)),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(RUN/name/'protocol.json')))

if __name__=='__main__':main()
