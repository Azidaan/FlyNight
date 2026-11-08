import sys,ast,time,gzip,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery2_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import bounded_write


def main():
    import psutil
    from evaluation.hota_eval import VALID
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);assert not (RECOVERY/'started.json').exists()
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    recovery_phase();assert read(FULL/'analyses_complete.json')['passed'] and not (FULL/'independent_audit.json').exists()
    failed=read(RECOVERY1/'children/full_audit.ended.json');assert failed['returncode']==1 and failed['all_owned_ended']
    roots={int(pid):birth for pid,birth in failed['guardian']['owned'].items()};parent=read(RECOVERY1/'started.json');roots[parent['pid']]=parent['created']
    for pid,birth in roots.items():
        try:p=psutil.Process(pid);assert abs(p.create_time()-birth)>=.01 or not p.is_running(),pid
        except psutil.NoSuchProcess:pass
    completed=list((FULL/'completed').glob('*.json'));assert len(completed)==20
    for path in completed:
        for name,h in read(path)['artifacts'].items():assert sha(ROOT/name)==h
    checked=0
    for path in (FULL/'metrics').glob('*/*.json'):
        m=read(path);details=path.with_suffix('.details.json.gz')
        with gzip.open(details,'rt',encoding='utf-8') as f:d=json.load(f)[m['video']]
        assert all('IDSW' in d[c]['CLEAR'] and 'IDSw' not in d[c]['CLEAR'] for c in VALID if c in d)
        assert sum(d[c]['CLEAR']['IDSW'] for c in VALID if c in d)==m['metrics']['IDSw'];checked+=1
    assert checked==200
    write(RECOVERY/'field_alias_probe.json',dict(passed=True,reports=checked,official_serialized_field='CLEAR.IDSW',public_scalar_field='metrics.IDSw',all_values_match=True,
        evaluator_mapping_source_sha256=sha(ROOT/'evaluation/hota_eval.py'),new_evaluations=0,new_tracker_updates=0),new=True)
    protocol=dict(registered=datetime.now(MSK).isoformat(),reason='Auditor used public scalar spelling IDSw inside official CLEAR block, whose actual key is IDSW.',
        correction='Separate copies map only CLEAR.IDSw lookup to CLEAR.IDSW. All assertions, datasets, outcomes, numerical values, bootstrap criteria and tolerances remain unchanged. Original failed sources/logs retained.',
        action='Resume read-only full audit, then registered AGEcontrol and direct-error diagnostic. All20full trajectories/official evaluations already complete and must not be repeated.',
        original_failure_sha256=sha(RECOVERY1/'children/full_audit.log'),deadline='2026-11-08T14:00:00+03:00')
    write(RECOVERY/'protocol.json',protocol,new=True)
    entry=(f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Полный пилот собственного шаблона завершён на 20 видео (`runs/singleton_verify_20261108/full/analyses_complete.json`). Первый независимый аудит остановился до завершения первого видео из-за моей ошибки имени поля: в официальном подробном CLEAR ключ `IDSW`, тогда как публичная сводка проекта называет его `IDSw`. Во всех 200 сохранённых отчётах прямой пересчёт `CLEAR.IDSW` совпал со сводкой (`full_recovery2/field_alias_probe.json`); TrackEval и трекер повторно не запускались. Исправление сделано отдельными копиями аудиторов `research_singleton_verify_full_recovery2_20261108/`, меняется только обращение к сериализованному полю, все проверки/значения/допуски сохранены. Прежний KeyError и неуспешный контроллер остаются (`full_recovery1/children/full_audit.log`, `full_audit.ended.json`). Новый корень выполняет аудит, контроль давности и разбор ошибок последовательно до прежнего срока 14:00 МСК. Предварительная сводка полного пилота не прошла критерии; научный итог фиксируется после независимой сверки, без изменения условий.\n\n")
    write(RECOVERY/'registration_journal.json',dict(entry=entry),new=True);plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1;plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    sources=read(RECOVERY1/'phase.json')['sources'].copy()
    for path in Path(__file__).parent.glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [RECOVERY/'protocol.json',RECOVERY/'field_alias_probe.json',RUN/'full_audit_phase.json',FULL/'comparison_report.json',FULL/'additional_report.json',FULL/'analyses_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RECOVERY/'phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only_auditor_schema_recovery=True),new=True)
    bounded_write(RECOVERY/'status.json',dict(stage='registered_before_audit',updated=datetime.now(MSK).isoformat()))
    with (RECOVERY/'controller.log').open('x',encoding='utf-8') as out,(RECOVERY/'controller.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_recovery2_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_singleton_verify_full_recovery2_20261108/controller.py');write(RECOVERY/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)
    write(RECOVERY/'deadline_registered.json',dict(passed=True,root=root,deadline=protocol['deadline']),new=True)


if __name__=='__main__':main()
