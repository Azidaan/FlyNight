"""Freeze the full first-problem pilot after conditional selection and independent audit."""
import sys,shutil,ast,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK),'No new full pilot in the last5minutes of the author window.'
    phase();report=read(RUN/'comparison_report.json');assert report['full_tracker_gate']['proceed'] and read(RUN/'independent_audit.json')['passed']
    parent=BASE/'full_precision';assert read(parent/'analyses_complete.json')['passed'];full=RUN/'full';full.mkdir(exist_ok=True)
    selection=read(RUN/'selection.json');old=read(parent/'protocol.json');first_protocol=read(RUN/'protocol.json')
    p=dict(id='singleton_free_full_v1_20261108',videos=old['videos'],native_config=old['native_config'],
        authorization='Explicit first-problem priority and authorized matched comparison; prospective continuation after the fixed diagnostic gate.',
        active_problem=1,deferred_problems=[2,3,4],deadline='2026-11-08T14:00:00+03:00',
        parent_protocol=dict(path=str((parent/'protocol.json').relative_to(ROOT)),sha256=sha(parent/'protocol.json')),
        cohort='Same20historical val videos, detector-unseen but researcher-seen. No parameter selection on them.',
        input='Original causal Retina, native dtype before resize,391x391 and721BoxEye columns; same source checkpoint and original normalization.',
        inference=first_protocol['full_acceptance'],parameters=selection['parameters'],aliases={})
    for arm in ['U_GEO','U_THRESHOLD']:
        if p['parameters'][arm]['alpha']==0:p['aliases'][arm]='B0'
    for arm in ARMS[3:]:
        if p['parameters'][arm]['beta']==0:p['aliases'][arm]='U_GEO'
    actual=[a for a in ARMS if a!='B0' and a not in p['aliases']]
    p['cpu_arms']=[a for a in actual if a not in ['U_FLYVIS','U_PERM']];p['gpu_arms']=[a for a in actual if a in ['U_FLYVIS','U_PERM']]
    p['selection_sha256']=sha(RUN/'selection.json');p['conditional_sha256']=sha(RUN/'comparison_report.json');p['registered']=datetime.now(MSK).isoformat()
    p['primary_comparisons']=read(RUN/'protocol.json')['primary_comparisons'];p['first_problem_protocol_sha256']=sha(RUN/'protocol.json')
    p['safety_addition_before_full_run']=dict(criterion='Known wrong-target return count must not increase for any accepted comparator.',
        reason='The author specifically requires avoiding extra foreign assignments. HOTA/IDSw/FP/FN alone do not explicitly enforce this return count.',
        status='Added before any new full rollout or its outcomes; same sparse return definition as the parent, unknowns explicitly counted.')
    p['intervention_clarification']='The preassociation Kalman mean/covariance is not directly replaced. Standard updates after a new assignment remain active; future states and histories may consequently differ.'
    p['acceptance_level_clarification_before_full_run']='Useful-system acceptance of U_FLYVIS uses its already-registered comparison againstB0 plus all harm guards againstB0. Additional FlyVis contribution still requires every matched-control comparison. This distinguishes the author system from the core, perAGENTS§1; it adds no comparison, changes no confidence level and is recorded before full outcomes. At beta0 the signal is inactive and its variant is an exact geometry alias.'
    p['disposable_scratch']='Decoded full-frame float64 input may be released only after every arm output and metric is sealed; all used retinal views/maps, original source movies and checkpoints retained.'
    write(full/'protocol.json',p,new=True);shutil.copytree(parent/'runtime_inputs',full/'runtime_inputs')
    for video in p['videos']:
        for kind,suffixes in [('outputs',['.json']),('metrics',['.json','.details.json.gz'])]:
            for suffix in suffixes:
                source=parent/kind/'B0'/(video+suffix);dest=full/kind/'B0'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    sources=dict(read(RUN/'phase.json')['sources'])
    for path in (ROOT/'research_singleton_free_20261108').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['tracking/classical_motion.py','tracking/flyvis_activity.py','research_spatial_identity_20261107/extract_reference.py',
        'research_identity_assignment_run_20261108/full_precision_io.py','research_identity_assignment_run_20261108/full_retina.py',
        'research_identity_assignment_run_20261108/full_planner.py','research_identity_assignment_run_20261108/full_run.py',
        'research_identity_assignment_run_20261108/full_evaluate.py','research_identity_assignment_run_20261108/full_additional.py',
        'research_identity_assignment_run_20261108/full_statistics.py','evaluation/hota_eval.py']:
        sources[name]=sha(ROOT/name)
    inherited=read(BASE/'full_precision_execution_phase.json')['sources']
    for name,h in inherited.items():
        assert sha(ROOT/name)==h
        sources.setdefault(name,h)
    for path in [full/'protocol.json',RUN/'selection.json',RUN/'comparison_report.json',RUN/'independent_audit.json',parent/'analyses_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,deadline='2026-11-08T14:00:00+03:00'),new=True)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — По заранее зарегистрированномуJ-гейту и после независимого аудита начат полный исторический20val-пилот первой проблемы `singleton_free_cost_v1_20261108`. Параметры неизменно взяты изfit20-selection; девять вариантов,11первичных сравнений,20тысячпарных выборокпо видео, исправленный уровеньДИ99,54545%. До полного прогона явно добавлено ограничение: число известных чужих возвратов не увеличивается; прежниеHOTA/AssA/IDF1/IDSw/FP/FN-ограничения сохранены. Исходные назначения сохраняются внутри текущего кадра; прогноз не подменяется до сопоставления, но штатные обновления после возврата изменяют будущие состояния. Это уточняет прежнюю краткую фразу о неизменности состоянияКалмана. ГотовыеB0траектории/метрики и совпавшие карты переиспользуются; проверяется новая полная нулевая ветвь. Источники `runs/singleton_free_20261108/full/protocol.json`, `full_phase.json`. Остальные проблемы отложены.\n\n"
    write(RUN/'full_registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    with (full/'manager.log').open('x',encoding='utf-8') as out,(full/'manager.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/full_manager.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),script='research_singleton_free_20261108/full_manager.py')
    write(full/'manager_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(full/'deadline_registered.json',dict(passed=True,root=root,deadline=p['deadline']),new=True)


if __name__=='__main__':main()
