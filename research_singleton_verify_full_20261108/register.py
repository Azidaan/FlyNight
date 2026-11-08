"""Freeze the full20val pilot after the independently verified screening gate."""
import sys,subprocess,time,shutil,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    report=read(RUN/'comparison_report.json');audit=read(RUN/'independent_audit.json');assert report['full_tracker_gate']['proceed'] and audit['passed'] and audit['full_gate']
    assert sha(RUN/'comparison_report.json')==audit['comparison_sha256']
    ended=read(RUN/'children/parent_full_audit.ended.json');assert ended['returncode']==0 and ended['all_owned_ended']
    assert read(BASE/'full_precision/independent_audit.json')['passed']
    selection=read(RUN/'selection.json');original=read(BASE/'full_precision/protocol.json');arms=['B0','GEO','THRESHOLD','SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']
    FULL.mkdir(exist_ok=True);assert not (FULL/'protocol.json').exists()
    parameters={};aliases={}
    for arm in arms:
        threshold='off' if arm=='B0' else selection['thresholds'].get(arm,-1)
        parameters[arm]=dict(alpha=0 if arm=='B0' or threshold=='off' else .2,beta=0,threshold=threshold)
        if arm!='B0' and threshold=='off':aliases[arm]='B0'
    actual=[a for a in arms if a!='B0' and a not in aliases];system=selection['selected_system']
    p=dict(id='singleton_own_template_full_v1_20261108',registered=datetime.now(MSK).isoformat(),active_problem=1,deferred=[2,3,4],deadline='2026-11-08T14:00:00+03:00',
        videos=original['videos'],native_config=original['native_config'],parameters=parameters,aliases=aliases,selected_system=system,
        cpu_arms=[a for a in actual if a not in ['FLYVIS','PERM']],gpu_arms=[a for a in actual if a in ['FLYVIS','PERM']],
        primary_comparisons=[(system,'B0'),('SUPPORT','B0')]+[('FLYVIS',a) for a in arms if a!='FLYVIS'],
        inference=read(RUN/'protocol.json')['full_acceptance'],selection_sha256=sha(RUN/'selection.json'),conditional_sha256=sha(RUN/'comparison_report.json'),
        cohort='Same20historical val videos; detector-unseen, researcher-seen. No tuning on full pilot.',
        intervention='Native first-stage pairs fixed; the same free sole-geometric edge gets costalpha0.2 only when its own-template score passes its frozen threshold. Ordinary later lifecycle and Kalman updates remain active.',
        input='Original Retina native per-frame dtype before resize;391x391BoxEye and721columns, same kernel/checkpoint/normalizations and causal4observation prefix plus search.',
        closure_addition_before_full_run=dict(criterion='To call problem1solved, accepted selected system must also increase both known correct return count and sustained-success count versusB0.',
            definition='Same previously frozen sparse-label return definition: correct at return and at next2available target observations; unknown/censored separate.',
            reason='The requested identity recovery must be demonstrated beyond one frame; a HOTA increase alone does not prove it. Added before any full rollout outcomes, without weakening earlier criteria.'),
        source_level='Whole author tracking system; no change to FlyVis core. Additional core/readout contribution requires all matched-control comparisons.',
        scratch='Decoded float64 full frames are disposable only after every arm output/metric is sealed. Original movies, all used retinal views/maps and causal checkpoints are retained.',
        exact_reuse='Reuse complete parent feature sequences, retinal views and official metrics only under matching source hashes and full-output-array identity. No repeated detector or baseline evaluation.')
    write(FULL/'protocol.json',p,new=True);shutil.copytree(BASE/'full_precision/runtime_inputs',FULL/'runtime_inputs')
    for video in p['videos']:
        for kind,suffixes in [('outputs',['.json']),('metrics',['.json','.details.json.gz'])]:
            for suffix in suffixes:
                source=BASE/'full_precision'/kind/'B0'/(video+suffix);dest=FULL/kind/'B0'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    sources=read(RUN/'comparison_phase.json')['sources'].copy()
    for document in ['full_precision_execution_phase.json','feature_phase.json']:
        for n,h in read(BASE/document)['sources'].items():assert sha(ROOT/n)==h;sources[n]=h
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['research_singleton_free_20261108/full_manager.py','research_singleton_free_20261108/full_features.py',
        'research_singleton_free_20261108/full_paths.py','research_singleton_free_20261108/full_additional.py',
        'research_identity_assignment_run_20261108/full_additional.py','research_identity_assignment_run_20261108/full_statistics.py',
        'research_identity_assignment_run_20261108/full_evaluate.py','research_identity_assignment_run_20261108/full_planner.py',
        'research_identity_assignment_run_20261108/full_run.py','research_identity_assignment_run_20261108/full_retina.py',
        'research_identity_assignment_run_20261108/full_precision_io.py','tracking/classical_motion.py','tracking/flyvis_activity.py','evaluation/hota_eval.py']:
        sources[name]=sha(ROOT/name)
    for path in [FULL/'protocol.json',RUN/'selection.json',RUN/'comparison_report.json',RUN/'independent_audit.json',BASE/'full_precision/independent_audit.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,deadline=p['deadline']),new=True)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Независимый аудит проверки собственного шаблона `singleton_own_template_v1_20261108` пройден:2156пространственныхоценок,13937строкисходов,11088матриц и10уникальныхпарныхинтервалов (`runs/singleton_verify_20261108/independent_audit.json`). До26наfit20выбранFlyVis: порог0,25; на641запросе45правильных/6чужих против41/6исходных, deltaJ+4,95%интервална100запросов[0,142653;1,284404]; строгий предварительный гейт пройден (`comparison_report.json`). Изображение тоже дало45/6, но три новых правильныхUIDобщие, ещё по одному различаются; совпадение сумм не означает равенство методов. Дополнительный вкладFlyVisне установлен.\n\n*Постановка полного пилота до исходов.* Начат тот же20val-когортный причинный полный проход с10вариантами ификсированными11первичными сравнениями. Выбор системы, все пороги, геометрия, альфа0,2, модель, детекции и нормализации заморожены. Для закрытия первой проблемы до полного прогона дополнительно уточнено: помимоHOTAи прежних ограничений вреда должны увеличиться известные правильные возвраты и возвраты, сохранившие правильную цель на двух следующих доступных наблюдениях. Неизвестные/цензурированные случаи считаются отдельно. Готовые исходные метрики, совпавшие карты и полные одинаковые траектории переиспользуются; нулевой и выключенный шлюз проверяются на полном видео до вмешательства. Источники `runs/singleton_verify_20261108/full/protocol.json`, `full_phase.json`. Остальные три проблемы отложены; полного выводаHOTAдляновойверсии ещё нет.\n\n"
    write(RUN/'full_registration_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    with (FULL/'manager.log').open('x',encoding='utf-8') as out,(FULL/'manager.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_20261108/manager.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_verify_full_20261108/manager.py')
    write(FULL/'manager_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(FULL/'deadline_registered.json',dict(passed=True,root=root,deadline=p['deadline']),new=True)


if __name__=='__main__':main()
