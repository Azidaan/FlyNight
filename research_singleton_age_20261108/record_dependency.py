"""Static causal reach audit of the actual fixed readout; no neural inference."""
from collections import deque
from research_singleton_age_20261108.common import *


def main():
    exposure=read(ROOT/'runs/flyvis_separate_inputs_20261107/activation_exposure.json')
    for name,h in exposure['sources_sha256'].items():assert sha(ROOT/name)==h
    protocol=read(OWN/'protocol.json');assert protocol['controls']['FLYVIS']=='Same fixed standardized8T4/T5 spatial maps.'
    spec_path=ROOT/'tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json';spec=read(spec_path)
    distance={t:0 for t in spec['input_units']};queue=deque(distance)
    while queue:
        node=queue.popleft()
        for edge in spec['edges']:
            if edge['src']==node and edge['tar'] not in distance:distance[edge['tar']]=distance[node]+1;queue.append(edge['tar'])
    bound={t:distance[t]+1 for t in spec['output_units'] if t.startswith(('T4','T5'))}
    assert bound==exposure['min_explicit_updates'] and min(bound.values())==3
    # Same explicitly inspected build_arm(C): make_motion_estimator(30Hz,.02s).
    # integration_plan uses ceil((1/30)/.02)=2 updates. The patched velocity
    # and Network._next_state use old-state source activity, then Euler update.
    steps=2;assert exposure['observed_internal_steps_per_frame']==steps
    report=read(OWN/'comparison_report.json');outcomes=read(OWN/'outcomes/evaluate26_FLYVIS.json');improved=[r for r in outcomes['rows'] if r['primary_known'] and r['baseline']['kind']=='none' and r['outcome']['kind']=='correct']
    assert {r['uid'] for r in improved}=={30818,33601,33988,36902}
    inventory=read(OWN/'inventory_recovery1.json');exports=read(BASE/'export_complete.json')['videos'];witnesses=[];sources={}
    for result in improved:
        entry=exports[result['video']];path=ROOT/entry['path'];assert sha(path)==entry['sha256'];sources[path.relative_to(ROOT).as_posix()]=entry['sha256']
        snapshot=read(path)['snapshots'][str(result['frame'])];row=next(r for r in snapshot['rows'] if r['tid']==result['tid']);gap=result['frame']-row['history'][-1]['frame']
        event=next(e for e in inventory['events'] if e['video']==result['video'] and e['frame']==result['frame'] and e['tid']==result['tid'] and e['column']==result['first_high_index'])
        sequence=inventory['requests'][event['current_key']]['sequence'];reference=inventory['requests'][event['reference_key']]['sequence'];search=sequence[len(reference):]
        assert sequence[:len(reference)]==reference and len(search)==1 and search[-1][0]==result['frame'] and gap==2
        witnesses.append(dict(uid=result['uid'],video=result['video'],frame=result['frame'],last_observation_frame=row['history'][-1]['frame'],gap=gap,search_inputs=len(search),reference_sequence=reference,current_sequence=sequence,
            current_frame_can_affect_T4T5_this_step=False,basis='Strict old-state dependency lower bound, not a new measured intervention.'))
    names=['research_singleton_age_20261108/record_dependency.py','tracking/z13_readout.py','tracking/foveated_eyes.py','tracking/flyvis_motion.py','scripts/z13_target_model.py',
        'scripts/run_bdd_video_rate_experiment.py','tools/flyvis-runtime/flyvis/network/network.py','tools/flyvis-runtime/flyvis/network/dynamics.py',
        'tools/flyvis-runtime/flyvis/connectome/connectome.py','tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json',
        'research_singleton_free_20261108/full_features.py','research_identity_assignment_run_20261108/full_planner.py',
        'runs/flyvis_separate_inputs_20261107/activation_exposure.json','runs/singleton_verify_20261108/inventory_recovery1.json','runs/singleton_verify_20261108/comparison_report.json','runs/singleton_verify_20261108/outcomes/evaluate26_FLYVIS.json']
    sources.update({n:sha(ROOT/n) for n in names})
    value=dict(passed=True,updated=datetime.now(MSK).isoformat(),kind='Static readout dependency audit plus existing exposure evidence; no repeated inference.',sources=sources,
        current_frame_updates=steps,minimum_explicit_updates=bound,known_added_correct_witnesses=witnesses,
        measured_previous_exposure=dict(pairs=exposure['pairs'],changed_T4T5_cases=exposure['conditions']['same_history']['changed_T4T5_map_cases'],source='runs/flyvis_separate_inputs_20261107/activation_exposure.json'),
        conclusion='The selected T4/T5 current map cannot depend on the newly supplied current input within this two-update readout. For all4new known correct conditional returns only1search input was supplied; thus these4gains cannot establish matching the current candidate visual appearance viaT4/T5. Past appearance, coordinate correspondence and prediction remain possible sources.',
        scope='Concrete frozen full FlyVis module/readout configuration, not a general claim about the FlyVis core. This does not study direction-reversal latency and does not invalidate potential past-signal benefits.',
        alternative='The existing correction benefits from a recent predictable past spatial pattern and geometry; recency alone may explain part, but is not proven sufficient.',
        next='Finish the unchanged full tracker, independent audit and registered age-only control. Any later current-appearance readout must first demonstrate that its input reaches its selected signal; do not increase simulation time silently.')
    write(RUN/'current_input_dependency.json',value,new=True)
    entry=(f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Уточнена граница объяснения первой проблемы по коду и готовым данным (`runs/singleton_age_20261108/current_input_dependency.json`). В текущем считывании T4/T5 новый текущий вход получает два одновременных явных обновления, а по фактической схеме связей требуется не менее трёх обновлений до T4 и четырёх до T5. В прежнем контроле одного текущего входа карты не изменились во всех 172 случаях (`runs/flyvis_separate_inputs_20261107/activation_exposure.json`). Все четыре новых известных правильных условных возврата имеют пропуск два кадра и только один поисковый вход (UID 30818, 33601, 33988, 36902); по этой схеме текущие пиксели кандидата ещё не могут повлиять на выбранные T4/T5. Это вывод из проверенных зависимостей, а не новая нейронная симуляция. Нельзя объяснять эти четыре возврата проверкой текущей внешности через T4/T5; остаются прошлый пространственный образ, геометрическое соответствие и прогноз. Их полезность не опровергнута. Текущий полный пилот и контроль давности продолжаются без изменения условий. Отдельная задержка после разворота не исследуется; проблемы 2–4 отложены.\n\n")
    assert not (RUN/'comparison_phase.json').exists(),'Avoid changing chronology during the queued catalog check.'
    write(RUN/'dependency_journal.json',dict(entry=entry),new=True);plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')


if __name__=='__main__':main()
