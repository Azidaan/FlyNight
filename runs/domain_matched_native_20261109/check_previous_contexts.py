"""Read ready counterfactuals on the six prior direct labels; no recomputation."""
from evaluation_common import *
assert not (OUT/'C34_complete.json').exists() and not (OUT/'rollouts/C34').exists()
old=read(OLD/'id_fate.json');delivery=read(OUT/'delivery_report.json')
by={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in delivery['records']}
rows=[]
for r in old['records']:
    if r['arm']!='C34' or not r['scored'] or r['kind']=='unknown':continue
    q=by[(r['video'],r['frame'],r['old_id'],r['raw_id'])]
    rows.append(dict(video=r['video'],frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],kind=r['kind'],
        old_head_accept=r['head_accept'],old_global_selected=r['global_selected'],old_public_ID=r['old_id_emitted_on_candidate'],
        new_head_score=q['new_logit'],new_head_accept=q['accepted']))
assert len(rows)==6 and sum(r['kind']=='own' for r in rows)==5
false_refusals=[r for r in rows if r['kind']=='own' and not r['old_head_accept']]
assert len(false_refusals)==2
write('previous_contexts_check.json',dict(passed=True,recorded=datetime.now(MSK).isoformat(),records=rows,
    own_false_refusals_repaired=sum(r['new_head_accept'] for r in false_refusals),decision_changes=sum(r['old_head_accept']!=r['new_head_accept'] for r in rows),
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OLD/'id_fate.json',OUT/'delivery_report.json',OUT/'deployment.json']},
    no_scoring_or_fit_repeated=True,new_C34_native_not_started=True,
    prediction='On the exact same old six histories the new head keeps every sign, including both own refusals and the foreign permission. Improved training-fold admission therefore does not itself fix those old cases. Native histories may diverge; compare future runs through raw reference anchors, never equality of numeric IDs.',
    decision='Continue the registered full five-arm pilot without altering thresholds,heads,videos or success criteria. No new final-test or adoption claim.',goal_achieved=False))
body='**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Новая голова сохраняет решения на прежних шести подтверждённых контекстах\n\n*Получено.* Прочитаны готовые результаты проверки доставки без повторного вычисления баллов: на прежних 5 своих и 1 чужой истории новая голова не изменила ни одного решения. Оба своих отказа сохранились; чужая связь также разрешена. Источник runs/domain_matched_native_20261109/previous_contexts_check.json → records,decision_changes,own_false_refusals_repaired; исходные метки runs/full_template_native_20261109/id_fate.json, готовые баллы delivery_report.json. Локальная прибавка на обучающих межвидеовых складках не означает исправление этих прежних случаев.\n\n*Решение.* Полный зарегистрированный пилот продолжается без изменения моделей, порогов и выборки. Новая ветвь C34 ещё не запускалась в момент записи; предсказание относится к старым историям и не выдаётся за фактическую судьбу новых ID. После расхождения историй сравнивать через исходные детекции, а не равенство номеров.\n\n'
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
temp=path.with_suffix('.previous_contexts.pending');assert not temp.exists();temp.write_bytes(before.replace(marker,marker+nl+body.replace('\n',nl.decode()).encode(),1));assert path.read_bytes()==before
for _ in range(30):
    try:temp.replace(path);break
    except PermissionError:time.sleep(.1)
else:raise PermissionError(str(path))
print('PREVIOUS_CONTEXT_SIGNS_UNCHANGED',len(rows),'own refusals repaired',sum(r['new_head_accept'] for r in false_refusals),flush=True)
