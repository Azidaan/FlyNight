"""Verify and record the complete refusal-cause investigation, retaining history."""
from pathlib import Path
from collections import Counter
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_refusal_causes_20261025 import RUN, OLD, IDENT, read, write, sha, now, journal


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_refusal_trace_20261025 import mechanism
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (RUN/'finish_recorded.json').exists()
    ps=WindowsProcesses()
    for stage in (RUN,RUN/'trace'):
        p,e=read(stage/'protocol.json'),read(stage/'controller_result.json')
        assert e['exit_code']==0 and e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
        verify_sources(p)
    first=read(RUN/'report.json');second=read(RUN/'trace/report.json')
    assert first['complete'] and second['complete'] and second['semantic_rows_exact']
    assert sha(RUN/'rows.json')==first['rows_sha256']
    rows=read(RUN/'rows.json');group={a:[r for r in rows if r['arm']==a] for a in first['arms']}
    for arm,armrows in group.items():
        assert len(armrows)==8326 and len({r['uid'] for r in armrows})==8326
        counts=Counter()
        for row in armrows:
            side='selected' if row['selected'] else 'refused'
            counts[side+'_unknown' if row['outcome']=='unknown' else side+'_known']+=1
            if row['outcome']!='unknown':counts[side+'_'+row['outcome']]+=1
        assert dict(counts)==first['arms'][arm]['known_unknown_and_outcomes']
    traces={}
    for video,digest in second['video_sha256'].items():
        path=RUN/'trace/videos'/(video+'.json');assert sha(path)==digest
        job=read(path);assert job['checks']['raw_replay_exact'] and job['checks']['all_predictions_and_identities_bound']
        old={r['uid']:r for r in read(OLD/'videos'/(video+'.json'))['rows']}
        for row in job['rows']:
            assert {k:v for k,v in row.items() if k!='trace'}==old[row['uid']]
            traces[row['uid']]=row
    assert set(traces)==set(read(RUN/'trace/protocol.json')['wanted_uids'])
    reason_counts={key:dict(Counter(x['reason'] for x in second[key])) for key in ('selected','missed_beneficial_top1','old_center_useful_harmful')}
    for key in reason_counts:
        for row in second[key]:
            source=traces[row['uid']];rank=row.get('rank',0)
            assert row['reason']==mechanism(source,'D',rank)['reason']
    d=first['arms']['D'];parts=d['failure_partition'];out=d['known_unknown_and_outcomes']
    selected_known=out['selected_known'];helpful=out.get('selected_beneficial',0);harmful=out.get('selected_harmful',0)
    missed=out.get('refused_beneficial',0)
    no_proposal=sum(v for k,v in parts.items() if k.startswith('no_current_candidate_can_help;'))
    other=sum(v for k,v in parts.items() if k.startswith('only_other_candidate_can_help;'))
    cost=[]
    for row in second['old_center_useful_harmful']:
        tr=traces[row['uid']]['trace'];ids=row['correct_high_detection_indices'];rank=row['rank']
        if ids:
            before=min(tr['baseline_cost_row'][j] for j in ids);after=min(tr['candidate_cost_rows']['D'][rank][j] for j in ids)
            cost.append(dict(uid=row['uid'],rank=rank,correct_cost_before=before,correct_cost_after=after,increased=after>before,reason=row['reason']))
    diagnosis=dict(at=now().isoformat(),all_sources_verified=True,all_26_replays_exact=True,all_semantic_rows_exact=True,
        D_selected_known=selected_known,D_selected_beneficial=helpful,D_selected_harmful=harmful,
        D_selected_unknown=out['selected_unknown'],D_missed_beneficial_top1=missed,
        D_no_current_candidate_can_help=no_proposal,D_only_other_candidate_can_help=other,
        reason_counts=reason_counts,old_center_harm_cost_changes=cost,
        first_report_sha256=sha(RUN/'report.json'),trace_report_sha256=sha(RUN/'trace/report.json'),
        scientific_status='Descriptive one-step failure decomposition, not a new FlyVis efficacy result or a whole-tracker causal attribution.')
    write(RUN/'diagnosis.json',diagnosis)
    source='runs/z13_refusal_causes_20261025/report.json';trace='runs/z13_refusal_causes_20261025/trace/report.json'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='Фиксированный отказ центрированнойD пропускает все полезныеtop1в этой проверке;проверенные выбранные действия нейтральны.',
        obtained=source+f' → arms/D:из84выбранных{selected_known}с известным исходом,{out["selected_unknown"]}с неизвестным;полезных{helpful},вредных{harmful},нейтральных{out.get("selected_neutral",0)}. Все{missed}полезныхtop1отвергнуты;ещё'+str(other)+'ситуаций имеют полезныйинойкандидат,а'+str(no_proposal)+'ошибочныхисходныхназначенийнеисправляетниодинизтрёх. '+trace+' → reason counts: '+json.dumps(reason_counts,ensure_ascii=False),
        interpretation='Управление запуском и отказ различны:трекер вызываетмухутолькопослепотери,центрированнаяDразрешаетзаменуцентрапоотносительновысокомупику,прежняяDиспользуетголовусметкойточностицентра. Ни одноизэтихправилсамопосебенепредсказываетвыигрышполногоназначенияотносительноисходноготрекера. Декомпозициядоказываетнесоответствиепровереннойполитикиоднокадровойпользе,нонеустанавливаетпричинупредставленийиливсейHOTA.',
        decision='Не подбирать порог по исходам26/20val. Следующийузелдляправки — считываниеотносительнойпользысучётомисходногопрогнозаиназначения,присохраненииобразцаиконкурирующихвариантов;контрольбезмухиобязателен. Сначала проверитьнаготовыхкартах,сохраниласьлитребуемаяинформация,неповторятьпрежнийпроваленныйпилотобученияпометкампоследствий. Дополнительныеизмерениязаписыватьотдельнымпротоколом,невыдаватьновуюцельзаужеисправленнуюмодель.')
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:
            card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Завершён и проверен разбор существующей политики и точных назначений.'))
        card[key]=value
    card.update(state='inconclusive',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True,evidence=source+';'+trace+';diagnosis.json')
    card['evidence'] += [dict(kind='file',ref=source,claim='Фиксированный отбор,пропуски,кандидаты и прежняя метрика'),
        dict(kind='file',ref=trace,claim='Полные назначения,конкуренты и точное повторение26исходныхвыходов'),
        dict(kind='file',ref='runs/z13_refusal_causes_20261025/diagnosis.json',claim='Независимая проверка счётчиков и механизмов')]
    write(cp,cat)
    journal('итог:почему высокий пик не означает полезного вмешательства',
        '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+
        '\n\nСтараянаходка846→37/17/792относиласьковсемтрёмальтернативам,некакфактическиприменённымдействиям. '
        'Текущийразборфиксированногопорогапоказывает0/64полезныхсредиизвестных,20неизвестныхотдельно;экстраполяциябезопасностизапрещена. '
        'HOTAнепересчитывалась,проценткbaselineидоляпотолканеопределены. Нового принятогоузла нет,архитектурнаявеханеизменилась;графобновляетсякакитогдиагностики.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),first_report_sha256=sha(RUN/'report.json'),trace_report_sha256=sha(RUN/'trace/report.json'),diagnosis_sha256=sha(RUN/'diagnosis.json'),all_owned_ended=True))
    print(json.dumps(diagnosis,ensure_ascii=False))


if __name__=='__main__':
    record()
