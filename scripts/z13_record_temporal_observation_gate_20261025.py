"""Independently audit temporal metadata and record only the preparation result."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_temporal_observation_gate_20261025 import RUN,IDENT,journal
from scripts.z13_continue import paired
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_margin_rule_audit_20261025 import revise


def main():
    assert not (RUN/'finish_recorded.json').exists()
    p=read(RUN/'protocol.json');r=read(RUN/'report.json');end=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert end['exit_code']==0 and end['worker_ended'] and end['all_owned_ended'] and not end['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    uid=set();new={};old={};pairs=0;first_frame_refs=0
    for name,digest in r['manifest'].items():
        assert sha(ROOT/name)==digest
        d=read(ROOT/name);v=d['rows'][0]['video'];entries=read(ROOT/f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries']
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:data={k:z[k] for k in ('ep_offset','ep_tid','fr_frame','fr_crop')}
        numerator=old_num=denominator=0
        for row in d['rows']:
            u=row['uid'];assert u not in uid;uid.add(u);e=row['episode_local'];en=entries[str(u)];index=row['current_array_index']
            assert int(data['ep_offset'][e])+4<=index<int(data['ep_offset'][e+1]) and int(data['fr_frame'][index])==row['current_frame']
            assert int(data['ep_tid'][e])==row['track_id'] and row['current_choices']==en['choices']
            ref=row['previous']
            if ref is not None:
                k=ref['frame_array_index'];assert int(data['ep_offset'][e])<=k<index
                assert int(data['fr_frame'][k])==ref['frame']<row['current_frame'] and data['fr_crop'][k].tolist()==ref['crop']
                first_frame_refs+=k==int(data['ep_offset'][e])
            eligible=ref is not None and len(en['choices'])>1;assert eligible==row['eligible'];pairs+=eligible
            known=en['kind']>=0;good=bool(known and any(en['good'][1:]))
            numerator+=int(good and eligible);old_num+=int(good and row['old_pair_eligible']);denominator+=known
        assert len(d['rows'])==len(entries) and (numerator,old_num,denominator)==(d['oracle_helpful'],d['old_oracle_helpful'],d['known'])
        if v in p['evaluate26']:new[v]=numerator/denominator;old[v]=old_num/denominator
    zeros={v:0. for v in new};assert len(new)==26
    assert paired(new,zeros)==r['expanded_oracle'] and paired(old,zeros)==r['old_oracle'] and paired(new,old)==r['reserve_increment']
    assert len(uid)==37296 and pairs==r['supported_pairs']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all_saved_rows_bindings_prior_frames_and_oracle_intervals_verified=True,rows=len(uid),supported_pairs=pairs,first_stored_frame_references=first_frame_refs,all_owned_ended=True,report_sha256=sha(RUN/'report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(obtained='runs/z13_temporal_observation_gate_20261025/report.json: supported_pairs='+str(pairs)+'из37296вместо14252; expanded_oracle='+str(r['expanded_oracle'])+'; reserve_increment='+str(r['reserve_increment'])+'. independent_audit.json проверил строки/прошедшие кадры/метки/интервалы.88,281сметаданных,0моделей.',
        interpretation='Подготовка более ранних наблюдений расширила доступные пары до59,23%. Идеальный запас на26повторныхвидео вырос с2,069до5,142п.п.; это доступность действий при знании исхода, не качество FlyVis и не HOTA. Старый запас точно воспроизведён. Истинная личность не использовалась для выбора предыдущего наблюдения; оно не обязано содержать ту же истинную цель.',
        decision='Допущена только отдельная подготовка равных временных признаков D/C/R1 с проверкой текущих центров/баллов и стоимости. Обучение временного узла ещё не выполнено и автоматически не разрешается этим числом; использовать действующее авторское разрешение и отдельный протокол. Пространственный модуль временно отложен, не возобновляется этим опытом.',
        cannot_claim='Нельзя заявлять прирост модели на3,07п.п., доказанную причину прошлых ошибок, готовую память личности, равные уже извлечённые признаки D/C/R1, улучшениеHOTA или независимый тест. Есть только проверенные ссылки на прошлые наблюдения/области; следующее извлечение должно сохранить текущую историю и пройти прежние допуски.')
    for k,v in fields.items():revise(card,k,v,'Независимо проверен полный результат подготовки130видео без обучения/GPU.')
    card.update(state='completed_positive' if r['resource_admission'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Проверка подготовки данных пройдена; качество модели не измерялось.')
    for x in card['acceptance']:x.update(met=True,evidence='runs/z13_temporal_observation_gate_20261025/independent_audit.json;report.json')
    card['evidence'].append(dict(kind='file',ref='runs/z13_temporal_observation_gate_20261025/independent_audit.json',claim='Проверенная готовность ссылок на прошлые наблюдения; не эффективность модели'))
    write(ROOT/'experiments/project_history.json',cat)
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,preparation_only=True,model_trained=False,hota_measured=False))
    write(RUN/'confirmed_review.json',dict(eligible_new_model_results=0,reason='Input availability and oracle reserve only, no learned model.'))
    journal('итог подготовки временных наблюдений','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA/проценткстенду/доляреализованногопотолканеизмерены. Исходные пороги/данные неизменны. Применён OpenAI Docs только для уточнения существующего наблюдателя:2листа, временно отложенный модуль; интервал120минут и срок прежние.')


if __name__=='__main__':main()
