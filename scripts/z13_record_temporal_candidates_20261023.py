"""Record the temporal pilot only after independent primary-artifact checks."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_temporal_candidates_20261023 import RUN,OLD,IDENT,ARMS,MODES,read,write,sha,now,journal
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    assert not (RUN/'finish_recorded.json').exists()
    p=read(RUN/'protocol.json');end=read(RUN/'controller_result.json');controller=read(RUN/'controller.json')
    assert end['exit_code']==0 and end['all_owned_ended'] and end['worker_ended'] and not end['hard_timeout']
    ps=WindowsProcesses();assert ps.creation(controller['pid'])!=controller['birth']
    assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    verify_sources(p);manifest=read(RUN/'cache_manifest.json');assert all(sha(ROOT/n)==h for n,h in manifest.items())
    reproduction=read(RUN/'reproduction.json');assert len(reproduction)==6
    assert all(v['rows']==8326 and v['old_max_score_delta']<=1e-5 and v['zero_residual_both_contexts_exact'] for v in reproduction.values())
    rows={};models={};parameters=set()
    for arm in ARMS:
        for mode in MODES:
            for seed in (0,1,2):
                folder=RUN/arm/mode/f'seed{seed}';assert (folder/'done.json').exists()
                tr=read(folder/'training.json');old=read(OLD/arm/('activity' if mode.endswith('activity') else 'geometry')/f'seed{seed}/training.json')
                assert tr['steps']==1150 and tr['presented_rows']==145950 and tr['pos_weight']==old['pos_weight']
                assert tr['initial_base_sha256']==old['model_sha256'] and tr['model_sha256']==sha(folder/'model.pt')
                models[folder.relative_to(ROOT).as_posix()]=tr['model_sha256'];parameters.add(tr['parameters'])
            rows[arm+'_'+mode]=read(RUN/arm/mode/'ensemble_rows.json')
        for mode in ('activity','geometry'):rows[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for mode in ('top1','kalman'):rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    assert len(models)==36 and len(parameters)==1
    report=read(RUN/'report.json');pv={k:per_video_useful(v) for k,v in rows.items()};assert pv==report['per_video']
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    c=report['paired']['D_previous_activity-D_current_activity']
    criteria=[True,c['mean_per_video']>=.02 and c['ci95'][0]>0,
        all(report['paired']['D_previous_activity-'+b]['ci95'][0]>0 for b in ('D_old_activity','D_previous_geometry','R1_previous_activity','D_old_geometry'))]
    assert all(criteria)==report['primary_pass']
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    state='completed_positive' if all(criteria) else 'completed_negative'
    source='runs/z13_temporal_candidates_20261023/report.json'
    obtained=source+', paired: '+json.dumps(report['paired'],ensure_ascii=False)+'. Все36 моделей и18 исходных контролей проверены; одинаковое число параметров,1150дополнительных обновлений и145950предъявлений; training.json каждой модели и reproduction.json.'
    interpretation=('Фиксированный временной узел прошёл все поисковые критерии на train26; это ещё не подтверждение пользы HOTA.' if all(criteria) else 'Этот временной узел не прошёл все заранее заданные критерии. Исход не доказывает бесполезность памяти или невозможность идентификации по FlyVis; проверено одно мягкое сопоставление с предыдущим набором.')
    decision=('Следующая ступень — заранее зафиксированная проверка применения с контролем без мухи; не менять модель по dev.' if all(criteria) else 'Эту версию не переносить в стенд. Следующий узел выбирается по полным контрастам, без перебора порога или сида на train26; прежний вариант сохраняется.')
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=card[field],reason='Полный проверенный исход; постановка сохранена.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence=source+'; reproduction.json; controller_result.json; */*/seed*/training.json')
    card['metrics']=[dict(key='useful_center_fraction',value=report['summary']['D_previous_activity']['useful_mean_video'],baseline=report['summary']['D_current_activity']['useful_mean_video'],subset='train26 D previous activity',source=source,locator='summary.D_previous_activity.useful_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_previous_activity-D_current_activity','D_previous_activity-D_old_activity','D_previous_activity-R1_previous_activity','C_previous_activity-C_current_activity','C_previous_activity-R1_previous_activity'):
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=IDENT+'_'+key.replace('-','_vs_'),title='Временной контекст: '+key,category='representation' if 'R1' in key else 'readout',metric='useful_center_fraction',unit='fraction',comparison=key,subset='26train, известные присутствующие цели',conditions='Те же104train,3сида,1150дополнительных шагов, прежние метки; контексты одной ёмкости. Бутстрэп5000 по видео,сид1.',limitations='Частный эффект не отменяет остальные критерии; повторно использованные26train, нет HOTA, идентичностей и калиброванного отказа.',evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'].extend(dict(kind='file',ref='runs/z13_temporal_candidates_20261023/'+name,claim=claim) for name,claim in [('report.json','Все парные исходы'),('reproduction.json','Исходные модели и нулевая добавка совпадают'),('controller_result.json','Полное завершение своего дерева')])
    write(cp,cat)
    journal('полный итог обучаемого временного контекста','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к стенду: HOTA не измерена; процент улучшения стенда и доля потолка HOTA неизвестны. Текущий контекст контролирует дополнительное обучение и ёмкость; R1 контролирует вход глаза. Положительные частные сравнения занесены отдельно в исходную карточку.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),state=state,primary_pass=all(criteria),report_sha256=sha(RUN/'report.json'),models=models,all_owned_ended=True))
    print(json.dumps(dict(state=state,main=c,criteria=criteria),ensure_ascii=False))


if __name__=='__main__':main()
