"""Verify primary rows and record the complete target-weight pilot."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_search_conditioned_20261023 import RUN,OLD,IDENT,ARMS,read,write,sha,now,journal,prior
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    assert not (RUN/'finish_recorded.json').exists()
    end=read(RUN/'controller_result.json');control=read(RUN/'controller.json')
    assert end['exit_code']==0 and end['worker_ended'] and end['all_owned_ended'] and not end['hard_timeout']
    ps=WindowsProcesses();assert ps.creation(control['pid'])!=control['birth'];assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    prior.verify_sources(read(RUN/'protocol.json'));assert all(sha(ROOT/n)==h for n,h in read(RUN/'cache_manifest.json').items())
    rep=read(RUN/'reproduction.json');assert len(rep)==6 and all(v['rows']==8326 and v['centers_outcomes_equal'] and v['max_abs_score']<=1e-5 for v in rep.values())
    assert read(RUN/'timing.json')['old_loss_max_abs']<=1e-6
    rows={};seedpv={};models={};counts=set()
    for arm in ARMS:
        for access in ('search','template'):
            for mode in ('activity','geometry'):
                key=arm+'_'+access+'_'+mode;seedpv[key]={}
                for seed in (0,1,2):
                    sd=RUN/arm/access/mode/f'seed{seed}';assert (sd/'done.json').exists();tr=read(sd/'training.json');old=read(OLD/arm/mode/f'seed{seed}/training.json')
                    assert tr['steps']==old['steps']==1150 and tr['presented_rows']==old['presented_rows']==145950 and tr['pos_weight']==old['pos_weight']
                    assert sha(sd/'model.pt')==tr['model_sha256'];models[sd.relative_to(ROOT).as_posix()]=tr['model_sha256'];counts.add(tr['parameters'])
                    seedpv[key][str(seed)]=per_video_useful(read(sd/'rows.json'))
                rows[key]=read(RUN/arm/access/mode/'ensemble_rows.json')
        for mode in ('activity','geometry'):rows[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for mode in ('top1','kalman'):rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    assert len(models)==36 and len(counts)==1
    report=read(RUN/'report.json');pv={k:per_video_useful(v) for k,v in rows.items()};assert pv==report['per_video']
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    assert seedpv==read(RUN/'seed_metrics.json')['per_video']
    for arm,values in report['seeds'].items():
        for seed,value in values.items():assert paired(seedpv[arm+'_search_activity'][seed],seedpv[arm+'_template_activity'][seed])==value
    c=report['paired']['D_search_activity-D_template_activity'];criteria=[True,c['mean_per_video']>=.02 and c['ci95'][0]>0,all(report['paired']['D_search_activity-'+b]['ci95'][0]>0 for b in ('D_old_activity','D_old_geometry','R1_search_activity','D_search_geometry'))]
    assert all(criteria)==report['primary_pass'];state='completed_positive' if all(criteria) else 'completed_negative'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source='runs/z13_search_conditioned_20261023/report.json'
    obtained=source+', paired: '+json.dumps(report['paired'],ensure_ascii=False)+'. Все36моделей завершены,параметров на модель '+str(next(iter(counts)))+';1150шагов и145950предъявлений;18старыхмоделей и32старыхшага воспроизведены.'
    interpretation=('Доступ поиска прошёл все заданные поисковые критерии при равной ёмкости. Перенос, отказ и HOTA ещё не подтверждены.' if all(criteria) else 'Эта адаптация предсказания модели цели не прошла все заданные критерии. Это не отрицание механизма ToMP или всей линии FlyVis; наш средний образец и3пика существенно беднее исходной модели.')
    decision=('Зафиксировать версию для отдельной проверки переноса и применения, не менять по dev.' if all(criteria) else 'Не заменять рабочий считыватель этой версией; не выбирать лучший сид или подбирать по train26. Следующая правка должна иметь иной механизм и измеримую мишень.')
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=card[field],reason='Полный проверенный исход предсказателя весов.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence=source+'; reproduction.json; timing.json; controller_result.json')
    card['metrics']=[dict(key='useful_center_fraction',value=report['summary']['D_search_activity']['useful_mean_video'],baseline=report['summary']['D_template_activity']['useful_mean_video'],subset='train26 search-conditioned D',source=source,locator='summary.D_search_activity.useful_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_search_activity-D_template_activity','D_search_activity-D_old_activity','D_search_activity-D_old_geometry','D_search_activity-R1_search_activity','C_search_activity-C_template_activity','C_search_activity-R1_search_activity'):
        if report['paired'][key]['ci95'][0]>0:card['confirmed_results'].append(dict(id=IDENT+'_'+key.replace('-','_vs_'),title='Предсказание модели цели: '+key,category='representation' if 'R1' in key else 'readout',metric='useful_center_fraction',unit='fraction',comparison=key,subset='Повторно использованные26train,известные присутствующие цели',conditions='104train,3сида,1150шагов145950предъявлений; одинаковая архитектура, маска доступа поиска в предсказатель весов;парный бутстрэп5000сид1.',limitations='Частный результат не отменяет остальные критерии. Не полная ToMP; shared encoder also changes enriched features through the changed exemplar path. Нет HOTA/безопасного отказа.',evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'].extend(dict(kind='file',ref='runs/z13_search_conditioned_20261023/'+n,claim=c) for n,c in [('report.json','Все парные исходы'),('reproduction.json','Воспроизведение старого чтения'),('controller_result.json','Своё дерево завершено')]);write(cp,cat)
    baseline=report['summary']['D_template_activity']['useful_mean_video'];relative=100*c['mean_per_video']/baseline if baseline else None
    journal('полный итог модели цели с текущим поиском','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+f'\n\nСводка: полезные центры D относительно равного template-контроля {relative}%. HOTA и доля её потолка не измерены. Маска влияет на путь образца внутри общего кодировщика и через него также на обогащённые признаки; это не раздельная оценка только чисел весов при неизменных признаках.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),state=state,primary_pass=all(criteria),report_sha256=sha(RUN/'report.json'),models=models,all_owned_ended=True,relative_percent=relative));print(json.dumps(dict(state=state,main=c,criteria=criteria)))


if __name__=='__main__':main()
