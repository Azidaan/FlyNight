"""Verify complete primary rows before recording the descriptor experiment."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_filter_descriptor_20261023 import RUN,OLD,IDENT,ARMS,read,write,sha,now,journal,prior
from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses


def ended(run):
    result=read(run/'controller_result.json');controller=read(run/'controller.json');ps=WindowsProcesses()
    assert result['exit_code']==0 and result['worker_ended'] and result['all_owned_ended'] and not result['hard_timeout']
    assert ps.creation(controller['pid'])!=controller['birth']
    assert all(ps.creation(int(pid))!=birth for pid,birth in result['owned'].items())


def main():
    assert not (RUN/'finish_recorded.json').exists();ended(CRUN);ended(RUN)
    prior.verify_sources(read(RUN/'protocol.json'));cache=read(CRUN/'report.json');assert cache['parity_pass'] and cache['observed_frames_only']
    assert len(cache['summary'])==9
    for item in cache['summary'].values():assert item['max_seed']<=.001 and item['max_center']<.5 and item['max_score']<.001
    assert all(sha(ROOT/n)==h for n,h in read(RUN/'cache_manifest.json').items())
    rep=read(RUN/'reproduction.json');assert len(rep)==6 and all(v['rows']==8326 and v['centers_outcomes_equal'] and v['max_abs_score']<=1e-5 for v in rep.values())
    assert read(RUN/'timing.json')['old_loss_max_abs']<=1e-6
    rows={};seedpv={};models={};counts=set()
    for arm in ARMS:
        for access in ('ridge','mean'):
            for mode in ('activity','geometry'):
                key=arm+'_'+access+'_'+mode;seedpv[key]={}
                for seed in (0,1,2):
                    sd=RUN/arm/access/mode/f'seed{seed}';assert (sd/'done.json').exists();tr=read(sd/'training.json');old=read(OLD/arm/mode/f'seed{seed}/training.json')
                    assert tr['steps']==old['steps']==1150 and tr['presented_rows']==old['presented_rows']==145950 and tr['pos_weight']==old['pos_weight']
                    assert sha(sd/'model.pt')==tr['model_sha256'];models[sd.relative_to(ROOT).as_posix()]=tr['model_sha256'];counts.add(tr['parameters'])
                    seedpv[key][str(seed)]=per_video_useful(read(sd/'rows.json'))
                rows[key]=read(RUN/arm/access/mode/'ensemble_rows.json')
        # Completely ablated descriptors must yield identical geometry models.
        for seed in (0,1,2):
            a=read(RUN/arm/'ridge/geometry'/f'seed{seed}/training.json');b=read(RUN/arm/'mean/geometry'/f'seed{seed}/training.json');assert a['loss']==b['loss']
        assert rows[arm+'_ridge_geometry']==rows[arm+'_mean_geometry']
        for mode in ('activity','geometry'):rows[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for mode in ('top1','kalman'):rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    assert len(models)==36 and len(counts)==1
    report=read(RUN/'report.json');pv={k:per_video_useful(v) for k,v in rows.items()};assert pv==report['per_video']
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    assert seedpv==read(RUN/'seed_metrics.json')['per_video']
    for arm,values in report['seeds'].items():
        for seed,value in values.items():assert paired(seedpv[arm+'_ridge_activity'][seed],seedpv[arm+'_mean_activity'][seed])==value
    main_comparison=report['paired']['D_ridge_activity-D_mean_activity']
    criteria=[True,main_comparison['mean_per_video']>=.02 and main_comparison['ci95'][0]>0,all(report['paired']['D_ridge_activity-'+b]['ci95'][0]>0 for b in ('D_old_activity','D_old_geometry','R1_ridge_activity','D_ridge_geometry'))]
    assert all(criteria)==report['primary_pass'];state='completed_positive' if all(criteria) else 'completed_negative'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source='runs/z13_filter_descriptor_20261023/report.json'
    obtained=source+', paired: '+json.dumps(report['paired'],ensure_ascii=False)+'. Воспроизведение наблюдаемых кадров: runs/z13_filter_descriptor_cache_20261023/report.json. Завершены36моделей,1150шагов145950предъявлений на модель;число параметров '+str(next(iter(counts)))+'. Старые18контролей/32шага воспроизведены, обе геометрические абляции совпадают.'
    interpretation=('Описание цели коэффициентами прошло заданные поисковые критерии при равной ёмкости. Нужна отдельная проверка переноса и применения.' if all(criteria) else 'Эта реализация описания цели коэффициентами не прошла заданные критерии. Причина недостатка пользы всей FlyVis этим опытом не установлена; фильтр уже участвовал в карте кандидатов.')
    decision=('Сохранить версию для следующего этапа переноса без настройки по dev.' if all(criteria) else 'Не заменять действующий считыватель; сохранять результат и выбирать следующий механизм по первичным измерениям. Не выбирать лучший сид или пороги на train26.')
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=card[field],reason='Полный проверенный исход описания цели.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence=source+'; reproduction.json; timing.json; controller_result.json; runs/z13_filter_descriptor_cache_20261023/report.json')
    card['metrics']=[dict(key='useful_center_fraction',value=report['summary']['D_ridge_activity']['useful_mean_video'],baseline=report['summary']['D_mean_activity']['useful_mean_video'],subset='train26 observed-filter D',source=source,locator='summary.D_ridge_activity.useful_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_ridge_activity-D_mean_activity','D_ridge_activity-D_old_activity','D_ridge_activity-D_old_geometry','D_ridge_activity-R1_ridge_activity','C_ridge_activity-C_mean_activity','C_ridge_activity-R1_ridge_activity'):
        if report['paired'][key]['ci95'][0]>0:card['confirmed_results'].append(dict(id=IDENT+'_'+key.replace('-','_vs_'),title='Коэффициенты модели цели: '+key,category='representation' if 'R1' in key else 'readout',metric='useful_center_fraction',unit='fraction',comparison=key,subset='Повторно использованные26train,известные присутствующие цели',conditions='104train,3сида,1150шагов145950предъявлений; одинаковые отдельные кодировщики,коэффициенты наблюдаемого фильтра вместо среднего;парный бутстрэп5000сид1.',limitations='Частный результат не отменяет остальные критерии. Не полная DiMP; карта уже использовала этот фильтр. Нет HOTA и безопасного отказа.',evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'].extend(dict(kind='file',ref='runs/z13_filter_descriptor_20261023/'+n,claim=c) for n,c in [('report.json','Все парные исходы'),('reproduction.json','Воспроизведение старого чтения'),('controller_result.json','Своё дерево завершено')]);write(cp,cat)
    baseline=report['summary']['D_mean_activity']['useful_mean_video'];relative=100*main_comparison['mean_per_video']/baseline if baseline else None
    journal('полный итог описания цели коэффициентами','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+f'\n\nСводка: полезные центры D относительно равного контроля среднего образца {relative}%. HOTA и доля её потолка не измерены.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),state=state,primary_pass=all(criteria),report_sha256=sha(RUN/'report.json'),models=models,all_owned_ended=True,relative_percent=relative));print(json.dumps(dict(state=state,main=main_comparison,criteria=criteria)))


if __name__=='__main__':main()
