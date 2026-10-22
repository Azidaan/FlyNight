"""Record completed semantic sampling pilot, independently checking primary rows."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_semantic_sampling_20261023 import RUN, OLD, IDENT, ARMS, read, write, sha, now, journal, verify_sources
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    assert not (RUN/'finish_recorded.json').exists()
    control=ROOT/read(RUN/'execution.json')['controller_root'] if (RUN/'execution.json').exists() else RUN
    p=read(RUN/'protocol.json');end=read(control/'controller_result.json');controller=read(control/'controller.json')
    assert end['exit_code']==0 and end['worker_ended'] and end['all_owned_ended'] and not end['hard_timeout']
    ps=WindowsProcesses();assert ps.creation(controller['pid'])!=controller['birth']
    assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    verify_sources(p);rep=read(RUN/'reproduction.json');assert len(rep)==6 and all(v['all_candidates_centers_outcomes_equal'] and v['max_abs_score']<=1e-5 for v in rep.values())
    timing=read(RUN/'timing.json');assert timing['old_loss_max_abs']<=1e-6
    rows={};models={}
    for arm in ARMS:
        for mode in ('activity','geometry'):
            for seed in (0,1,2):
                sd=RUN/arm/mode/f'seed{seed}';assert (sd/'done.json').exists();tr=read(sd/'training.json');ref=read(OLD/arm/mode/f'seed{seed}/training.json')
                assert tr['steps']==ref['steps'] and tr['presented_rows']==ref['presented_rows'] and tr['pos_weight']==ref['pos_weight']
                assert sha(sd/'model.pt')==tr['model_sha256'];models[str(sd.relative_to(ROOT))]=tr['model_sha256']
            rows[arm+'_new_'+mode]=read(RUN/arm/mode/'ensemble_rows.json')
            rows[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for mode in ('top1','kalman'):rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    report=read(RUN/'report.json');pv={key:per_video_useful(rr) for key,rr in rows.items()}
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value,key
    assert pv==report['per_video']
    c=report['paired']['D_new_activity-D_old_activity']
    criteria=[True,c['mean_per_video']>=.02 and c['ci95'][0]>0,all(report['paired'][k]['ci95'][0]>0 for k in ('D_new_activity-D_old_geometry','D_new_activity-R1_new_activity'))]
    assert all(criteria)==report['primary_pass']
    assert read(RUN/'semantic_audit.json')==read(RUN/'preflight_target.json')
    assert all(sha(ROOT/n)==h for n,h in read(RUN/'cache_manifest.json').items())
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(v for v in cat['experiments'] if v['id']==IDENT)
    result='completed_positive' if report['primary_pass'] else 'completed_negative'
    obtained='runs/z13_semantic_sampling_20261023/report.json, paired → '+json.dumps({k:report['paired'][k] for k in ('D_new_activity-D_old_activity','D_new_activity-D_old_geometry','D_new_activity-R1_new_activity','D_new_activity-C_new_activity')},ensure_ascii=False)+'. semantic_audit.json → '+json.dumps({a:{k:v for k,v in read(RUN/'semantic_audit.json')['arms'][a].items() if k!='per_video'} for a in ARMS},ensure_ascii=False)+'. Все18 моделей завершены;1150шагов и145950предъявлений, прежние веса классов; воспроизведение18 контролей и32 старых шагов пройдено.'
    interpretation=('Новая частота примеров прошла функциональные критерии выбора на поисковом train26. Отказ и HOTA пока не измерены.' if report['primary_pass'] else 'Эта замена частоты примеров не прошла все заранее заданные критерии. Не доказана гипотеза, что увеличение частоты похожих соседей решает выбор. Это исход одной реализации, не ограничение всей FlyVis.')
    decision=('Перейти к заранее фиксированной проверке применения; не подбирать порог на dev.' if report['primary_pass'] else 'Не применять эту версию к стенду. Не подбирать долю трудных строк по train26. Дальнейшее действие требует иной проверяемой мишени; эта реализация не доказывает непригодность семантических отрицательных примеров вообще.')
    for field,new in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if card.get(field) and card[field]!=new:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=card[field],reason='Полный проверенный исход новой частоты примеров; исходная постановка сохранена.'))
        card[field]=new
    card.update(state=result,done=f'{now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence='runs/z13_semantic_sampling_20261023/report.json; reproduction.json; timing.json; semantic_audit.json')
    card['metrics']=[dict(key='useful_center_fraction',value=report['summary']['D_new_activity']['useful_mean_video'],baseline=report['summary']['D_old_activity']['useful_mean_video'],subset='train26; D new activity',source='runs/z13_semantic_sampling_20261023/report.json',locator='summary.D_new_activity.useful_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_new_activity-D_old_activity','C_new_activity-C_old_activity','D_new_activity-R1_new_activity','D_new_activity-C_new_activity'):
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=IDENT+'_'+key.replace('-','_vs_'),title='Новая частота примеров выбора: '+key,category='representation' if 'R1' in key else 'readout',metric='useful_center_fraction',unit='fraction',comparison=key,subset='Повторно использованные26train, известные присутствующие цели',conditions='104train; те же кэши,нормировка,архитектура,три сида,веса классов,1150шагов; общая для D/C/R1 половина строк с похожими соседями; прежние метки полезности; бутстрэп5000,сид1.',limitations='Частное сравнение не отменяет непройденных критериев; нет HOTA и безопасного отказа. Вклад анатомии отдельно не установлен.',evidence=dict(source='runs/z13_semantic_sampling_20261023/report.json',sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'].extend([dict(kind='file',ref='runs/z13_semantic_sampling_20261023/report.json',claim='Полный исход и все контроли'),dict(kind='file',ref='runs/z13_semantic_sampling_20261023/reproduction.json',claim='Воспроизведение старых18моделей'),dict(kind='file',ref=(control/'controller_result.json').relative_to(ROOT).as_posix(),claim='Полное завершение своего дерева')])
    write(catpath,cat)
    baseline=report['summary']['D_old_activity']['useful_mean_video'];relative=100*c['mean_per_video']/baseline
    reserve=read(ROOT/'runs/z13_candidate_reserve_20261021/report.json')['arms']['D']['oracle3_mean']-baseline
    fraction=c['mean_per_video']/reserve if reserve>0 else None
    journal('полный итог изменения частоты похожих соседей','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+f'\n\nСводка: полезные центры D → {relative:.5f}% относительно своего прежнего выборщика; доля остатка до оракульного выбора среди тех же3кандидатов {fraction}. Это диагностический предел с доступом к истинной разметке, не достижимый прогноз. HOTA не измерялась; процент к стенду и доля потолка HOTA неизвестны.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),state=result,primary_pass=report['primary_pass'],all_owned_ended=True,report_sha256=sha(RUN/'report.json'),models=models,relative_percent_vs_old=relative,fraction_of_oracle_remaining=fraction))
    print(json.dumps(dict(state=result,main=c,criteria=criteria),ensure_ascii=False))


if __name__=='__main__':main()
