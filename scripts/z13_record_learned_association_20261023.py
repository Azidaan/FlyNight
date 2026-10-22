"""Independently verify and record the learned identity association pilot."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_learned_association_20261023 import RUN,OLD,IDENT,ARMS,read,write,sha,now,journal
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    assert not (RUN/'finish_recorded.json').exists()
    p=read(RUN/'protocol.json');end=read(RUN/'controller_result.json');controller=read(RUN/'controller.json')
    assert end['exit_code']==0 and end['all_owned_ended'] and end['worker_ended'] and not end['hard_timeout']
    ps=WindowsProcesses();assert ps.creation(controller['pid'])!=controller['birth']
    assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    verify_sources(p);assert all(sha(ROOT/n)==h for n,h in read(RUN/'cache_manifest.json').items())
    reproduction=read(RUN/'reproduction.json');assert set(reproduction)==set(ARMS)
    assert all(v['rows']==8326 and v['all_fields_equal'] for v in reproduction.values())
    audit=read(RUN/'pair_audit.json');assert all(v['real_pairs']>0 and v['synthetic_source_frames']>0 for v in audit.values())
    rows={};models={};parameters=set();seedpv={}
    for arm in ARMS:
        for mode in ('activity','geometry'):
            seedpv[arm+'_'+mode]={}
            for seed in (0,1,2):
                sd=RUN/arm/mode/f'seed{seed}';assert (sd/'done.json').exists();tr=read(sd/'training.json')
                assert tr['steps']==1150 and tr['real_presentations']==73600 and tr['synthetic_presentations']==73600
                assert tr['model_sha256']==sha(sd/'model.pt');models[sd.relative_to(ROOT).as_posix()]=tr['model_sha256'];parameters.add(tr['parameters'])
                seedpv[arm+'_'+mode][str(seed)]=per_video_useful(read(sd/'rows.json'))
            rows[arm+'_'+mode]=read(RUN/arm/mode/'ensemble_rows.json')
            rows[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        rows[arm+'_fixed']=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/association_rows.json')
        for mode in ('top1','kalman'):rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    assert len(models)==18 and len(parameters)==1
    report=read(RUN/'report.json');pv={k:per_video_useful(v) for k,v in rows.items()};assert pv==report['per_video']
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(pv[a],pv[b])==value
    assert seedpv==read(RUN/'seed_metrics.json')['per_video']
    for arm,values in report['seeds'].items():
        for seed,value in values.items():assert paired(seedpv[arm+'_activity'][seed],pv[arm+'_fixed'])==value
    c=report['paired']['D_activity-D_fixed'];criteria=[True,c['mean_per_video']>=.02 and c['ci95'][0]>0,all(report['paired']['D_activity-'+b]['ci95'][0]>0 for b in ('D_geometry','R1_activity','D_old_geometry'))]
    assert all(criteria)==report['primary_pass'];state='completed_positive' if all(criteria) else 'completed_negative'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source='runs/z13_learned_association_20261023/report.json'
    obtained=source+', paired: '+json.dumps(report['paired'],ensure_ascii=False)+'. pair_audit.json: '+json.dumps(audit,ensure_ascii=False)+'. Все18моделей,1150шагов,73600реальных и73600искусственныхпар на модель; воспроизведение старого соответствия на8326строках каждого ядра точное.'
    interpretation=('Обучаемое соответствие прошло все поисковые критерии выбора на train26. Калибровка отказа и HOTA ещё не подтверждены.' if all(criteria) else 'Эта версия обучаемых соответствий не прошла все заранее заданные критерии. Это исход конкретного обучения и применения, не доказательство отсутствия информации об идентичности в FlyVis.')
    decision=('Перейти к зафиксированной отдельно проверке применения с контролем R1, без изменения по dev.' if all(criteria) else 'Не заменять рабочий считыватель этой реализацией. Не подбирать порог или лучший сид по train26. Следующее действие должно отличаться механизмом и иметь проверяемую мишень.')
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        if card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=card[field],reason='Полный проверенный исход явного соответствия.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for item,met in zip(card['acceptance'],criteria):item.update(met=met,evidence=source+'; reproduction.json; pair_audit.json; controller_result.json')
    card['metrics']=[dict(key='useful_center_fraction',value=report['summary']['D_activity']['useful_mean_video'],baseline=report['summary']['D_fixed']['useful_mean_video'],subset='train26 D learned association',source=source,locator='summary.D_activity.useful_mean_video')]
    card['confirmed_results']=[]
    for key in ('D_activity-D_fixed','D_activity-D_old_activity','D_activity-D_old_geometry','D_activity-R1_activity','C_activity-C_fixed','C_activity-R1_activity'):
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=IDENT+'_'+key.replace('-','_vs_'),title='Явное соответствие: '+key,category='representation' if 'R1' in key else 'readout',metric='useful_center_fraction',unit='fraction',comparison=key,subset='26train, известные присутствующие цели',conditions='104train,3сида,1150шагов; частичные реальные и искусственные соответствия; исходные3пика, причинный банк объектов. Парный бутстрэп5000,сид1.',limitations='Частное сравнение не отменяет остальные критерии; повторно использованные26train, нет HOTA и калиброванного отказа; не полная KeepTrack.',evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired',key])))
    card['evidence'].extend(dict(kind='file',ref='runs/z13_learned_association_20261023/'+name,claim=claim) for name,claim in [('report.json','Все парные исходы'),('reproduction.json','Точное воспроизведение фиксированного контроля'),('pair_audit.json','Настоящие размеченные пары и пробелы'),('controller_result.json','Своё дерево завершено')])
    write(cp,cat)
    baseline=report['summary']['D_fixed']['useful_mean_video'];relative=100*c['mean_per_video']/baseline if baseline else None
    journal('полный итог обучаемых соответствий','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+f'\n\nСводка: доля полезных центров D относительно прежнего фиксированного сопоставления {relative}%. Это не процент HOTA. HOTA и доля её потолка не измерены; отказ в данном узле не калиброван.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),state=state,primary_pass=all(criteria),report_sha256=sha(RUN/'report.json'),models=models,all_owned_ended=True,relative_percent_vs_fixed=relative))
    print(json.dumps(dict(state=state,main=c,criteria=criteria),ensure_ascii=False))


if __name__=='__main__':main()
