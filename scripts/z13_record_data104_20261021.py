"""Audit equal-compute train20/train104 comparison without changing science."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_data104_20261021 import RUN,OLD,IDENT,read,write,sha,verify,now,journal
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_train import success


def main():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(RUN/'protocol.json');verify(p)
    for stage in ('cache','train'):
        r=read(RUN/(stage+'_controller_result.json'));assert r['worker_ended'] and r['exit_code']==0 and not r['hard_timeout']
    assert len(p['train_full104'])==104 and len(p['evaluate'])==84 and len(p['fit'])==20 and len(p['evaluation26'])==26 and not set(p['train_full104'])&set(p['evaluation26'])
    report=read(RUN/'report.json');budget=read(RUN/'training_budget.json');seedmetrics=read(RUN/'seed_metrics.json');oldseeds=read(OLD/'seed_metrics.json')['per_video'];pv={};keyorder=None;models={}
    for rel,digest in read(RUN/'cache_manifest.json').items():assert sha(ROOT/rel)==digest
    for arm in ('C','D','R1'):
        for mode in ('activity','geometry'):
            rows=read(RUN/arm/mode/'ensemble_rows.json');keys=[(r['video'],r['label'],r['kind']) for r in rows];assert len(keys)==8326
            if keyorder is None:keyorder=keys
            else:assert keys==keyorder
            for row in rows:
                if row['kind']==1:assert row['useful']==bool(success(row))
            name=arm+'104_'+mode;pv[name]=per_video_useful(rows);assert np.isclose(np.mean(list(pv[name].values())),report['summary'][name]['useful_mean_video'],rtol=0,atol=1e-14)
            pv[arm+'20_'+mode]=per_video_useful(read(OLD/arm/mode/'ensemble_rows.json'))
            for seed in (0,1,2):
                folder=RUN/arm/mode/f'seed{seed}';tr=read(folder/'training.json');ref=read(OLD/arm/mode/f'seed{seed}/training.json')
                assert tr['steps']==50*((ref['known_rows']+127)//128)==budget[arm]['steps']
                assert tr['presented_rows']==50*ref['known_rows']==budget[arm]['presented']
                assert tr['pos_weight']==tr['reference_pos_weight']==ref['pos_weight'] and tr['seed']==seed
                assert len(tr['loss'])==tr['steps'] and sha(folder/'model.pt')==tr['model_sha256'];models[folder.relative_to(ROOT).as_posix()]=tr['model_sha256']
        for baseline in ('top1','kalman'):pv[arm+'_'+baseline]=per_video_useful(read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{baseline}_rows.json'))
        for seed,v in seedmetrics['per_video'][arm+'_activity'].items():assert paired(v,oldseeds[arm+'_activity'][seed])==report['seed_data_gain'][arm][seed]
    for key,obj in report['paired'].items():
        if key=='C_data_effect-R1_data_effect':
            a={v:pv['C104_activity'][v]-pv['C20_activity'][v] for v in pv['C20_activity']};b={v:pv['R1104_activity'][v]-pv['R120_activity'][v] for v in pv['R120_activity']}
        else:a,b=[pv[n] for n in key.split('-')]
        assert paired(a,b)==obj
    d=report['paired']['C104_activity-C20_activity'];others=['C104_activity-C104_geometry','C104_activity-R1104_activity','C104_activity-C_top1','C104_activity-C_kalman']
    first=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in report['seed_data_gain']['C'].values());second=all(report['paired'][k]['ci95'][0]>0 for k in others);passed=first and second;assert passed==report['primary_pass']
    write(RUN/'result_audit.json',dict(rows_and_paired_intervals_recomputed=True,matched8326keys=True,equal_optimizer_and_sample_budgets=True,fixed_class_weights_and_seeds=True,normalizer_train20_unchanged=True,no_evaluation26_in_training=True,model_sha256=models,cache_and_source_hashes_unchanged=True,report_sha256=sha(RUN/'report.json')))
    reserve=read(ROOT/'runs/z13_candidate_reserve_20261021/report.json');shares={}
    for arm in ('C','D'):
        oldbase=report['summary'][arm+'20_activity']['useful_mean_video'];cap=reserve['arms'][arm]['oracle3_mean']-oldbase;gain=report['paired'][arm+'104_activity-'+arm+'20_activity']['mean_per_video']
        shares[arm]=dict(relative_to_train20=gain/oldbase,remaining_same_candidate_oracle_increment=cap,signed_share=gain/cap if cap else None)
    write(RUN/'ceiling_share.json',shares)
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive')
    fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    def interval(v):return f"{fmt(v['mean_per_video'])} п.п. [{fmt(v['ci95'][0])}; {fmt(v['ci95'][1])}]"
    obtained='runs/z13_candidate_data104_20261021/report.json → paired: '+'; '.join(k+' '+interval(v) for k,v in report['paired'].items())+'. training_budget.json: '+str(budget)+'. result_audit.json: одинаковые шаги/предъявленные кадры, веса BCE,18 моделей и8326 строк проверены; нормировка20 и ядра прежние.'
    interpretation=('Все основные критерии расширения данных выполнены.' if passed else 'Основной критерий расширения данных этой реализации не выполнен.')+' Изменён пул обучающих видео, а не число обновлений или размер модели. Нехватка данных не объявляется причиной прошлого провала; сравнение поисковое, train26 использовался многократно. HOTA не измерялась.'
    if not passed and d['ci95'][1]<.02:interpretation+=' Верхняя граница прироста C ниже заранее заданных2п.п. в этом режиме.'
    decision='Сохранить фиксированную версию как функционального кандидата; проверить перенос с теми же контролями до применения в стенде.' if passed else 'Эту версию не переносить в стенд и не подбирать по train26/dev. Старые модели и исходный центрированный считыватель сохранены; для следующей замены нужен другой проверяемый механизм.'
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(v for v in cat['experiments'] if v['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный проверенный итог однофакторного опыта.'))
        card[field]=value
    card['state']=state;card['done']=f'{now():%Y-%m-%d}'
    for i,met in enumerate([True,first,second]):card['acceptance'][i].update(met=bool(met),evidence='runs/z13_candidate_data104_20261021/report.json')
    for name in ('report.json','training_budget.json','seed_metrics.json','result_audit.json','ceiling_share.json','cache_controller_result.json','train_controller_result.json'):
        card['evidence'].append(dict(kind='file',ref=(RUN/name).relative_to(ROOT).as_posix(),claim='Полный итог и проверка однофакторного сравнения'))
    card['confirmed_results']=[];src=(RUN/'report.json').relative_to(ROOT).as_posix();digest=sha(RUN/'report.json')
    for arm in ('C','D'):
        key=arm+'104_activity-'+arm+'20_activity'
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=arm+'_train104_vs_train20_equal_budget',title='Разнообразие train104 против train20 при равном бюджете — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',arm+'20_activity','useful_mean_video'],sha256=digest),subset='train26,26 видео; многократно использованная выборка',comparison='Одна архитектура, три фиксированных сида, равные обновления/предъявления; меняется пул обучающих видео.',conditions='Исходные20 модели переиспользованы. Нормировка20,веса BCE и настройки прежние. Парный бутстрэп по видео.',limitations='Частное сравнение не заменяет геометрию,R1 и все критерии карточки. Не HOTA, не безопасный отказ и не доказательство экономии данных.'))
    write(catpath,cat)
    rel=f"{100*shares['C']['relative_to_train20']:+.3f}%".replace('.',',');share=f"{100*shares['C']['signed_share']:+.3f}%".replace('.',',')
    journal('полный итог расширения данных считывателя','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к совпадающему train20 C: доля полезных центров '+rel+'; отношение сдвига к оставшемуся оракульному запасу тех же3 кандидатов '+share+' — точечное отношение без отдельного интервала. HOTA не измерялась. Формулировки проверены: предмет/инструмент, отсутствие лишнего повтора, соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,report_sha256=digest,recorded=now().isoformat()));print(state,interval(d))


if __name__=='__main__':main()
