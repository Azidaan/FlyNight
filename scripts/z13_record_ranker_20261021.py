"""Independently audit and record the fixed three-seed candidate ranker."""
from __future__ import annotations
import sys
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_ranker_20261021 import RUN,IDENT,read,write,verify,sha,now,journal
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_train import success


def main():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(RUN/'protocol.json');verify(p);result=read(RUN/'full_controller_result.json')
    assert result['worker_ended'] and result['exit_code']==0 and not result['hard_timeout']
    report=read(RUN/'report.json');metrics=read(RUN/'seed_metrics.json');pv={};models={}
    for path,digest in read(RUN/'cache_manifest.json').items():assert sha(ROOT/path)==digest
    keys=None
    for arm in ('C','D','R1'):
        for mode in ('activity','geometry'):
            rr=read(RUN/arm/mode/'ensemble_rows.json');k=[(r['video'],r['label'],r['kind']) for r in rr]
            assert len(k)==8326
            if keys is None:keys=k
            else:assert k==keys
            for r in rr:
                if r['kind']==1:assert r['useful']==bool(success(r))
            pv[arm+'_'+mode]=per_video_useful(rr)
            assert np.isclose(np.mean(list(pv[arm+'_'+mode].values())),report['summary'][arm+'_'+mode]['useful_mean_video'],rtol=0,atol=1e-14)
            for seed in (0,1,2):
                folder=RUN/arm/mode/f'seed{seed}';tr=read(folder/'training.json');digest=sha(folder/'model.pt')
                assert digest==tr['model_sha256'] and tr['epochs']==50 and len(tr['loss'])==50 and tr['seed']==seed
                models[folder.relative_to(ROOT).as_posix()]=digest
        for mode in ('top1','kalman'):pv[arm+'_'+mode]=per_video_useful(read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json'))
    for key,obj in report['paired'].items():
        x,y=key.split('-');assert paired(pv[x],pv[y])==obj
    for arm in ('C','D','R1'):
        for s,v in metrics['per_video'][arm+'_activity'].items():assert paired(v,pv[arm+'_top1'])==report['seeds_against_top1'][arm][s]
    d=report['paired']['C_activity-C_top1'];other=['C_activity-C_geometry','C_activity-C_kalman','C_activity-R1_activity']
    first=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(r['mean_per_video']>=0 for r in report['seeds_against_top1']['C'].values())
    second=all(report['paired'][k]['ci95'][0]>0 for k in other);passed=first and second;assert passed==report['primary_pass']
    write(RUN/'result_audit.json',dict(rows_and_intervals_recomputed=True,matched_keys=True,fixed_three_seeds_verified=True,model_sha256=models,cache_and_source_hashes_unchanged=True,worker_ended=True,exit_code=0,report_sha256=sha(RUN/'report.json')))
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(x for x in cat['experiments'] if x['id']==IDENT)
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<0 else 'inconclusive')
    fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    def interval(v):return f"{fmt(v['mean_per_video'])} п.п. [{fmt(v['ci95'][0])}; {fmt(v['ci95'][1])}]"
    names=['C_activity-C_top1',*other,'D_activity-D_top1','D_activity-D_geometry','D_activity-D_kalman','D_activity-R1_activity']
    obtained='runs/z13_candidate_ranker_20261021/report.json → paired: '+'; '.join(k+' '+interval(report['paired'][k]) for k in names)+'. result_audit.json:8326 совпадающих строк,18 моделей (3 сида×2 режима×3 представления), исходники и кэши проверены; выбор лучшего сида не выполнялся.'
    reserve=read(ROOT/'runs/z13_candidate_reserve_20261021/report.json');shares={}
    for arm in ('C','D'):
        gain=report['paired'][arm+'_activity-'+arm+'_top1']['mean_per_video'];ceiling=reserve['arms'][arm]['reserve']['mean_per_video']
        shares[arm]=dict(gain=gain,oracle_increment=ceiling,signed_share=gain/ceiling,relative_to_top1=gain/report['summary'][arm+'_top1']['useful_mean_video'])
    write(RUN/'ceiling_share.json',shares)
    obtained+=' ceiling_share.json: сдвиг C / измеренный прирост оракула '+f"{100*shares['C']['signed_share']:+.3f}%".replace('.',',')+'; это отношение точечных оценок, без отдельного интервала и не HOTA.'
    interpretation=('Все основные критерии обучаемого выбора кандидатов выполнены.' if passed else 'Основной критерий этого обучаемого выбора кандидатов не выполнен; частные положительные сравнения не заменяют непройденные контроли.')+' Сравнение поисковое: train26 использовался многократно. Полученный выбор центра не является калиброванным отказом или результатом HOTA.'
    decision='Сохранить кандидата и проверить перенос фиксированных моделей на заранее выделенной дополнительной функциональной выборке; настройки не менять. Затем отдельно проверять причинное применение в стенде.' if passed else 'В стенд эту версию не переносить, не выбирать удачный сид и не подбирать по dev. Сохранить прежний центрированный считыватель; дальнейшая замена требует другого обоснованного механизма.'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный результат с независимым пересчётом.'))
        card[field]=value
    card['state']=state;card['done']=f'{now():%Y-%m-%d}'
    for i,met in enumerate([True,first,second]):card['acceptance'][i].update(met=bool(met),evidence='runs/z13_candidate_ranker_20261021/report.json')
    for name in ('report.json','seed_metrics.json','result_audit.json','ceiling_share.json','full_controller_result.json'):
        card['evidence'].append(dict(kind='file',ref=(RUN/name).relative_to(ROOT).as_posix(),claim='Полный первичный итог, сиды и независимая сверка'))
    card['confirmed_results']=[];src=(RUN/'report.json').relative_to(ROOT).as_posix();digest=sha(RUN/'report.json')
    for arm in ('C','D'):
        key=arm+'_activity-'+arm+'_top1';obj=report['paired'][key]
        if obj['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=arm+'_candidate_ranker_vs_top1_train26',title='Обучаемый выбор пика против максимума — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',arm+'_top1','useful_mean_video'],sha256=digest),subset='train26,26 видео; многократно использованная выборка',comparison='Те же3 пика и ядро; обучаемый выбор против прежнего максимума.',conditions='Фиксированный ансамбль3 сидов; обучение только fit20; парный бутстрэп по видео.',limitations='Частное сравнение не заменяет главные контроли геометрии и R1. Не HOTA, не безопасная политика отказа.'))
    write(catpath,cat)
    base=report['summary']['C_top1']['useful_mean_video'];rel=f"{100*d['mean_per_video']/base:+.3f}%".replace('.',',')
    share=f"{100*shares['C']['signed_share']:+.3f}%".replace('.',',')
    journal('полный итог обучаемого выбора кандидатов','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к top1 C: доля полезных центров '+rel+'; отношение сдвига к измеренному приросту оракула '+share+'. Отрицательное отношение означает удаление от потолка; это точечное отношение без собственного интервала. Не является HOTA. Научные формулировки проверены на предмет/инструмент, отсутствие лишнего повтора и соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,report_sha256=digest,recorded=now().isoformat()))
    print(state,interval(d))


if __name__=='__main__':main()
