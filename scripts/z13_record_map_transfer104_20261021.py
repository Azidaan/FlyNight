"""Audit fixed dev35 transfer from primary rows; no new prediction or fitting."""
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_map_transfer104_20261021 import RUN,DEV,IDENT,read,write,sha,verify,now,journal
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_train import success


def main():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(RUN/'protocol.json');verify(p)
    for name in ('technical','cache','evaluate'):
        end=read(RUN/(name+'_controller_result.json'));assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    assert read(RUN/'technical.json')['passed'];report=read(RUN/'report.json');repro=read(RUN/'reproduction.json');seeds=read(RUN/'seed_metrics.json');pv={};keys=None
    for rel,digest in read(RUN/'cache_manifest.json').items():assert sha(ROOT/rel)==digest
    for arm in ('C','D','R1'):
        assert repro[arm]['rows']==9314 and repro[arm]['max_center_px']<.5 and repro[arm]['max_score']<.001
        for mode in ('map','top1','nearest'):
            name=arm+'_'+mode;rows=read(RUN/arm/(mode+'_rows.json'));current=[(r['video'],r['label'],r['kind']) for r in rows];assert len(rows)==9314 and len(set(current))==9314
            if keys is None:keys=current
            else:assert current==keys
            assert set(r['video'] for r in rows)==set(p['videos'])
            for r in rows:
                if r['kind']==1:assert bool(r['useful'])==bool(success(r))
            pv[name]=per_video_useful(rows);assert np.isclose(np.mean(list(pv[name].values())),report['summary'][name]['useful_mean_video'],rtol=0,atol=1e-14)
    pv['D_prior']=per_video_useful([r for v in p['videos'] for r in read(DEV/'D'/v/'prior_rows.json')])
    for key,obj in report['paired'].items():
        if key=='C_gain-R1_gain':
            a={v:pv['C_map'][v]-pv['C_top1'][v] for v in pv['C_top1']};b={v:pv['R1_map'][v]-pv['R1_top1'][v] for v in pv['R1_top1']}
        else:a,b=[pv[k] for k in key.split('-')]
        assert paired(a,b)==obj
    for seed,v in seeds['C'].items():assert paired(v,pv['C_top1'])==report['C_seeds_vs_top1'][seed]
    d=report['paired']['C_map-C_top1'];first=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in report['C_seeds_vs_top1'].values());second=all(report['paired'][k]['ci95'][0]>0 for k in ('C_map-R1_map','C_map-C_nearest'));passed=first and second;assert passed==report['primary_pass']
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive')
    fmt=lambda x:f'{100*x:+.3f}'.replace('.',',')
    def interval(o):return f"{fmt(o['mean_per_video'])} п.п. [{fmt(o['ci95'][0])}; {fmt(o['ci95'][1])}]"
    obtained='runs/z13_map_transfer104_20261021/report.json → paired: '+'; '.join(k+' '+interval(v) for k,v in report['paired'].items())+'. reproduction.json: все9314 строк/35видео и прежний top1 воспроизведены с исходными допусками; модели, кэши и источники SHA256 проверены.'
    interpretation=('Все основные критерии фиксированного переноса C выполнены.' if passed else 'Основные критерии фиксированного переноса C не выполнены.')+' Это другой ранее использованный dev35 без подбора. D вторична; общий эффект считывателя отдельно показан на R1. Нет HOTA и доказанного безопасного отказа.'
    decision='Сохранить версию для технической проверки покадрового применения и стоимости; сравнение стенда с R1 и без поправки обязательно.' if passed else 'Не переносить эту версию в стенд и не подбирать на dev. Сохранить положительный исход train26 как ограниченный им; прежнее центрированное считывание остаётся рабочей версией. Следующая замена требует новой проверяемой постановки.'
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный проверенный фиксированный перенос.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for i,value in enumerate([True,first,second]):card['acceptance'][i].update(met=bool(value),evidence='runs/z13_map_transfer104_20261021/report.json')
    for name in ('report.json','reproduction.json','result_audit.json','seed_metrics.json'):
        card['evidence'].append(dict(kind='file',ref='runs/z13_map_transfer104_20261021/'+name,claim='Полный исход и независимая проверка строк/интервалов'))
    card['confirmed_results']=[];digest=sha(RUN/'report.json');src='runs/z13_map_transfer104_20261021/report.json'
    for arm in ('C','D'):
        key=arm+'_map-'+arm+'_top1'
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=arm+'_map104_vs_top1_dev35',title='Фиксированное чтение карты104 против максимума на dev35 — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',arm+'_top1','useful_mean_video'],sha256=digest),subset='dev35,35 ранее использованных видео',comparison='Замороженный ансамбль3 geometry-моделей104 на прежних3 пиках; обучение и настройки по dev не менялись.',conditions='Воспроизведение9314 строк, первичные интервалы и исходные хеши проверены.',limitations='Частное сравнение не заменяет все критерии. D — вторична, не спасает основной исход C. Карта сохраняет FlyVis, эффект общего узла отдельно проверяется на R1. Нет HOTA или безопасного отказа.'))
    write(RUN/'result_audit.json',dict(primary_rows_and_intervals_recomputed=True,all_35_videos_9314_keys=True,all_three_stage_trees_ended_exit0=True,strict_original_reproduction_passed=True,model_and_source_sha256_unchanged=True,report_sha256=digest));write(path,cat)
    base=report['summary']['C_top1']['useful_mean_video'];rel=100*d['mean_per_video']/base
    journal('полный итог фиксированного переноса карты104','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к C top1 на том же dev35: доля полезных центров '+f'{rel:+.3f}%'.replace('.',',')+'; доля измеренного потолка не определена — оракул3пиков на этом dev не измерялся, потолок train26 не переносится. HOTA не измерялась. Формулировки проверены: предмет/инструмент, отсутствие лишнего повтора и соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,report_sha256=digest,recorded=now().isoformat()));write(RUN/'night_finished.json',dict(compute_ended=True,result_recorded=True,publication_pending=True,at=now().isoformat()));print(state,interval(d))


if __name__=='__main__':main()
