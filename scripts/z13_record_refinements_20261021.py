"""Audit and record the fixed query-memory/local-peak refinements."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,verify,sha,now
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_train import success


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['query','local'],required=True);x=ap.parse_args()
    kind='query_memory' if x.kind=='query' else 'local_peak';ident='z13_'+kind+'_v1';run=ROOT/('runs/z13_'+kind+'_20261021')
    if (run/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(run/'protocol.json');verify(p);r=read(run/'full_controller_result.json');assert r['worker_ended'] and r['exit_code']==0 and not r['hard_timeout']
    report=read(run/'report.json');modes=('last','fixed','query') if x.kind=='query' else ('weighted','quadratic');pv={};keys=None
    for path,digest in read(run/'cache_manifest.json').items():assert sha(ROOT/path)==digest
    for arm in ('C','D','R1'):
        for mode in modes:
            rows=read(run/arm/(mode+'_rows.json'));k=[(v['video'],v['label'],v['kind']) for v in rows];assert len(k)==8326
            if keys is None:keys=k
            else:assert k==keys
            for row in rows:
                if row['kind']==1:assert row['useful']==bool(success(row))
            pv[arm+'_'+mode]=per_video_useful(rows);assert np.isclose(np.mean(list(pv[arm+'_'+mode].values())),report['summary'][arm+'_'+mode]['useful_mean_video'],rtol=0,atol=1e-14)
        rep=read(run/arm/'reproduction.json') if x.kind=='query' else read(run/'reproduction.json')[arm]
        assert rep['rows']==8326 and rep['max_center_px']<(.5 if x.kind=='query' else 1e-6) and rep['max_score']<(.001 if x.kind=='query' else 1e-9)
    for key,obj in report['paired'].items():
        if key=='C_gain-R1_gain':
            a={v:pv['C_quadratic'][v]-pv['C_weighted'][v] for v in pv['C_weighted']};b={v:pv['R1_quadratic'][v]-pv['R1_weighted'][v] for v in pv['R1_weighted']}
        else:
            a,b=[pv[n] for n in key.split('-')]
        assert paired(a,b)==obj
    if x.kind=='query':
        mainkey='C_query-C_last';others=['C_query-C_fixed','C_query-R1_query'];basekey='C_last'
        same_original=report['choices']['C']['query']['best']['eta']==0
        for arm,c in report['choices'].items():
            for mode,value in c.items():assert value['best']==max(value['grid'],key=lambda item:item['fit_score'])
        explanation=' Подбор на fit20 выбрал для C нулевой вклад памяти: применяется прежняя карта.' if same_original else ''
        cannot='Не полный STMTrack, не HOTA и не надёжный отказ. Вывод только об этой процедуре чтения отдельных моделей. Оракул прежних3 пиков не является потолком новой карты.'
    else:
        mainkey='C_quadratic-C_weighted';others=['C_quadratic-R1_quadratic','C_gain-R1_gain'];basekey='C_weighted';same_original=False
        explanation=' Прежний центр уже непрерывный; проверена другая локальная оценка той же вершины.'
        cannot='Не полная C-COT; текущие ошибки не объявляются следствием дискретизации. Нет HOTA, выбора другого пика или надёжного отказа. Оракул3 прежних центров не потолок новых координат.'
    d=report['paired'][mainkey];first=d['mean_per_video']>=.02 and d['ci95'][0]>0;second=all(report['paired'][k]['ci95'][0]>0 for k in others);passed=first and second;assert passed==report['primary_pass']
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive')
    fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    def interval(v):return f"{fmt(v['mean_per_video'])} п.п. [{fmt(v['ci95'][0])}; {fmt(v['ci95'][1])}]"
    obtained=(run/'report.json').relative_to(ROOT).as_posix()+' → paired: '+'; '.join(k+' '+interval(v) for k,v in report['paired'].items())+'. '+('choices: '+str(report['choices']['C']['query']['best'])+'. ' if x.kind=='query' else 'fallback: '+str(report['fallback'])+'. ')
    obtained=obtained.rstrip()
    interpretation=('Все основные критерии выполнены.' if passed else 'Основной критерий этой реализации не выполнен.')+explanation+' Поисковое сравнение на многократно использованном train26; это не HOTA и не оценка безопасности вмешательства.'
    if not passed and d['ci95'][1]<.02:interpretation+=' Верхняя граница прироста C ниже заранее заданных2п.п.; это ограничение данной реализации, не всей линии.'
    decision='Сохранить как функционального кандидата; следующий этап — перенос фиксированной версии с теми же контролями, затем отдельно применение в стенде.' if passed else 'В стенд не переносить; параметры по train26/dev не менять. Сохранить прежний центрированный считыватель. Другой механизм требует самостоятельной постановки и контроля без FlyVis.'
    audit=dict(worker_ended=True,exit_code=0,same8326keys=True,rows_and_intervals_recomputed=True,source_and_cache_hashes_unchanged=True,report_sha256=sha(run/'report.json'))
    write(run/'result_audit.json',audit)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(v for v in cat['experiments'] if v['id']==ident);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision),('cannot_claim',cannot)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный результат с независимой сверкой.'))
        card[field]=value
    card['state']=state;card['done']=f'{now():%Y-%m-%d}'
    for i,met in enumerate([True,first,second]):card['acceptance'][i].update(met=bool(met),evidence=(run/'report.json').relative_to(ROOT).as_posix())
    for name in ('report.json','result_audit.json','cache_manifest.json','full_controller_result.json'):
        card['evidence'].append(dict(kind='file',ref=(run/name).relative_to(ROOT).as_posix(),claim='Первичный итог и независимая сверка'))
    # Confirmed comparisons remain explicitly partial if the full gate failed.
    card['confirmed_results']=[];src=(run/'report.json').relative_to(ROOT).as_posix();digest=sha(run/'report.json')
    for arm in ('C','D'):
        key=(arm+'_query-'+arm+'_last') if x.kind=='query' else (arm+'_quadratic-'+arm+'_weighted');obj=report['paired'][key]
        if obj['ci95'][0]>0:
            bkey=arm+('_last' if x.kind=='query' else '_weighted')
            card['confirmed_results'].append(dict(id=arm+'_'+kind+'_vs_control_train26',title='Замена узла '+kind+' против совпадающего контроля — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',bkey,'useful_mean_video'],sha256=digest),subset='train26,26 видео; многократно использованная выборка',comparison='Те же фиксированные ядро и данные, один узел считывания.',conditions='Заранее зафиксированная процедура; парный бутстрэп по видео; настройки не выбираются на train26.',limitations='Частный эффект не заменяет все основные критерии. Нет HOTA или подтверждения безопасного применения.'))
    write(catpath,cat);baseline=report['summary'][basekey]['useful_mean_video'];relative=f'{100*d["mean_per_video"]/baseline:+.3f}%'.replace('.',',')
    text='*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к совпадающему контролю C: доля полезных центров '+relative+'; доля потолка не определена, соответствующий потолок этих новых карт/координат не измерялся. HOTA не измерялась. Формулировки проверены: предмет/инструмент, отсутствие лишнего повтора и соответствие фактически выполненному.'
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=plan.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1;plan.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — полный итог `{ident}`\n\n'+text+'\n',1),encoding='utf-8')
    write(run/'finish_recorded.json',dict(state=state,primary_pass=passed,report_sha256=digest,recorded=now().isoformat()));print(state,interval(d))


if __name__=='__main__':main()
