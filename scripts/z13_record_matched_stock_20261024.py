"""Verify and record complete stock/no-FlyVis controls, preserving main report."""
from pathlib import Path
import sys,csv,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_stock_controls_20261024 import STAGE,RUN,SYSTEMS,VIDEOS,read,write,sha,now,IDENT,journal


def main():
    import numpy as np
    from scripts.z13_record_matched_trackers_20261024 import ended,main as record_parent
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not (STAGE/'finish_recorded.json').exists();ended(STAGE)
    p=read(STAGE/'protocol.json');verify_sources(p);r=read(STAGE/'report.json');assert r['parent_report_sha256']==sha(RUN/'report.json') and len(r['pooled'])==20 and r['videos']==list(VIDEOS)
    jobs=list((STAGE/'jobs').glob('*.json'));assert len(jobs)==35
    for path in jobs:
        m=read(path)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
    for n,h in r['output_sha256'].items():assert sha(ROOT/n)==h
    for arm in SYSTEMS:
        for video in VIDEOS:
            with np.load(RUN/arm/'signatures'/video/'signatures.npz') as z:assert np.isfinite(z['outputs']).all() and int(z['frame_count'])==read(RUN/'protocol.json')['frames'][video]
            with np.load(RUN/arm/'masa'/(video+'.npz')) as z:assert all(np.isfinite(z[k]).all() for k in ('first','last'))
        for kind in ('raw','masa'):assert sorted(r['per_video'][arm+'_'+kind])==sorted(VIDEOS)
    parent=read(RUN/'report.json')
    for k,v in parent['pooled'].items():assert r['pooled'][k]==v and r['per_video'][k]==parent['per_video'][k]
    pv={k:{v:m['HOTA'] for v,m in values.items()} for k,values in r['per_video'].items()}
    for k,v in r['paired'].items():
        a,b=k.split('-');assert paired(pv[a],pv[b])==v
    if not (RUN/'finish_recorded.json').exists():record_parent()
    names={'bytetrack':'ByteTrack','botsort':'BoT-SORT','tracktrack':'TrackTrack','xywh':'Наш XYWHByteTrack без мухи','legacyC':'Штатное ядро C, прежний считыватель','legacyD':'Обученное ядро D, прежний считыватель','centerC':'Штатное ядро C, центрированный считыватель','centerD':'Обученное ядро D, центрированный считыватель','legacyB':'Контроль B: малая рекуррентная сеть без FlyVis','centerR1':'Контроль R1: временные каналы входа глаза'}
    order=tuple(names);base=r['pooled']['xywh_masa']['HOTA'];summary={a:dict(raw=r['pooled'][a+'_raw']['HOTA'],masa=r['pooled'][a+'_masa']['HOTA'],pooled_delta=r['pooled'][a+'_masa']['HOTA']-base,relative_percent=100*(r['pooled'][a+'_masa']['HOTA']-base)/base) for a in order}
    text='Все строки измерены на тех же7ранее использованныхval-видео,всех исходных кадрах,одних детекциях YOLO26s960/conf0,01 и одном оценщике TrackEval. Дополнительного обучения или подбора по этим видео нет.\n\n| Система | HOTA без сшивки | HOTA с одинаковой MASA |\n|---|---:|---:|\n'
    for a in order:text+=f"| {names[a]} | {summary[a]['raw']:.4f} | {summary[a]['masa']:.4f} |\n"
    text+='\nСравнения штатной и обученной мухи включают индивидуально обученные считыватели и заранее выбранные пороги. Это сравнение готовых модулей,не изолированный причинный эффект только ядра. B/R1 — соответствующие контроли безFlyVis,но не равные ей по ёмкости и устройству. Общая MASA-голова ранее обучена на жизнях XYWH;её перенос может различаться. Поэтому обе колонки обязательны. Ни перекрытие интервала с нулём,ни близость общих баллов не доказывают эквивалентность. Эти7видео не входят в градиентное обучение детектора,но егоval использовался при выборе весов и сами7неоднократно участвовали в исследовании;это не независимый итоговый тест.\n\n'
    for key in p['contrasts']:
        a,b=key.split('-');d=r['paired'][a+'_masa-'+b+'_masa'];pooled=r['pooled'][a+'_masa']['HOTA']-r['pooled'][b+'_masa']['HOTA']
        text+=f"{names[a]} против {names[b]}: общая разность {pooled:+.6f}пункта;средняя повидео {d['mean_per_video']:+.6f},95%интервал[{d['ci95'][0]:+.6f};{d['ci95'][1]:+.6f}],лучше{d['better']}/7.\n\n"
    text+='Все проценты относительно того же XYWH+MASA и первичные сравнения: verified_summary.json;данные:report.json. Доля измеренного потолка неизвестна. Прежние61,65на35dev и200val сюда не переносятся.\n'
    (STAGE/'results.md').write_text(text,encoding='utf-8')
    with (STAGE/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['Система','HOTA без сшивки','HOTA одинаковаяMASA','Разность к XYWH+MASA','Процент к XYWH+MASA'])
        for a in order:w.writerow([names[a],*summary[a].values()])
    write(STAGE/'verified_summary.json',dict(summary=summary,all35jobs_verified=True,all_parent_results_unchanged=True,all_owned_ended=True,report_sha256=sha(STAGE/'report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT);source=(STAGE/'report.json').relative_to(ROOT).as_posix();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='Общее сравнение штатной мухи,обученной D и соответствующих контролей завершено. Знак каждого эффекта определяется отдельным парным интервалом.',obtained=source+' → pooled/per_video/paired. Полная таблица и проценты: '+(STAGE/'verified_summary.json').relative_to(ROOT).as_posix()+' → summary: '+json.dumps(summary,ensure_ascii=False),interpretation='ШтатнаяC — замороженное предобученное ядро с адаптированным считывателем;самостоятельногоHOTA для ядра нет. СравнениеC/Dне выделяет чистый вклад ядра из отдельно обученных считывателей. Все исходы сохранены,положительные выводы только для точно совпадающих контролей с нижнейграницей>0.',decision='Использовать эту таблицу10систем сn=7;не подбирать пороги по ней. Сохранять различие улучшения считывания,признаков и всего стенда.')
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=k,old=card[k],reason='Полный проверенный итог дополнительных контролей.'))
        card[k]=v
    card.update(done=f'{now():%Y-%m-%d}',state='inconclusive')
    for a in card['acceptance']:a.update(met=True,evidence=source+'; verified_summary.json; controller_result.json')
    card['metrics']=[dict(key='HOTA',value=r['pooled'][a+'_masa']['HOTA'],baseline=base,subset='7 common val '+a+' sameMASA',source=source,locator='pooled.'+a+'_masa.HOTA') for a in SYSTEMS]
    card['confirmed_results']=[]
    for key in p['contrasts']:
        a,b=key.split('-');contrast=a+'_masa-'+b+'_masa'
        if r['paired'][contrast]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=IDENT+'_'+key,title=names[a]+' против '+names[b],category='tracker',metric='HOTA',unit='points',comparison=contrast,subset='7ранее использованныхval',conditions=p['method'],limitations=p['limitations'],evidence=dict(source=source,sha256=sha(STAGE/'report.json'),locator=['paired',contrast])))
    if any(x['comparison'] in ('legacyD_masa-legacyB_masa','centerD_masa-centerR1_masa') for x in card['confirmed_results']):card['state']='completed_positive'
    card['evidence'] += [dict(kind='file',ref=(STAGE/n).relative_to(ROOT).as_posix(),claim=c) for n,c in [('verified_summary.json','Все35заданий,данные и сравнения проверены'),('results.md','Единая таблица10систем')]];write(cp,cat)
    journal('полный итог штатной мухи и контролей без неё','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nОтносительные проценты к базе сохранены в verified_summary.json. Доля потолка неизвестна.')
    write(STAGE/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(STAGE/'report.json'),all35jobs_verified=True))


if __name__=='__main__':main()
