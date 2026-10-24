"""Local completion monitor and independent artifact audit for common20."""
from pathlib import Path
import os,sys,time,subprocess,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_common_20261024 as S
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_weekend_driver_20261024 import verify_and_graph,execute
W=ROOT/'runs/z13_weekend_20261024'


def record():
    import numpy as np
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    from evaluation.hota_eval import evaluate_hota
    S.ended(S.STAGE);p=read(S.STAGE/'protocol.json');verify_sources(p);r=read(S.STAGE/'report.json');assert len(r['pooled'])==20 and r['videos']==p['videos'] and r['all_conditions_common']
    markers=list((S.STAGE/'jobs').glob('*.json'));assert len(markers)==231
    for f in markers:
        for n,h in read(f)['outputs'].items():assert sha(ROOT/n)==h
    for n,h in r['output_sha256'].items():assert sha(ROOT/n)==h
    for arm in S.SYSTEMS:
        for v in p['videos']:
            with np.load(S.RUN/arm/'signatures'/v/'signatures.npz') as z:
                assert int(z['frame_count'])==p['frames'][v] and np.isfinite(z['outputs']).all()
                if arm in S.ARMS:
                    with np.load(S.RUN/'raw'/arm/'raw'/(v+'.npz')) as old:assert np.array_equal(z['outputs'],old['outputs'])
            with np.load(S.RUN/arm/'masa'/(v+'.npz')) as z:assert all(np.isfinite(z[k]).all() for k in ('first','last'))
        for kind in ('raw','masa'):
            assert sorted(r['per_video'][arm+'_'+kind])==sorted(p['videos']);assert evaluate_hota(S.RUN/arm/('evaluated_'+kind),p['videos'],split='val')==r['pooled'][arm+'_'+kind]
    pv={a:{v:x['HOTA'] for v,x in rr.items()} for a,rr in r['per_video'].items()}
    for key,val in r['paired'].items():a,b=key.split('-');assert paired(pv[a],pv[b])==val
    cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(x for x in cat['experiments'] if x['id']==S.IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source=(S.STAGE/'report.json').relative_to(ROOT).as_posix()
    summary={a:dict(raw_hota=r['pooled'][a+'_raw']['HOTA'],same_masa_hota=r['pooled'][a+'_masa']['HOTA'],relative_percent_from_xywh_masa=100*(r['pooled'][a+'_masa']['HOTA']-r['pooled']['xywh_masa']['HOTA'])/r['pooled']['xywh_masa']['HOTA']) for a in S.SYSTEMS}
    write(S.STAGE/'verified_summary.json',dict(summary=summary,all231jobs_verified=True,all_paired_recomputed=True,pooled_hota_recomputed=True,report_sha256=sha(S.STAGE/'report.json'),videos=20,all_owned_ended=True))
    fields=dict(summary='Общее сравнение10систем на20val завершено и проверено;положительный статус означает полноту оценки,а не автоматический выигрыш FlyVis.',obtained=source+' → pooled/per_video/paired; verified_summary.json: '+json.dumps(summary,ensure_ascii=False),interpretation='Для каждой добавки D,сравнения со штатной C и контроля без мухи действует свой парный интервал;положительный нижний предел необходим. Общую HOTA и среднюю разность по видео не смешивать. Все исходы показаны;настройки по20val не выбирались.',decision='Использовать общую таблицу с n=20 и ограничениями. Не подбирать порог/сид и не открывать75. Только положительные конкретные контрасты могут попасть в лучшие результаты;нового обучения этот опыт не выполняет.')
    for k,v in fields.items():
        if c.get(k) and c[k]!=v:c.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=k,old=c[k],reason='Полное независимо проверенное сравнение20val.'))
        c[k]=v
    c.update(state='completed_positive',done=f'{now():%Y-%m-%d}')
    for item in c['acceptance']:item.update(met=True,evidence=source+';verified_summary.json;controller_result.json;231jobs')
    c['metrics']=[dict(key='HOTA',value=r['pooled'][a+'_masa']['HOTA'],baseline=r['pooled']['xywh_masa']['HOTA'],subset='20 common val '+a+' sameMASA',source=source,locator='pooled.'+a+'_masa.HOTA') for a in S.SYSTEMS];c['confirmed_results']=[]
    for a in ('legacyD','centerD','legacyC','centerC'):
        key=a+'_masa-xywh_masa'
        if r['paired'][key]['ci95'][0]>0:c['confirmed_results'].append(dict(id=S.IDENT+'_'+a,title=a+' против одинакового стенда на20val',category='tracker',metric='HOTA',unit='points',comparison=key,subset='20ранее использованныхval,не входят в gradient-train детектора',conditions=p['settings'],limitations=p['limitations']+' '+p['controls'],evidence=dict(source=source,sha256=sha(S.STAGE/'report.json'),locator=['paired',key])))
    c['evidence'].append(dict(kind='file',ref=source,claim='Все общие исходы и парные интервалы'));write(cp,cat)
    labels=dict(xywh='XYWHByteTrack',bytetrack='ByteTrack',botsort='BoT-SORT',tracktrack='TrackTrack',legacyD='XYWH + обученная D, прежнее считывание',centerD='XYWH + D, центрированное считывание',legacyC='XYWH + штатное ядро C, прежнее считывание',centerC='XYWH + C, центрированное считывание',legacyB='XYWH + малая обученная GRU B',centerR1='XYWH + временной вход глаза R1')
    text='Все20val,исходные кадры,общие детекции960/conf0,01,TrackEval. Показаны раздельно исходные трекеры и одна общая замороженная сшивка MASA. Набор не является независимым итоговым тестом.\n\n| Система | HOTA без сшивки | HOTA с одинаковой MASA |\n|---|---:|---:|\n'
    for a in S.SYSTEMS:text+='| '+labels[a]+' | '+f"{summary[a]['raw_hota']:.4f}".replace('.',',')+' | '+f"{summary[a]['same_masa_hota']:.4f}".replace('.',',')+' |\n'
    text+='\n'+p['controls']+'\n\n'+p['limitations']+'\n\nПервичный источник: '+source+'. Парные интервалы всех сравнений: paired. Проценты к базе: verified_summary.json/summary. Доля измеренного потолка неизвестна.\n';(S.STAGE/'results.md').write_text(text,encoding='utf-8')
    S.journal('полный итог общего сравнения20val','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nПроценты к базе:verified_summary.json/summary;доля потолка не определена. Условныецентры и прошлые7val не подменяют эти HOTA.')
    write(S.STAGE/'finish_recorded.json',dict(at=now().isoformat(),complete=True,report_sha256=sha(S.STAGE/'report.json'),publication_pending=True))


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    import msvcrt
    f=(S.STAGE/'completion.lock').open('a+b');f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);ps=WindowsProcesses();write(S.STAGE/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (S.STAGE/'controller_result.json').exists():time.sleep(5)
    controller=read(S.STAGE/'controller.json')
    while ps.creation(controller['pid'])==controller['birth']:time.sleep(.5)
    e=read(S.STAGE/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:
        write(W/'val20_common_failure.json',dict(at=now().isoformat(),result=e,status=read(S.STAGE/'status.json')));return
    execute('record_val20_common',['scripts/z13_val20_common_finish_20261024.py','--record']);verify_and_graph('val20_common');write(W/'val20_common_complete.json',dict(at=now().isoformat(),publication_pending=True))


if __name__=='__main__':
    if '--record' in sys.argv:record()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:write(W/'val20_common_monitor_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit finish mode required')
