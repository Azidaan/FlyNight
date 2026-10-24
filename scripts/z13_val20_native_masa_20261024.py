"""Independent four-tracker MASA baseline while the FlyVis replay gate is pending."""
from pathlib import Path
from datetime import datetime
import sys,os,time,inspect,shutil,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_common_20261024 as S
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OLD=S.STAGE;RUN=S.RUN;STAGE=RUN/'native_masa';IDENT='z13_val20_native_masa_v1';ORIGINAL_EVALUATE=S.evaluate


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as J
    before=J.IDENT
    try:J.IDENT=IDENT;J.journal(title,text)
    finally:J.IDENT=before


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not STAGE.exists();S.ended(RUN/'common_replay_diagnostic');p=read(OLD/'protocol.json');verify_sources(p);assert not read(RUN/'common_replay_diagnostic/report.json')['old_runner_vs_original']['exact']
    names=['scripts/z13_val20_native_masa_20261024.py','scripts/z13_val20_native_masa_console.ps1'];auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    p={**p,'id':IDENT,'created':now().isoformat(),'budget_seconds':36000,'save_at':auth['save_at'],'hard_stop':auth['hard_stop'],'stop_request_path':(STAGE/'stop.request').relative_to(ROOT).as_posix(),'systems':list(S.ARMS),'models':{},'sha256':{**p['sha256'],**{n:sha(ROOT/n) for n in names}},'action':'Only the4native trackers with identical MASA on the fixed20. Independent useful benchmark baseline requested by author; no FlyVis inference,weights,gate relaxation or claim of FlyVis efficacy. The pending10system protocol remains stopped,not silently reduced to success.','gates':'Preserve4exact existing control signatures;fresh baseline MASA on old control must exactly repeat previous embeddings. Then fixed first new video4signatures+MASA retained;2x walltime*19+900 fits remaining36000s. Native raw arrays must exactly match independent CPU20 and projected raw JSON must match. All105jobs/20videos and raw/MASA required. Exact guards unchanged.','primary':'All4raw/MASA metrics,paired-video95%CI for every native-vs-XYWH and within-system MASA contrast. Strict positive lower required for that exact benefit claim. No FlyVis benefit can be claimed.','budget_reason':'Independent native-only baseline,not extension of stopped10-system FlyVis gate or prior stages. Frozen actual native first-video signatures~576s plus prior sharedMASA156s:conservative2x732x19+900~28716s,below36000;mandatory fresh measured gate remains. No new core/readout training.'}
    write(STAGE/'protocol.json',p)
    for arm in S.ARMS:
        old=read(OLD/'jobs'/f'probe_signature_{arm}_{p["probe_video"]}.json');assert old['result']['exact_replay']
        target=STAGE/'probe'/arm/'signatures'/p['probe_video']/'signatures.npz';target.parent.mkdir(parents=True,exist_ok=True);source=OLD/'probe'/arm/'signatures'/p['probe_video']/'signatures.npz';assert sha(source)==old['result']['signature_sha256'];shutil.copy2(source,target);assert sha(target)==sha(source)
        write(STAGE/'jobs'/f'probe_signature_{arm}_{p["probe_video"]}.json',dict(at=now().isoformat(),result=old['result'],outputs={target.relative_to(ROOT).as_posix():sha(target)},reused_verified_control=old,wall_seconds_with_imports=old['wall_seconds_with_imports']))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();common=next(c for c in cat['experiments'] if c['id']==S.IDENT)
    fields=dict(summary='Общее10-системное сравнение остановлено на точном контроле D;готовы4контроля,эффективность не измерена.',obtained='runs/z13_val20_matched_20261024/common/controller_result.json:exit1,all_owned_ended=true;common_replay_diagnostic/report.json:старыйпроизводительтакженеповторилпобитово,максимум0,000030517578125px,числастрок/кадры/идентификаторы/классы/уверенностиравны. Сохранены4точныхконтроляобычныхтрекеров;ниодногоиз20основныхроликов10системнеобработано.',interpretation='Строгий технический критерий не выполнен;это не научный отрицательный итог D. Конкретный источник малого межпроходного разброса не установлен.',decision='Не продолжать ветвь FlyVis с изменённым допуском без решения автора;вопрос задан. Независимо выполняется отдельная общая MASA четырёхтрекеров с прежними точнымиконтролями. Старые ошибка/источники/код/журналы сохранены;бесконечногоповтораDнет.')
    for k,v in fields.items():
        if common.get(k) and common[k]!=v:common.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=k,old=common[k],reason='Строгий контроль старой D повторно не выполнен;допуск без автора не ослаблен.'))
        common[k]=v
    common.update(state='inconclusive',done=f'{now():%Y-%m-%d}');common['acceptance'][0].update(met=False,evidence='runs/z13_val20_matched_20261024/common_replay_diagnostic/report.json')
    assert not any(c['id']==IDENT for c in cat['experiments']);cat['experiments'].append(dict(id=IDENT,title='20val: четыре исходных трекера с одинаковой MASA',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['benchmark','infrastructure'],question='Каков вклад одной замороженной MASA и рейтинг четырёх исходных трекеров в одинаковых условиях20val?',hypothesis='Знак добавки MASA и порядок не предполагаются;это окружение проверки FlyVis,не её улучшение.',expected=p['action']+' '+p['gates'],acceptance=[dict(criterion=p['gates'],met=None),dict(criterion='Полные105заданий/20видео,все исходные строки/хеши/парные интервалы проверены;exit0 и конец дерева.',met=None),dict(criterion=p['primary'],met=None)],summary='Отдельный базис без FlyVis;проваленный контроль D не обходится.',obtained='',interpretation='',decision='',cannot_claim=p['limitations']+' УлучшенияFlyVisнет;её запуск ожидает отдельного решения по точному воспроизведению.',metrics=[],commits=[],files=names,evidence=[dict(kind='file',ref=(STAGE/'protocol.json').relative_to(ROOT).as_posix(),claim='Протокол до исходов,с точными контролями и прежними данными')]))
    write(cp,cat);journal('общий базис MASA продолжается без спорного контроля FlyVis','*Постановка.* '+p['action']+'\n\n'+p['gates']+'\n\n'+p['budget_reason']+'\n\nОбщий10системныйопыт не объявленуспешным:егоитогтехническинеопределён,старыйточныйкритерийнепройден. Автору заданвопрособотдельномслужебномдопуске;доответаFlyVisневычисляется. Четыреготовыхстрогоравныхконтроляскопированыипроверены,GPUнеповторяется.')


def bind():
    S.STAGE=STAGE;S.SYSTEMS=S.ARMS;S.T0=time.monotonic();S.__file__=str(Path(__file__))
    def status(stage,**kw):
        S.check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-S.T0,2),jobs_done=len(list((STAGE/'jobs').glob('*.json'))),jobs_total=105,**kw))
    S.status=status


def evaluate():
    source=inspect.getsource(ORIGINAL_EVALUATE);lines=source.splitlines();drop=[s for s in lines if 'for a,b in ' in s];assert len(drop)==1;source='\n'.join(s for s in lines if s not in drop)+'\n';ns=vars(S).copy();exec(compile(source,str(Path(__file__)),'exec'),ns);ns['evaluate']()


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(STAGE/'protocol.json');verify_sources(p);S.child('masa','all',p['probe_video'],True)
    for i,v in enumerate(p['videos']):
        start=time.monotonic()
        for a in S.ARMS:S.status('Четыре исходных трекера',variant=a+' '+v);S.child('signature',a,v)
        S.child('masa','all',v)
        if i==0:
            wall=time.monotonic()-start;estimate=2*wall*19+900;remaining=p['budget_seconds']-(time.monotonic()-S.T0);write(STAGE/'price.json',dict(first_new_video_seconds=wall,conservative_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate<remaining));assert estimate<remaining,'Native-only measured price exceeds original36000s'
    evaluate();write(STAGE/'status.json',dict(state='completed',stage='4трекера×20val,raw/MASA готовы;безFlyVis',updated=now().isoformat(),jobs_done=105,jobs_total=105))


def record():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    from evaluation.hota_eval import evaluate_hota
    import numpy as np
    S.ended(STAGE);p=read(STAGE/'protocol.json');verify_sources(p);r=read(STAGE/'report.json');assert len(r['pooled'])==8 and len(list((STAGE/'jobs').glob('*.json')))==105
    for path in (STAGE/'jobs').glob('*.json'):
        for n,h in read(path)['outputs'].items():assert sha(ROOT/n)==h
    for n,h in r['output_sha256'].items():assert sha(ROOT/n)==h
    for a in S.ARMS:
        for v in p['videos']:
            with np.load(RUN/a/'signatures'/v/'signatures.npz') as x,np.load(RUN/'raw'/a/'raw'/(v+'.npz')) as y:assert np.array_equal(x['outputs'],y['outputs'])
        for kind in ('raw','masa'):assert evaluate_hota(RUN/a/('evaluated_'+kind),p['videos'],split='val')==r['pooled'][a+'_'+kind]
    pv={a:{v:x['HOTA'] for v,x in rows.items()} for a,rows in r['per_video'].items()}
    for key,value in r['paired'].items():a,b=key.split('-');assert paired(pv[a],pv[b])==value and value['n']==20
    cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();source=(STAGE/'report.json').relative_to(ROOT).as_posix();fields=dict(summary='Полный общий базис четырёхтрекеров с MASA завершён и проверен;не результат FlyVis.',obtained=source+' → pooled/per_video/paired: '+json.dumps(dict(pooled=r['pooled'],paired=r['paired']),ensure_ascii=False),interpretation='Для каждого сравнения действует его отдельный парный интервал по20видео;общий и среднийповидео HOTA различаются. Вклад FlyVis не измерялся.',decision='Сохранить общий базис для последующего сравнения мухи. Не подбирать настройки по20val,не открывать75. Ветвь FlyVis требует отдельного разрешённого контроля воспроизведения.')
    for k,v in fields.items():
        if c.get(k) and c[k]!=v:c.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=k,old=c[k],reason='Полный независимо проверенный базис.'))
        c[k]=v
    c.update(state='completed_positive',done=f'{now():%Y-%m-%d}')
    for a in c['acceptance']:a.update(met=True,evidence=source+';controller_result.json;finish_recorded.json')
    c['metrics']=[dict(key='HOTA',value=r['pooled'][a+'_masa']['HOTA'],baseline=r['pooled']['xywh_masa']['HOTA'],subset='20 common val '+a+' sameMASA withoutFlyVis',source=source,locator='pooled.'+a+'_masa.HOTA') for a in S.ARMS];write(cp,cat)
    summary={a:dict(raw=r['pooled'][a+'_raw']['HOTA'],masa=r['pooled'][a+'_masa']['HOTA'],relative_percent_from_xywh_masa=100*(r['pooled'][a+'_masa']['HOTA']-r['pooled']['xywh_masa']['HOTA'])/r['pooled']['xywh_masa']['HOTA']) for a in S.ARMS};write(STAGE/'verified_summary.json',dict(summary=summary,report_sha256=sha(STAGE/'report.json'),all105jobs_verified=True,no_flyvis=True))
    text='Одни20val,исходныекадры,детекции960/conf0,01,TrackEval и замороженная MASA;FlyVis не используется. Набор не независимый итоговый тест.\n\n| Система | HOTA без сшивки | HOTA с одинаковой MASA |\n|---|---:|---:|\n'
    for a,label in zip(S.ARMS,('XYWHByteTrack','ByteTrack','BoT-SORT','TrackTrack')):text+='| '+label+' | '+f"{summary[a]['raw']:.4f}".replace('.',',')+' | '+f"{summary[a]['masa']:.4f}".replace('.',',')+' |\n'
    (STAGE/'results.md').write_text(text+'\nИсточник:report.json/pooled,парные интервалы:paired,проценты:verified_summary.json/summary. Доля потолка не определена.\n'+p['limitations'],encoding='utf-8');journal('полный итог общего базиса MASA20val','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nПроценты к базе:verified_summary.json/summary;доля потолка неизвестна.');write(STAGE/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(STAGE/'report.json'),complete=True))


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    import msvcrt
    lock=(STAGE/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);ps=WindowsProcesses();write(STAGE/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (STAGE/'controller_result.json').exists():time.sleep(5)
    p=read(STAGE/'controller.json')
    while ps.creation(p['pid'])==p['birth']:time.sleep(.5)
    e=read(STAGE/'controller_result.json');W=ROOT/'runs/z13_weekend_20261024'
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:write(W/'val20_native_masa_failure.json',dict(at=now().isoformat(),result=e,status=read(STAGE/'status.json')));return
    execute('record_val20_native_masa',[str(Path(__file__)),'--record']);verify_and_graph('val20_native_masa');write(W/'val20_native_masa_complete.json',dict(at=now().isoformat(),publication_pending=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        bind()
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    elif '--job' in sys.argv:
        bind();mode,a,v=sys.argv[2:5];probe='--probe' in sys.argv;p=read(STAGE/'protocol.json');assert v in ([p['probe_video']] if probe else p['videos']) and a in (*S.ARMS,'all');result=S.build_signature(a,v,probe) if mode=='signature' else S.build_masa(v,probe);base=STAGE/'probe' if probe else RUN;paths=[base/a/'signatures'/v/'signatures.npz'] if mode=='signature' else [base/x/'masa'/(v+'.npz') for x in S.ARMS];write(STAGE/'jobs'/f'{"probe_" if probe else ""}{mode}_{a}_{v}.json',dict(at=now().isoformat(),result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths}))
    elif '--record' in sys.argv:bind();record()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:write(ROOT/'runs/z13_weekend_20261024/val20_native_masa_monitor_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit native-only mode required')
