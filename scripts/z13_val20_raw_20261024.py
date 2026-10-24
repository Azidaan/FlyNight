"""Additional fixed20val: native raw trackers only, before a shared MASA stage."""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_matched_trackers_20261024 as M
from scripts import z13_val20_inputs_20261024 as I
RUN=I.RUN;STAGE=RUN/'raw';IDENT='z13_val20_raw_v1';T0=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as J
    old=J.IDENT
    try:J.IDENT=IDENT;J.journal(title,text)
    finally:J.IDENT=old


def register():
    cp=ROOT/'experiments/project_history.json';c=read(cp);assert not any(x['id']==IDENT for x in c['experiments'])
    c['experiments'].append(dict(id=IDENT,title='Общие20val: четыре исходных трекера до сшивки MASA',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['benchmark'],question='Каковы результаты четырёх трекеров на фиксированных дополнительных20val при одинаковых исходных кадрах и детекциях?',hypothesis='Проверка сравнительного окружения FlyVis;знак и победитель заранее не предполагаются.',expected='После полного проверенного входного этапа:XYWHByteTrack,ByteTrack,BoT-SORT,TrackTrack,старые общие настройки без подбора. Сначала точный повтор всех4на прежнем контрольном видео;затем20фиксированных новых.3600с CPU,цена2xпервого полного видео всех4сHOTAнаоставшиеся19+120с. Никакого сравнения с MASA или D до отдельного общего этапа.',acceptance=[dict(criterion='Все входы и контрольные4исходных трекера точно воспроизведены;все20×4задания и одинаковые размеченные кадры готовы.',met=None),dict(criterion='Официальная HOTA,парные95%ДИ по всем20видео и независимая сверка;для заявления преимущества нижняяграница>0.',met=None),dict(criterion='Свой бюджет3600с и независимый конец дерева соблюдены;частичные исходы не объявляются итогом сравнения.',met=None)],summary='Зависимая исходная ступень;без сшивки,FlyVis и улучшения мухи.',obtained='',interpretation='',decision='',cannot_claim=read(RUN/'selection.json')['limitations']+' Без MASA и FlyVis:нельзя сравнивать эту таблицу с прежними системами после MASA или заявлять результат D.',metrics=[],commits=[],files=['scripts/z13_val20_raw_20261024.py'],evidence=[dict(kind='file',ref=(RUN/'selection.json').relative_to(ROOT).as_posix(),claim='Список выбран до новых исходов')]));write(cp,c)
    journal('зарегистрирован автоматический исходный этап20val','*Постановка.* Фиксированные20val из selection.json;после полного входногоэтапа четыре прежних трекера,все исходныекадры и единая оценка. Конфигурации неизменны,4точныхповторастарогоконтрольноговидео обязательны. Бюджет3600сCPU;первый новыйролик всех4+HOTA оценивает консервативнуюцену остальных19. Эта стадия ещё не общий итог с MASA/D и не проверка пользы мухи. Все200val ранееиспользовались;независимого финального теста здесь нет.')


def check():
    p=read(STAGE/'protocol.json')
    if (STAGE/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Raw20 budget/deadline')


def status(stage,**kw):
    check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),jobs_done=len(list((STAGE/'jobs').glob('*.json'))),jobs_total=84,**kw))


def prepare():
    assert not (STAGE/'protocol.json').exists();r=read(I.STAGE/'report.json');assert r['jobs']==21 and r['detector_replay_exact']
    p=read(I.STAGE/'protocol.json');from scripts.z13_localization_labels_20261023 import verify_sources
    verify_sources(p);names=['scripts/z13_val20_raw_20261024.py','scripts/z13_val20_queue_20261024.py','scripts/z13_val20_raw_console.ps1','scripts/z13_matched_trackers_20261024.py','scripts/z13_continue.py','evaluation/hota_eval.py','tracking/xywh_bytetrack.py','tracking/offline_association.py']
    import inspect
    for arm in M.ARMS:names.append(Path(inspect.getfile(type(M.make_tracker(arm,p['configs'])))).relative_to(ROOT).as_posix())
    hashes={**p['sha256'],**{n:sha(ROOT/n) for n in names}}
    for v in p['videos']:
        marker=read(I.STAGE/'jobs'/(v+'.json'))
        for n,h in marker['outputs'].items():assert sha(ROOT/n)==h;hashes[n]=h
    for arm in M.ARMS:
        old=M.RUN/arm/'raw'/(p['probe_video']+'.npz');hashes[old.relative_to(ROOT).as_posix()]=sha(old)
    p={**p,'id':IDENT,'created':now().isoformat(),'device':'cpu','budget_seconds':3600,'stop_request_path':(STAGE/'stop.request').relative_to(ROOT).as_posix(),'sha256':hashes,'comparison':'Raw4 only. Same fixed20,frames,detections,annotation mapping,TrackEval;no MASA/FlyVis. No full-system claim or outcome-based tuning. All pairs versus XYWH are reported with paired-video95%CI;only strict lower>0 supports that exact raw comparison.'}
    write(STAGE/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);next(x for x in cat['experiments'] if x['id']==IDENT)['state']='running';write(cp,cat)


def one(arm,v,probe=False):
    import cv2,numpy as np
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from tracking.offline_association import _DetSlice
    p=read(STAGE/'protocol.json');cv2.setNumThreads(2);folder=STAGE/('probe' if probe else arm)/('raw' if not probe else arm);marker=STAGE/'jobs'/((('probe_'+arm) if probe else arm+'_'+v)+'.json')
    if marker.exists():
        d=read(marker)
        for n,h in d['outputs'].items():assert sha(ROOT/n)==h
        return
    base=ROOT/f'runs/video_rate/{v}' if probe else RUN/'cache'/v;dets=load_detection_cache(base/'detections/raw_video.npz');alignment=read(base/'alignment.json');truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');anchors={int(row['video_frame_index']):i for i,row in enumerate(alignment['matches'])};assert len(anchors)==len(truth)
    cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{v}.mov'));assert cap.isOpened();tracker=M.make_tracker(arm,p['configs']);raw=[];rows=[];start=time.monotonic()
    try:
        for f in range(p['frames'][v]):
            if f%25==0:status('Контроль4трекеров' if probe else 'Исходный трекер,без сшивки',variant=arm+' '+v,done=f,total=p['frames'][v])
            ok,img=cap.read();assert ok;xyxy,conf,cls=detections_for_frame(dets,f);out=tracker.update(_DetSlice(xyxy,conf,cls),img=img)
            if len(out):raw.append(np.column_stack([np.full(len(out),f),np.asarray(out)[:,:7]]).astype(np.float32))
            if f in anchors:rows.append(scalabel_frame(v,truth[anchors[f]],out))
        assert not cap.read()[0]
    finally:cap.release()
    assert len(rows)==len(truth);a=np.concatenate(raw) if raw else np.zeros((0,8),np.float32);folder.mkdir(parents=True,exist_ok=True)
    if probe:
        with np.load(M.RUN/arm/'raw'/(v+'.npz')) as z:assert np.array_equal(a,z['outputs']) and int(z['frame_count'])==p['frames'][v],('Raw replay',arm)
    np.savez_compressed(folder/(v+'.npz'),outputs=a,frame_count=p['frames'][v]);write(folder/(v+'.json'),rows)
    write(marker,dict(arm=arm,video=v,probe=probe,seconds=time.monotonic()-start,frames=p['frames'][v],annotated_frames=len(rows),outputs={q.relative_to(ROOT).as_posix():sha(q) for q in (folder/(v+'.npz'),folder/(v+'.json'))}))


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(STAGE/'protocol.json');verify_sources(p);per={a:{} for a in M.ARMS};pooled={}
    for a in M.ARMS:one(a,p['probe_video'],True)
    for i,v in enumerate(p['videos']):
        began=time.monotonic()
        for arm in M.ARMS:
            one(arm,v);status('Официальная HOTA отдельного видео',variant=arm+' '+v);per[arm][v]=evaluate_hota(STAGE/arm/'raw',[v],split='val');write(STAGE/arm/'per_video.json',per[arm])
        if i==0:
            wall=time.monotonic()-began;estimate=2*wall*19+120;remaining=p['budget_seconds']-(time.monotonic()-T0);write(STAGE/'price.json',dict(first_video_all4_seconds=wall,conservative_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate<remaining));assert estimate<remaining,'Raw20 remaining price does not fit'
    for arm in M.ARMS:check();pooled[arm]=evaluate_hota(STAGE/arm/'raw',p['videos'],split='val')
    pv={a:{v:x['HOTA'] for v,x in rows.items()} for a,rows in per.items()};contrasts={a+'-xywh':paired(pv[a],pv['xywh']) for a in M.ARMS if a!='xywh'};verify_sources(p)
    write(STAGE/'report.json',dict(videos=p['videos'],pooled=pooled,per_video=per,paired=contrasts,raw_only=True,no_flyvis_or_masa=True,limitations=p['limitations']));write(STAGE/'status.json',dict(state='completed',stage='4исходных трекера на20val готовы;безMASA/D',updated=now().isoformat(),jobs_done=84,jobs_total=84))


if __name__=='__main__':
    if '--register' in sys.argv:register()
    elif '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit raw20 mode required')
