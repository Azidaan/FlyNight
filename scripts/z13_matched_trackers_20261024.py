"""Common video/detection/evaluation protocol; native tracker settings frozen."""
from pathlib import Path
from datetime import datetime,timedelta
from types import SimpleNamespace
import os,sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from scripts.z13_candidate_reserve_20261021 import read,write,sha
from scripts.z13_localization_labels_20261023 import now
RUN=ROOT/'runs/z13_matched_trackers_20261024';IDENT='z13_matched_trackers_v1';T0=time.monotonic()
ARMS=('xywh','bytetrack','botsort','tracktrack')
VIDEOS=('b1c66a42-6f7d68ca','b1c81faa-3df17267','b1cac6a7-04e33135','b1cebfb7-284f5117','b1d22449-15fb948f','b232c7c9-d251d9ee','b251064f-e7a165fd')


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as p
    old=p.IDENT
    try:p.IDENT=IDENT;p.journal(title,text)
    finally:p.IDENT=old


def configurations():
    import yaml
    from ultralytics.trackers import track as ut
    from tracking.offline_association import make_tracker_args
    cfg=Path(ut.__file__).resolve().parents[1]/'cfg/trackers';common=dict(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,fuse_score=True)
    configs={'xywh':vars(make_tracker_args(**common))}
    for name in ARMS[1:]:
        c=yaml.safe_load((cfg/(name+'.yaml')).read_text(encoding='utf-8'));c.update(common)
        if name in ('botsort','tracktrack'):c['with_reid']=False
        configs[name]=c
    return configs


def make_tracker(arm,configs):
    from ultralytics.trackers.basetrack import BaseTrack
    BaseTrack.reset_id()
    if arm=='xywh':
        from tracking.xywh_bytetrack import XYWHByteTrack
        cls=XYWHByteTrack
    else:
        from ultralytics.trackers.track import TRACKER_MAP
        cls=TRACKER_MAP[arm]
    return cls(SimpleNamespace(**configs[arm]))


def check():
    p=read(RUN/'protocol.json')
    if (RUN/'raw.stop').exists() or time.monotonic()-T0>=3600 or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('CPU raw comparison budget/author deadline')


def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not RUN.exists();configs=configurations();trainfile=ROOT/'training/configs/generated/full_step5_train.txt';text=trainfile.read_text(encoding='utf-8');assert all(v not in text for v in VIDEOS)
    names=['scripts/z13_matched_trackers_20261024.py','tests/test_z13_matched_trackers.py','scripts/z13_matched_trackers_console.ps1','tracking/xywh_bytetrack.py','tracking/offline_association.py','evaluation/hota_eval.py','scripts/run_bdd_video_rate_experiment.py','scripts/build_fly_signatures.py','scripts/stitch_eval_v2.py','scripts/z13_shared_masa.py','weights/baseline_seed42/best.pt','training/configs/generated/full_step5_train.txt','runs/z13_center_hota_20261021/masa_head.joblib','runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_center_hota_20261021/D/config.json']
    from ultralytics.trackers import track as ut
    cfg=Path(ut.__file__).resolve().parents[1]/'cfg/trackers'
    for arm in ARMS[1:]:names += [(cfg/(arm+'.yaml')).relative_to(ROOT).as_posix()]
    import inspect
    for arm in ARMS:
        tr=make_tracker(arm,configs);names.append(Path(inspect.getfile(type(tr))).relative_to(ROOT).as_posix())
    frames={}
    for v in VIDEOS:
        base=ROOT/'runs/video_rate'/v;meta=read(base/'detections/raw_video.json');assert meta['identity']['weights_sha256']==sha(ROOT/'weights/baseline_seed42/best.pt') and meta['identity']['imgsz']==960 and meta['identity']['conf']==.01
        frames[v]=meta['frame_count']
        for n in ('detections/raw_video.npz','detections/raw_video.json','alignment.json'):names.append((base/n).relative_to(ROOT).as_posix())
        names += [f'data/bdd100k/videos/val/{v}.mov',f'data/bdd100k/labels/box_track_20/val/{v}.json',f'runs/fly_signatures_val/{v}/signatures.npz',f'runs/masa_embeddings/{v}.npz']
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0)
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author explicitly requested results in identical conditions; autonomous necessary runs/publication authorized.',device='cpu',budget_seconds=3600,save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),stop_request_path=(RUN/'raw.stop').relative_to(ROOT).as_posix(),videos=list(VIDEOS),frames=frames,configs=configs,
      comparison='All7existing val videos, same original frames and960/conf.01 YOLO cache, same annotation alignment and official TrackEval. Primary total-system comparison includes native ByteTrack,BoT-SORT,TrackTrack,XYWHByteTrack,XYWH+legacyD,XYWH+centerD. Show without and with identical frozen base+MASA stitcher (gap150,dist5,threshold.6). No values from old200val or35train-derived dev in this table.',
      settings='Common high.25,low.1,new.25,buffer30,fuse_scoreTrue. Each native algorithm retains its documented association-cost settings and GMC; these costs have different meanings, so a common numeric match threshold is not imposed. ReID disabled in all native trackers; identical MASA postprocessing is reported separately for every system. These fixed settings are not claimed optimal.',
      technical='Replay XYWH raw outputs exactly against existing val7 signatures; ensure every original frame consumed and every annotated frame emitted, identical detection hashes, labels only in evaluation/metadata alignment. Native tracker raw outputs independently replayed when generating appearance signatures. Validate XYAH->XYWH state conversion only in signature adapter, never change native tracking states.',
      acceptance=['All7videos and original cached inputs/metadata unchanged; no detector-training video overlap.','No-op XYWH output reproduced, every frame consumed and every annotation evaluated; all6systems both raw/MASA complete.','For any efficacy claim: paired-video95%CI lower>0 for that exact matched contrast; report all outcomes including unfavorable ones.'],
      stages='CPU raw4trackers<=3600s; then fixed first-video GPU price/reproduction, retain completed video; proceed with twoD/signatures/sharedMASA only if conservative projection fits remaining author window. Independent bounded controllers and visible consoles. Completed stages not repeated except explicit replay verification.',
      limitations='Only7previously used val videos; not independent final test and no75heldout. Detector validation set used in its training monitoring, but these videos are absent from gradient-training list. Settings not optimized per tracker; cannot infer universal ranking or compare directly with published benchmarks. The old MASA head was trained on XYWH life pairs and can transfer differently to other trackers; report raw/MASA separately.',sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';c=read(cp);assert not any(x['id']==IDENT for x in c['experiments'])
    c['experiments'].append(dict(id=IDENT,title='З13: трекеры и FlyVis в единых условиях на7val',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','benchmark'],question='Каковы HOTA внешних трекеров,стенда и двух D при одинаковых видео,детекциях,частоте,оценке и сшивке?',hypothesis='Часть прежнего видимого преимущества могла относиться к различию протоколов. Знак результата новой общей проверки не предполагается.',expected=p['comparison'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],summary='Общие условия и все сравнения зафиксированы до нового результата.',done='',obtained='',interpretation='',decision='',cannot_claim=p['limitations'],metrics=[],commits=[],files=['scripts/z13_matched_trackers_20261024.py'],evidence=[dict(kind='file',ref='runs/z13_matched_trackers_20261024/protocol.json',claim='Общие условия и порядок до измерения'),dict(kind='file',ref='reports/stand_hota_comparability_20261024.json',claim='Предыдущая таблица не была единым рейтингом')]))
    write(cp,c);journal('зарегистрировано общее сравнение трекеров','*Постановка.* Автор: «Хочу результаты на одинаковых условиях». '+p['comparison']+'\n\n'+p['settings']+'\n\nСначала CPU-прогон4базовых трекеров на всех7val и точное воспроизведение XYWH. Затем цена GPU/сигнатур/D и общая сшивка при достаточном остатке окна. Сохранение07:35,конец07:39:30MSK24.10;CPU3600с. Новые данные/75отложенных/подбор по val исключены. Ограничения: '+p['limitations']+'\n\nЧисла пока не получены; прежние200val/35dev в новую таблицу не переносятся.')


def worker():
    import cv2,numpy as np
    cv2.setNumThreads(2)
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from tracking.offline_association import _DetSlice
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p);durations={};per={};pooled={};parity={};done=0
    for arm in ARMS:
        for v in VIDEOS:
            folder=RUN/arm/'raw';marker=folder/(v+'.done.json')
            if marker.exists():
                m=read(marker);assert sha(folder/(v+'.json'))==m['json_sha256'] and sha(folder/(v+'.npz'))==m['raw_sha256'];done+=1;continue
            detections=load_detection_cache(ROOT/f'runs/video_rate/{v}/detections/raw_video.npz');alignment=read(ROOT/f'runs/video_rate/{v}/alignment.json');truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');anchors={int(row['video_frame_index']):i for i,row in enumerate(alignment['matches'])};assert len(anchors)==len(truth)
            cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{v}.mov'));assert cap.isOpened();tracker=make_tracker(arm,p['configs']);rows=[];raw=[];began=time.monotonic();tracktime=0.
            try:
                for f in range(p['frames'][v]):
                    if f%25==0:status('Общее исходное видео, без сшивки',variant=arm+' '+v,done=f,total=p['frames'][v],jobs_done=done,jobs_total=28)
                    ok,image=cap.read();assert ok,(v,f);xyxy,conf,cl=detections_for_frame(detections,f);start=time.monotonic();out=tracker.update(_DetSlice(xyxy,conf,cl),img=image);tracktime+=time.monotonic()-start
                    if len(out):raw.append(np.column_stack([np.full(len(out),f),np.asarray(out)[:,:7]]).astype(np.float32))
                    if f in anchors:rows.append(scalabel_frame(v,truth[anchors[f]],out))
                assert not cap.read()[0],'Video contains frames missing from detector cache'
            finally:cap.release()
            assert len(rows)==len(truth);a=np.concatenate(raw) if raw else np.zeros((0,8),np.float32);folder.mkdir(parents=True,exist_ok=True)
            if arm=='xywh':
                with np.load(ROOT/f'runs/fly_signatures_val/{v}/signatures.npz') as z:old=z['outputs'];assert int(z['frame_count'])==p['frames'][v]
                assert np.array_equal(a,old),('XYWH replay differs',v);parity[v]=True
            temp=folder/(v+'.next.npz');np.savez_compressed(temp,outputs=a,frame_count=p['frames'][v]);temp.replace(folder/(v+'.npz'));write(folder/(v+'.json'),rows)
            write(marker,dict(frames=p['frames'][v],annotated_frames=len(rows),seconds=time.monotonic()-began,tracker_seconds=tracktime,raw_sha256=sha(folder/(v+'.npz')),json_sha256=sha(folder/(v+'.json'))));done+=1
        status('Официальная HOTA всех7видео',variant=arm,jobs_done=done,jobs_total=28)
        pooled[arm]=evaluate_hota(RUN/arm/'raw',list(VIDEOS),split='val');per[arm]={}
        for v in VIDEOS:
            check();per[arm][v]=evaluate_hota(RUN/arm/'raw',[v],split='val');write(RUN/arm/'per_video.json',per[arm])
        write(RUN/arm/'metrics.json',pooled[arm]);durations[arm]={v:read(RUN/arm/'raw'/(v+'.done.json')) for v in VIDEOS}
    pv={arm:{v:x['HOTA'] for v,x in m.items()} for arm,m in per.items()};contrasts={arm+'-xywh':paired(pv[arm],pv['xywh']) for arm in ARMS if arm!='xywh'};verify_sources(p)
    write(RUN/'raw_report.json',dict(videos=list(VIDEOS),pooled=pooled,per_video=per,paired=contrasts,xywh_exact_replay=parity,timing=durations,not_full_pipeline_result=True,limitations=p['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='Исходные4трекера готовы;D/MASA ещё не итог',updated=now().isoformat(),jobs_done=28,jobs_total=28))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
