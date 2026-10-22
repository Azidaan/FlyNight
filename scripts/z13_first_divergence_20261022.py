"""Read-only instrumented prefix replay; never changes original scientific files."""
from pathlib import Path
import sys
import os
import time
import copy
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_first_divergence_20261022'
OLD=ROOT/'runs/z13_center_hota_20261021'
VIDEO='0355243b-d696291a';LAST=191;IDENT='z13_first_divergence_v1'


def journal(title,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf-8');mark='## Журнал изменений\n';assert s.count(mark)==1
    p.write_text(s.replace(mark,mark+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing diagnostic must not be overwritten')
    assert now().hour==7 and now().minute<30,'Insufficient morning window'
    RUN.mkdir()
    files=['scripts/z13_first_divergence_20261022.py','scripts/z13_night_control_20261021.py',
           'scripts/build_fly_signatures.py','tracking/z13_ridge_localizer.py','tracking/z13_ridge_localizer_v2.py',
           'tracking/xywh_bytetrack.py','tracking/foveated_eyes.py','tracking/retina.py',
           '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
           f'runs/train_cache/{VIDEO}/detections/raw_video.npz',f'runs/train_cache/{VIDEO}/alignment.json',
           f'runs/fly_signatures/{VIDEO}/signatures.npz',f'runs/z13_center_hota_20261021/R1/signatures/{VIDEO}/signatures.npz',
           'runs/z13_center_hota_20261021/R1/config.json','runs/z13_center_hota_20261021/eval/report.json',
           f'data/bdd100k/videos/train/{VIDEO}.mov',f'data/bdd100k/labels/box_track_20/train/{VIDEO}.json']
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author: trace first differing association; autonomous permitted window until07:40MSK22.10.',
           video=VIDEO,arm='R1',first_saved_difference=183,last_replayed_frame=LAST,budget_seconds=360,
           save_at='2026-10-22T07:35:00+03:00',hard_stop='2026-10-22T07:39:30+03:00',
           selection='Largest negative R1 HOTA on completed dev35,then first unequal raw output frame; post-hoc diagnostic only.',
           action='Observe existing center policy and associations; no model/threshold/data change, no training, no new HOTA.',
           acceptance=['Stand and R1 raw outputs bit-exact to saved originals through191 inclusive.',
                       'Save predicted and corrected lost-track boxes,confidence,full association costs and assignments at183.',
                       'Compare with nearby real annotated frames179,185,191,explicitly no GT at183; no interpolated truth.',
                       'Claim only this reproduced event; no attribution of whole35-video loss.'],
           counterargument='First changed association can be harmless; failure to recover one detection is not necessarily wrong identity or the source of total HOTA loss.',
           sha256={n:sha(ROOT/n) for n in files})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Подготовка повторения кадров0–191',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: первое расходящееся сопоставление в ухудшившемся R1',area='flyvis',state='ready',started='2026-10-22',done='',
        tags=['fly-problem-P8'],question='Как поправка центра связана с первым расхождением в ухудшившемся видео?',hypothesis='Возможна неверная поправка перед сопоставлением; это пока не установленная причина.',
        expected='Точное повторение начала видео и наблюдение существующей политики; без обучения/подбора/новойHOTA.',
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],summary='Диагностика зарегистрирована до повторения.',obtained='',interpretation='',decision='',
        cannot_claim='Разбор выбран после полного результата; один случай не объясняет все потери и не измеряет новое улучшение. На183 нет реальной разметки; соседние кадры не подменяют её.',
        evidence=[dict(kind='file',ref='runs/z13_first_divergence_20261022/protocol.json',claim='Зафиксированный префикс,проверка воспроизведения и срок')],metrics=[],commits=[],files=['scripts/z13_first_divergence_20261022.py']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка разбора первого расхождения','*Постановка.* Автор разрешил разбор. По сохранённым outputs первое отличие R1 от стенда на видео0355243b-d696291a — кадр183, отсутствует трек6; видео выбрано по наибольшему падению R1, поэтому это диагностический пример, не независимая проверка. Повтор0–191 включительно, точное воспроизведение обоих сохранённых выходов, фиксация центра/балла/стоимостей/назначений. Реальная разметка рядом179/185/191; на183 её нет, интерполяция не используется. Бюджет360с, независимый контроллер до07:35/07:39:30. Старые источники не редактируются, готовые полные этапы не повторяются, новых HOTA/обучения нет.')


def worker():
    import numpy as np
    from scripts import build_fly_signatures as builder
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking import xywh_bytetrack as xy
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers.utils import matching
    original=xy.XYWHByteTrack
    protocol=read(RUN/'protocol.json')
    for n,h in protocol['sha256'].items():assert sha(ROOT/n)==h,n
    cache=load_detection_cache(ROOT/f'runs/train_cache/{VIDEO}/detections/raw_video.npz')
    with np.load(ROOT/f'runs/fly_signatures/{VIDEO}/signatures.npz') as z:stand_saved=z['outputs'].copy()
    with np.load(OLD/f'R1/signatures/{VIDEO}/signatures.npz') as z:r1_saved=z['outputs'].copy()
    traces={};active='stand';frame=-1;T0=time.monotonic()
    def check():
        if time.monotonic()-T0>340 or (RUN/'trace.stop').exists() or now().isoformat()>protocol['save_at']:raise TimeoutError('Diagnostic budget/window')
    def snap(t):
        return dict(id=int(t.track_id),state=int(t.state),mean=t.mean.tolist() if t.mean is not None else None,box=np.asarray(t.xyxy).tolist(),cls=int(t.cls),score=float(t.score),idx=int(t.idx))
    def trace_assoc(tracker,pool,detections):
        cost=tracker.get_dists(pool,detections);m,ut,ud=matching.linear_assignment(cost,thresh=tracker.args.match_thresh)
        return dict(pool=[snap(t) for t in pool],detections=[snap(t) for t in detections],cost=cost.tolist(),threshold=float(tracker.args.match_thresh),matches=np.asarray(m).tolist(),unmatched_tracks=list(map(int,ut)),unmatched_detections=list(map(int,ud)))
    def compare(output,saved,f):
        actual=np.column_stack([np.full(len(output),f),np.asarray(output)[:,:7]]).astype(np.float32) if len(output) else np.empty((0,8),np.float32)
        expected=saved[saved[:,0]==f]
        assert np.array_equal(actual,expected),f'{active} prefix mismatch at frame{f}'
    class TraceBase(original):
        def _pre_first_associate(self,pool,unconfirmed,img,results_high):
            super()._pre_first_associate(pool,unconfirmed,img,results_high)
            if frame>=175:traces.setdefault(str(frame),{}).setdefault(active,{})['before']=trace_assoc(self,pool,self.init_track(results_high,img))
        def _first_association(self,pool,detections,activated,refind):
            if frame>=175:traces.setdefault(str(frame),{}).setdefault(active,{})['applied']=trace_assoc(self,pool,detections)
            return super()._first_association(pool,detections,activated,refind)
        def update(self,*args,**kwargs):
            nonlocal frame
            check();frame+=1;output=super().update(*args,**kwargs)
            compare(output,stand_saved if active=='stand' else r1_saved,frame)
            if frame%25==0:write(RUN/'status.json',dict(state='running',stage=active,done=frame,total=LAST,updated=now().isoformat()))
            if active=='R1' and frame==LAST:raise PrefixFinished()
            return output
    BaseTrack.reset_id();tracker=TraceBase(make_tracker_args(**builder.tracker_config(None)))
    for _ in range(LAST+1):
        boxes,conf,cls=detections_for_frame(cache,frame+1);tracker.update(_DetSlice(boxes,conf,cls))
    write(RUN/'stand_prefix_verified.json',dict(frames=LAST+1,bit_exact=True))
    active='R1';frame=-1
    from tracking.z13_compiled_backend import setup
    setup()
    from tracking import z13_readout
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    class TraceR1(RecoveredCenterRidgeLocalizer):
        def advance(self,eyes,current):
            answer=super().advance(eyes,current)
            if frame>=175:
                by={id(e):tid for tid,e in self.eyes.items()}
                traces.setdefault(str(frame),{}).setdefault(active,{}).setdefault('responses',[]).extend(dict(id=by[id(e)],eye_center=np.asarray(a[0]).tolist(),confidence=float(a[1]),tau=self.tau) for e,a in zip(eyes,answer))
            return answer
    class PrefixFinished(Exception):pass
    xy.XYWHByteTrack=TraceBase;z13_readout.Localizer=TraceR1
    saved=sys.argv
    try:
        sys.argv=['builder','--video-id',VIDEO,'--z13',str(OLD/'R1/config.json'),'--output',str(RUN/'prefix_output'),'--max-eyes','4']
        args=builder.parse_args();args.skip_fly=True
        from tracking.gpu_lock import GpuLock
        with GpuLock('flyvis_fovea_gpu'):
            try:builder.build_video(VIDEO,args)
            except PrefixFinished:pass
        assert frame==LAST
    finally:sys.argv=saved
    for n,h in protocol['sha256'].items():assert sha(ROOT/n)==h,n
    write(RUN/'trace.json',dict(video=VIDEO,prefix_bit_exact_stand=True,prefix_bit_exact_R1=True,frames=LAST+1,traces=traces,protocol_sha256=sha(RUN/'protocol.json')))
    write(RUN/'status.json',dict(state='completed',stage='Оба префикса воспроизведены точно; назначения сохранены',updated=now().isoformat()))
    print('Both prefixes bit-exact; trace saved',flush=True)


def controller():
    source=(ROOT/'scripts/z13_night_control_20261021.py').read_text(encoding='utf-8')
    assert '2026,10,21' in source
    source=source.replace('2026,10,21','2026,10,22')
    sys.argv=['controller','--run',str(RUN),'--script',str(Path(__file__).resolve()),'--label','trace','--budget','360']
    scope=dict(__name__='prefix_controller',__file__=str(ROOT/'scripts/z13_night_control_20261021.py'))
    exec(compile(source,'prefix_controller','exec'),scope);return scope['main']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--control' in sys.argv:sys.exit(controller())
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:
            write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit prepare/control/worker required')
