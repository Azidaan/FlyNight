"""Адаптация BUSCA только для дополнительной выдачи; полный сопоставимый пилот."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/busca_output_pilot'
OLD=ROOT/'runs/z13_val20_matched_20261024/raw'
MSK=timezone(timedelta(hours=3));IDENT='busca_output_recovery_pilot_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists()
    profile=ROOT/'runs/b123_20261027/busca_runtime_profile/recovery1'
    assert read(profile/'report.json')['complete'] and read(profile/'controller_result.json')['all_owned_ended']
    assert read(profile/'controller_result.json')['exit_code']==0
    old=read(OLD/'protocol.json');videos=old['videos']
    sources=['scripts/b123_busca_output_pilot_20261027.py','tracking/busca_runtime.py','tracking/busca_output_recovery.py',
        'tracking/xywh_bytetrack.py','scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py',
        'weights/busca_official_20261027/model_busca.pth','runs/b123_20261027/recovery_motion_control/report.json',
        'runs/b123_20261027/busca_runtime_profile/recovery1/report.json']
    sources += list(read(profile/'protocol.json')['sha256'])
    for video in videos:
        sources += [f'runs/z13_val20_matched_20261024/cache/{video}/detections/raw_video.npz',
            f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json',
            f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz',
            f'data/bdd100k/videos/val/{video}.mov',f'data/bdd100k/labels/box_track_20/val/{video}.json']
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),author_stop='2026-10-27T07:20:00+03:00',budget_seconds=5400,
        videos=videos,frames=old['frames'],configs=old['configs'],device='cuda',arms=['baseline','busca_output'],
        maximum_lost_age_original_frames=30,memory_length=11,batch_size=4,
        action='Не меняя назначений и состояния XYWH, дополнять только выдачу текущими lostKalman, если авторская сеть BUSCA выбирает собственный прогноз вместо соседей и вариантов отказа. Пять кандидатов, один из них прогноз; полная память обязательна. Порог исходного решения0.5 при one-hot лучшего кандидата, без подбора.',
        adaptation='Это собственная адаптация обученной BUSCA к выходной ветви XYWH, НЕ воспроизведение полного авторского трекера. Нет обновления трекера после принятия, компенсации камеры и фильтра плотности. Память ограничена последними11уверенными реальными назначениями (score>=0.25), без broader memory. Пакеты до4объектов фиксированы; BatchNorm адаптируется как у авторов, поэтому разбиение пакетов может менять представления. Решение только на оцениваемых кадрах; история наблюдается на всех исходных кадрах. Не использовать это как опровержение полной BUSCA.',
        cost_gate='После первого полного видео:2*его полная стоимость*19+120с должны помещаться в оставшиеся5400с и до07:20; иначе остановить без полного запуска. Первое видео не повторять. Параметры по его результату не менять.',
        acceptance=['Все20видео, точное побайтное воспроизведение исходных массивов XYWH и HOTA базы',
            'Память строго из прошлого; реальная авторская сеть, проверенные веса и нормировка; нет меток и будущего во входе',
            'HOTA/AssA/IDF1/FP/FN/IDSw и парный95%интервал по видео; для утверждения о пользе нижняя граница разности HOTA>0'],
        limitations='Повторно использованные20val, не независимый итоговый тест. Авторские веса перенесены без дообучения на BDD100K. Это адаптация, не результат исходной статьи. Вклад ядра FlyVis здесь не измеряется.',
        sha256={name:sha(ROOT/name) for name in sorted(set(sources))})
    write(RUN/'protocol.json',p)
    catalog=ROOT/'experiments/project_history.json';raw=catalog.read_bytes();data=json.loads(raw)
    assert not any(c['id']==IDENT for c in data['experiments'])
    data['experiments'].append(dict(id=IDENT,title='Восстановление потерянных объектов — проверять дополнительную выдачу сетью BUSCA',area='flyvis',state='ready',started='2026-10-27',
        question='Поможет ли обученный отказ BUSCA отфильтровать вредные продолжения без изменения исходного трекера?',
        hypothesis='История внешности и варианты отказа могут отделить полезный прогноз от ложной рамки; знак HOTA заранее неизвестен.',
        expected=p['action'],summary=p['adaptation'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],
        obtained='',interpretation='',decision='',cannot_claim=p['limitations']+' '+p['adaptation'],metrics=[],commits=[],files=sources[:3],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Фиксированное действие,контроль,адаптация,стоимостьикритериидоизмерения')]))
    assert catalog.read_bytes()==raw;write(catalog,data)
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT=IDENT;J.journal('поставлен полный пилот выходной ветви с решением BUSCA',
        '*Постановка.* '+p['action']+'\n\n*Отличие от статьи.* '+p['adaptation']+'\n\n'
        '*Условия.* Те же20val, детекции и XYWH, без MASA. Контроль простого продолжения уже сохранён; база повторяется точно. '
        '5400с на одинGPU, независимый контроллер, измеренная стоимость первого полного видео определяет продолжение без изменения порогов. '
        '*Принятие.* '+p['acceptance'][-1]+'. '+p['limitations']+' '
        'Источник: `runs/b123_20261027/busca_output_pilot/protocol.json`. HOTA ещё не измерена.')
    print('Prepared full adaptation pilot; validate before queue publication')
def run():
    import torch,numpy as np,cv2
    torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.busca_runtime import load_model
    from tracking.busca_output_recovery import BuscaOutputRecovery,valid_box
    from tracking.offline_association import _DetSlice
    from ultralytics.trackers.basetrack import BaseTrack
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');start=time.monotonic()
    deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    def check():
        if (RUN/'stop.request').exists() or time.time()>deadline-5:raise TimeoutError('Own stage/author deadline')
    def status(stage,**kw):
        check();write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    per={a:{} for a in p['arms']};timing={};exact={};counters={}
    for index,video in enumerate(p['videos']):
        check();began=time.monotonic();BaseTrack.reset_id()
        # Reset author weights and running statistics for every independent video.
        status('Загрузка фиксированной BUSCA для видео',video=video,video_index=index,total_videos=len(p['videos']))
        model,coverage=load_model('cuda')
        branch=BuscaOutputRecovery(model,p['configs']['xywh']['track_high_thresh'],p['memory_length'],p['batch_size'])
        tracker=XYWHByteTrack(SimpleNamespace(**p['configs']['xywh']))
        dets=load_detection_cache(ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/detections/raw_video.npz')
        align=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json')['matches']
        # Truth is passed only to output formatting / scorer; never to branch.
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
        anchors={int(row['video_frame_index']):i for i,row in enumerate(align)};assert len(anchors)==len(truth)
        cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{video}.mov'));assert cap.isOpened()
        rows={a:[] for a in p['arms']};raw=[];added=0
        for frame in range(p['frames'][video]):
            if frame%30==0:status('Неизменный XYWH и решение о продолжении',video=video,done=frame,total=p['frames'][video],model_calls=branch.calls,added=added)
            ok,image=cap.read();assert ok,(video,frame)
            xyxy,conf,classes=detections_for_frame(dets,frame)
            output=tracker.update(_DetSlice(xyxy,conf,classes))
            base=np.asarray(output,dtype=np.float32).reshape(-1,8)
            if len(base):raw.append(np.column_stack([np.full(len(base),frame),base[:,:7]]).astype(np.float32))
            if frame in anchors:
                ids={int(r[4]) for r in base}
                lost=[track for track in tracker.lost_stracks if track.track_id not in ids and track.is_activated
                    and 1<=tracker.frame_id-track.end_frame<=p['maximum_lost_age_original_frames'] and valid_box(track.xyxy,image.shape)]
                selected=branch.select(frame,lost,image,xyxy,conf,p['configs']['xywh']['track_low_thresh'])
                extra=np.asarray([[*track.xyxy,track.track_id,track.score,track.cls,-1] for track in selected],dtype=np.float32).reshape(-1,8)
                extended=np.concatenate([base,extra]) if len(extra) else base.copy();added+=len(extra)
                for arm,out in [('baseline',base),('busca_output',extended)]:rows[arm].append(scalabel_frame(video,truth[anchors[frame]],out))
            branch.remember(frame,base,image)
            branch.forget_removed({track.track_id for track in tracker.tracked_stracks+tracker.lost_stracks})
        cap.release()
        actual=np.concatenate(raw) if raw else np.zeros((0,8),np.float32)
        with np.load(OLD/'xywh/raw'/(video+'.npz')) as z:assert np.array_equal(actual,z['outputs']),('Baseline array mismatch',video)
        exact[video]=True
        for arm in p['arms']:
            assert len(rows[arm])==len(truth);write(RUN/arm/(video+'.json'),rows[arm])
            status('Оценка полного видео',video=video,arm=arm)
            per[arm][video]=evaluate_hota(RUN/arm,[video],split='val')
            write(RUN/arm/'per_video.json',per[arm])
        ref=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json')['per_video']['baseline'][video]
        assert per['baseline'][video]['HOTA']==ref['HOTA'],('Baseline HOTA mismatch',video)
        counters[video]=dict(calls=branch.calls,considered=branch.considered,accepted=branch.accepted,added=added,strict_past=True)
        timing[video]=time.monotonic()-began
        write(RUN/'completed_videos.json',dict(videos=list(exact),exact=exact,counters=counters,timing=timing))
        del branch,model;torch.cuda.empty_cache()
        if index==0:
            required=2*timing[video]*19+120;available=deadline-time.time()
            write(RUN/'price.json',dict(first_video_seconds=timing[video],conservative_remaining_seconds=required,available_seconds=available,proceed=required<available))
            assert required<available,'Measured conservative cost exceeds original budget; no full continuation'
    pooled={}
    for arm in p['arms']:
        status('Общая оценка20видео',arm=arm);pooled[arm]=evaluate_hota(RUN/arm,p['videos'],split='val')
    control=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json')
    assert pooled['baseline']['HOTA']==control['pooled']['baseline']['HOTA']
    contrasts={name:paired({v:m['HOTA'] for v,m in per['busca_output'].items()},
        {v:m['HOTA'] for v,m in ref.items()}) for name,ref in [('vs_baseline',per['baseline']),('vs_simple_continuation',control['per_video']['emit_lost_kalman'])]}
    write(RUN/'report.json',dict(complete=True,videos=p['videos'],pooled=pooled,per_video=per,paired=contrasts,
        simple_control_pooled=control['pooled']['emit_lost_kalman'],baseline_arrays_exact=exact,counters=counters,
        timing=timing,seconds=time.monotonic()-start,hota_measured=True,original_busca_reproduced=False,
        tracker_state_changed=False,flyvis_core_used=False,adaptation=p['adaptation'],limitations=p['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='Полная HOTA адаптации сохранена',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
