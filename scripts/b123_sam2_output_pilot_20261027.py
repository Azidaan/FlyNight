"""Полный пилот собственной масочной выдачи; не воспроизведение LTR."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_output_pilot';OLD=ROOT/'runs/z13_val20_matched_20261024/raw'
IDENT='sam2_mask_output_recovery_pilot_v1';MSK=timezone(timedelta(hours=3))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists()
    profile=ROOT/'runs/b123_20261027/sam2_runtime_profile'
    assert read(profile/'report.json')['complete'] and read(profile/'controller_result.json')['all_owned_ended']
    assert read(profile/'controller_result.json')['exit_code']==0
    old=read(OLD/'protocol.json');sources=['scripts/b123_sam2_output_pilot_20261027.py','tracking/sam2_mask_recovery.py',
        'tracking/xywh_bytetrack.py','scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py',
        'runs/b123_20261027/busca_output_pilot/report.json','runs/b123_20261027/recovery_motion_control/report.json']
    sources+=list(read(profile/'protocol.json')['sha256'])
    for video in old['videos']:
        sources += [f'data/bdd100k/videos/val/{video}.mov',f'data/bdd100k/labels/box_track_20/val/{video}.json',
            f'runs/z13_val20_matched_20261024/cache/{video}/detections/raw_video.npz',
            f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json',f'runs/z13_val20_matched_20261024/raw/xywh/raw/{video}.npz']
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),budget_seconds=7200,author_stop='2026-10-27T07:20:00+03:00',
        device='cuda',videos=old['videos'],frames=old['frames'],configs=old['configs'],arms=['baseline','sam2_mask'],max_lost_age=30,
        action='Сохранять исходный XYWH без изменений. После потери инициализировать SAM2 последней реально назначенной рамкой в прошлом кадре, причинно переносить маску до30исходныхкадров. Дополнять только выдачу ограничивающей рамкой маски. Отказ при object_score<=0, пустой маске или IoU>0.9 с текущим выходом. После отказа ждать реального повторного назначения.',
        adaptation='Собственная масочная ветвь SAM2.1tiny, НЕ LTR(SAM+27масок+обученныйселектор+DeAOT) и НЕ SGAR-MOT. Нативная оценка присутствия не доказывает личность объекта. Нет обучения на BDD100K, C++постобработки или изменения состояния исходного трекера. Штатный1024вход/нормировка, FP32; официальный код неизменён, ввод кадров ограничен прошлым. Решение и память обновляются на каждом исходном кадре, оценка только по прежним привязкам.',
        cost_gate='После первого полного видео:2*его стоимость*19+120с должны помещаться в оставшиеся7200с и до07:20. Иначе остановить, исходный бюджет не продлевать, готовое видео не повторять.',
        acceptance=['Все20видео и точное воспроизведение исходных массивов XYWH, базовой HOTA и привязок',
                    'Ни одного будущего кадра или метки во входе ветви; одинаковые веса и фиксированный отказ',
                    'HOTA/AssA/IDF1/FP/FN/IDSw и парный95%интервал по видео; нижняя граница разности HOTA>0 для утверждения о пользе'],
        limitation='Повторно использованные20val, не независимый итоговый тест. Измеряется эта собственная масочная ветвь. Её исход не подтверждает и не опровергает точную LTR, SGAR-MOT, Б1–Б3 или вклад ядра FlyVis.',
        sha256={name:sha(ROOT/name) for name in sorted(set(sources))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();data=json.loads(before)
    assert not any(c['id']==IDENT for c in data['experiments'])
    data['experiments'].append(dict(id=IDENT,title='Восстановление потерянного объекта — собственная масочная ветвь SAM2',area='flyvis',state='ready',started='2026-10-27',
        question='Даст ли причинный масочный перенос пользу HOTA без изменения исходного трекера?',
        hypothesis='Перенос маски и отказ при отсутствии объекта могут восстановить часть пропусков с меньшим вредом от ложных продолжений; знак HOTA неизвестен.',
        expected=p['action'],summary=p['adaptation'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],files=sources[:2],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие,контроль,отличия,ценаикритериидоизмерения')]))
    assert path.read_bytes()==before;write(path,data)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлен полный пилот собственной масочной ветви',
        '*Постановка.* '+p['action']+'\n\n*Отличия и возражение.* '+p['adaptation']+'\n\n'
        '*Условия.* Те же20val и детекции, без MASA, базовый XYWH повторяется точно. '
        'Первый полный видеопроход определяет продолжение по измеренной цене; '+p['cost_gate']+' '
        'Контроли простого продолжения и выходной BUSCA уже сохранены. Ничего не подбирается по результату. '
        '*Принятие.* '+p['acceptance'][-1]+'. '+p['limitation']+' '
        'Источник:runs/b123_20261027/sam2_output_pilot/protocol.json. HOTA этой ветви ещё нет.')
    print('Prepared full SAM2 adaptation pilot; validate before publication')
def run():
    import torch,numpy as np,cv2
    torch.set_num_threads(2);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    from tracking.sam2_mask_recovery import load_predictor,MaskOutputRecovery
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from ultralytics.trackers.basetrack import BaseTrack
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');start=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    def check():
        if time.time()>deadline-5 or (RUN/'stop.request').exists():raise TimeoutError('Original budget/author deadline')
    def status(stage,**kw):
        check();write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    per={a:{} for a in p['arms']};exact={};timing={};counters={}
    for index,video in enumerate(p['videos']):
        began=time.monotonic();status('Загрузка фиксированной SAM2',video=video,index=index,total_videos=len(p['videos']))
        BaseTrack.reset_id();model=load_predictor('cuda');branch=MaskOutputRecovery(model,p['frames'][video])
        tracker=XYWHByteTrack(SimpleNamespace(**p['configs']['xywh']))
        dets=load_detection_cache(ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/detections/raw_video.npz')
        align=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json')['matches']
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
        anchors={int(row['video_frame_index']):i for i,row in enumerate(align)};assert len(anchors)==len(truth)
        cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/val/{video}.mov'));assert cap.isOpened()
        raw=[];rows={a:[] for a in p['arms']};added=0
        for frame in range(p['frames'][video]):
            if frame%15==0:status('Неизменный XYWH и причинный масочный перенос',video=video,done=frame,total=p['frames'][video],initialized=branch.initialized,propagated=branch.propagated,added=added)
            ok,image=cap.read();assert ok,(video,frame)
            xyxy,conf,classes=detections_for_frame(dets,frame)
            output=tracker.update(_DetSlice(xyxy,conf,classes));base=np.asarray(output,dtype=np.float32).reshape(-1,8)
            if len(base):raw.append(np.column_stack([np.full(len(base),frame),base[:,:7]]).astype(np.float32))
            ids={int(row[4]) for row in base}
            lost=[track for track in tracker.lost_stracks if track.track_id not in ids and track.is_activated and 1<=tracker.frame_id-track.end_frame<=p['max_lost_age']]
            extra=branch.step(frame,image,base,lost)
            if frame in anchors:
                added+=len(extra);extended=np.concatenate([base,extra]) if len(extra) else base.copy()
                for arm,out in [('baseline',base),('sam2_mask',extended)]:rows[arm].append(scalabel_frame(video,truth[anchors[frame]],out))
        cap.release();actual=np.concatenate(raw) if raw else np.zeros((0,8),np.float32)
        with np.load(OLD/'xywh/raw'/(video+'.npz')) as z:assert np.array_equal(actual,z['outputs']),('Baseline mismatch',video)
        exact[video]=True
        for arm in p['arms']:
            assert len(rows[arm])==len(truth);write(RUN/arm/(video+'.json'),rows[arm]);status('Оценка полного видео',video=video,arm=arm)
            per[arm][video]=evaluate_hota(RUN/arm,[video],split='val');write(RUN/arm/'per_video.json',per[arm])
        control=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json')
        assert per['baseline'][video]==control['per_video']['baseline'][video]
        counters[video]=dict(initialized=branch.initialized,propagated=branch.propagated,rejected=branch.rejected,
            all_image_requests_causal=all(a<=b for b,a in branch.frames.requests),added_annotated_boxes=added)
        timing[video]=time.monotonic()-began
        write(RUN/'completed_videos.json',dict(videos=list(exact),exact=exact,counters=counters,timing=timing))
        del branch,model;torch.cuda.empty_cache()
        if index==0:
            price=2*timing[video]*19+120;available=deadline-time.time()
            write(RUN/'price.json',dict(first_video_seconds=timing[video],conservative_remaining_seconds=price,available_seconds=available,proceed=price<available))
            assert price<available,'Measured remaining cost exceeds original budget'
    pooled={}
    for arm in p['arms']:
        status('Общая оценка20видео',arm=arm);pooled[arm]=evaluate_hota(RUN/arm,p['videos'],split='val')
    controls=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json');busca=read(ROOT/'runs/b123_20261027/busca_output_pilot/report.json')
    assert pooled['baseline']==controls['pooled']['baseline']
    contrast={name:paired({v:m['HOTA'] for v,m in per['sam2_mask'].items()},
        {v:m['HOTA'] for v,m in other.items()}) for name,other in [('vs_baseline',per['baseline']),('vs_simple_continuation',controls['per_video']['emit_lost_kalman']),('vs_busca_output',busca['per_video']['busca_output'])]}
    write(RUN/'report.json',dict(complete=True,videos=p['videos'],pooled=pooled,per_video=per,paired=contrast,
        baseline_arrays_exact=exact,counters=counters,timing=timing,seconds=time.monotonic()-start,
        hota_measured=True,ltr_reproduced=False,sgar_reproduced=False,flyvis_core_used=False,tracker_state_changed=False,
        adaptation=p['adaptation'],limitations=p['limitation']))
    write(RUN/'status.json',dict(state='completed',stage='Полный итог масочной адаптации сохранён',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
