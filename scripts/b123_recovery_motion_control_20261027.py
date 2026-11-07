"""Общий контроль восстановления: только выдача текущих прогнозов потерянных треков."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/recovery_motion_control'
OLD=ROOT/'runs/z13_val20_matched_20261024/raw'
MSK=timezone(timedelta(hours=3))
IDENT='recovery_motion_output_control_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.next')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(path)


def prepare():
    assert not RUN.exists()
    old=read(OLD/'protocol.json');auth=read(ROOT/'runs/b123_20261027/recovery_followup_authorization.json')
    videos=old['videos'];assert len(videos)==20
    names=['scripts/b123_recovery_motion_control_20261027.py','tracking/xywh_bytetrack.py',
           'tracking/offline_association.py','scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py',
           'scripts/z13_continue.py','runs/z13_val20_matched_20261024/raw/protocol.json',
           'runs/z13_val20_matched_20261024/raw/report.json']
    for v in videos:
        names += [f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz',
                  f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json',
                  f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.npz',
                  f'data/bdd100k/labels/box_track_20/val/{v}.json']
    import inspect
    from ultralytics.trackers.byte_tracker import BYTETracker
    names.append(Path(inspect.getfile(BYTETracker)).relative_to(ROOT).as_posix())
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),timezone='Europe/Moscow',device='cpu',
        author_stop=auth['save_at'],budget_seconds=900,videos=videos,frames=old['frames'],configs=old['configs'],
        arms=['baseline','emit_lost_kalman'],maximum_lost_age_original_frames=30,
        action='После неизменного XYWHByteTrack дополнить только выдачу рамками его lost_stracks, возраст1..30 исходных кадров. Состояние, ассоциацию и память исходного трекера не менять. Все рамки — текущий прогноз из прошлого, без разметки/будущего/новых детекций. Это простой контроль выдачи, не реализация BUSCA/LTR и не полный контроль их влияния на память.',
        baseline='Каждая исходная рамка, идентификатор, уверенность, класс и кадр должны точно совпасть с сохранённым raw XYWH всех20видео. Тот же TrackEval и разметка; никакой MASA в обоих вариантах. Геометрический XYWH не использует изображения, поэтому повторно видео не декодируется.',
        cost_gate='После первого полного видео с обеими оценками:2*walltime*19+60 < remaining stage budget and author time. Иначе остановить и сохранить технический остаток;900с не продлевать.',
        acceptance=['Точное воспроизведение всех20исходных массивов и полнота размеченных кадров',
                    'Все20видео/оба варианта/одинаковая HOTA и парный95%интервал по видео; показать AssA/IDF1/FP/FN/IDSw',
                    'Для заявления пользы именно контроля нижняя граница разности HOTA >0; иначе сохранить отрицательный/неопределённый результат. Это не критерий закрытия BUSCA/LTR.'],
        limitations='Повторно использованные20val, без настройки по ним, не независимый итоговый тест. Нет learned refusal, внешности, сегментации или изменения памяти; вред простого контроля не опровергает BUSCA/LTR. Разметка читается только для оформления оцениваемых кадров и оценщика. Не общий потолок HOTA.',
        sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';raw=path.read_bytes();cat=json.loads(raw)
    assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Восстановление потерянного объекта — контроль выдачи калмановского прогноза',area='flyvis',state='ready',started='2026-10-27',
        question='Каковы польза и вред простого продолжения потерянного объекта до подключения BUSCA или масок?',
        hypothesis='Выдача прогнозов способна уменьшить пропуски и увеличить ложные рамки; знак HOTA неизвестен.',
        expected=p['action'],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],summary='Общий ограниченный контроль для порученного восстановления;условия до измерения.',
        obtained='',interpretation='',decision='',cannot_claim=p['limitations'],metrics=[],commits=[],files=[names[0]],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие,точный контроль,стоимость и критерии до измерения')]))
    assert path.read_bytes()==raw;write(path,cat)
    write(ROOT/'runs/b123_20261027/ready_jobs/60_recovery_motion_control.json',dict(title='Контроль восстановления потерянных объектов без обучаемой проверки',
        folder=RUN.relative_to(ROOT).as_posix(),script=names[0],script_sha256=sha(ROOT/names[0]),protocol_sha256=sha(RUN/'protocol.json')))
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT=IDENT
    J.journal('поставлен общий контроль восстановления перед BUSCA/LTR',
        '*Постановка.* '+p['action']+'\n\n*Условия и критерии до прогона.* '+p['baseline']+' '+p['cost_gate']+' '+p['acceptance'][-1]+'. '
        '20фиксированныхval,900сCPU после текущей очереди. Источник protocol.json в `runs/b123_20261027/recovery_motion_control/`. '
        'Никакого подбора порога по этим20,выдача≤30кадров берётся из уже действующего буфера. Это контроль,не BUSCA/LTR и не верхний потолок. Результатов ещё нет.')
    print('Registered '+IDENT)


def run():
    import numpy as np
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from ultralytics.trackers.basetrack import BaseTrack
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');start=time.monotonic()
    deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    def check():
        if (RUN/'stop.request').exists() or time.time()>deadline-3:raise TimeoutError('Own budget/author deadline')
    def status(stage,**kw):
        check();write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    per={a:{} for a in p['arms']};timing={};exact={};added={}
    for index,v in enumerate(p['videos']):
        check();began=time.monotonic();BaseTrack.reset_id()
        tracker=XYWHByteTrack(SimpleNamespace(**p['configs']['xywh']))
        dets=load_detection_cache(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz')
        alignment=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json')
        anchors={int(row['video_frame_index']):i for i,row in enumerate(alignment['matches'])};assert len(anchors)==len(truth)
        rows={a:[] for a in p['arms']};raw=[];count=0
        for f in range(p['frames'][v]):
            if f%50==0:status('Неизменный трекер и дополнительная выдача',video=v,done=f,total=p['frames'][v])
            xyxy,conf,classes=detections_for_frame(dets,f)
            output=tracker.update(_DetSlice(xyxy,conf,classes))
            if len(output):raw.append(np.column_stack([np.full(len(output),f),np.asarray(output)[:,:7]]).astype(np.float32))
            if f not in anchors:continue
            base=np.asarray(output,dtype=np.float32).reshape((-1,8))
            ids={int(r[4]) for r in base};extra=[]
            for track in tracker.lost_stracks:
                age=tracker.frame_id-track.end_frame
                if 1<=age<=p['maximum_lost_age_original_frames'] and track.track_id not in ids and track.is_activated:
                    box=np.asarray(track.xyxy,dtype=np.float32)
                    if np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]:
                        extra.append([*box,track.track_id,track.score,track.cls,-1])
            extended=np.concatenate((base,np.asarray(extra,dtype=np.float32).reshape((-1,8)))) if extra else base.copy()
            count+=len(extra)
            for arm,out in (('baseline',base),('emit_lost_kalman',extended)):
                rows[arm].append(scalabel_frame(v,truth[anchors[f]],out))
        actual=np.concatenate(raw) if raw else np.zeros((0,8),np.float32)
        with np.load(OLD/'xywh/raw'/(v+'.npz')) as z:
            assert int(z['frame_count'])==p['frames'][v] and np.array_equal(actual,z['outputs']),('Baseline mismatch',v)
        exact[v]=True;added[v]=count
        for arm in p['arms']:
            assert len(rows[arm])==len(truth)
            write(RUN/arm/(v+'.json'),rows[arm])
            status('Официальная оценка видео',video=v,arm=arm)
            per[arm][v]=evaluate_hota(RUN/arm,[v],split='val')
            write(RUN/arm/'per_video.json',per[arm])
        timing[v]=time.monotonic()-began
        if index==0:
            estimate=2*timing[v]*19+60;remaining=deadline-time.time()
            price=dict(first_video_both_arms_seconds=timing[v],conservative_remaining_seconds=estimate,available_seconds=remaining,proceed=estimate<remaining)
            write(RUN/'price.json',price);assert price['proceed'],'Remaining cost exceeds original budget'
    pooled={}
    for arm in p['arms']:
        status('Общая оценка всех20видео',arm=arm)
        pooled[arm]=evaluate_hota(RUN/arm,p['videos'],split='val')
    original=read(OLD/'report.json')
    assert pooled['baseline']['HOTA']==original['pooled']['xywh']['HOTA']
    contrast=paired({v:m['HOTA'] for v,m in per['emit_lost_kalman'].items()},
                    {v:m['HOTA'] for v,m in per['baseline'].items()})
    report=dict(complete=True,videos=p['videos'],pooled=pooled,per_video=per,paired=contrast,
                baseline_arrays_exact=exact,added_annotated_boxes=added,seconds=time.monotonic()-start,
                tracker_state_changed=False,busca_executed=False,ltr_executed=False,limitations=p['limitations'])
    write(RUN/'report.json',report)
    write(RUN/'status.json',dict(state='completed',stage='Контроль выдачи готов;это не BUSCA/LTR',updated=datetime.now(MSK).isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:
            write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
