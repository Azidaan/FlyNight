"""Проверить исполняемый путь: нулевой контроль и точный повтор полезного выхода."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import json,hashlib,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'online_supported_verification';IDENT='online_detection_support_verification_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists();parent=read(BASE/'detection_supported_recovery/protocol.json');motion=read(BASE/'recovery_motion_control/protocol.json')
    assert read(BASE/'detection_supported_recovery/independent_audit.json')['passed']
    names=['scripts/b123_online_supported_verification_20261027.py','tracking/detection_supported_kalman.py','tracking/recovery_detection_support.py','tracking/xywh_bytetrack.py',
        'tracking/offline_association.py','scripts/run_bdd_video_rate_experiment.py','runs/b123_20261027/detection_supported_recovery/report.json']
    for v in parent['videos']:
        names += [f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz',f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json',
            f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.npz',f'runs/b123_20261027/detection_supported_recovery/baseline/{v}.json',f'runs/b123_20261027/detection_supported_recovery/kalman_support/{v}.json']
    p=dict(id=IDENT,budget_seconds=900,author_stop='2026-10-27T07:20:00+03:00',device='cpu',videos=parent['videos'],frames=motion['frames'],config=motion['configs']['xywh'],
        action='Исполняемый необучаемый выходной модуль после неизменного XYWH; выключен по умолчанию. Включённый вариант получает только текущее lost-состояние/реальные детекции, без сохранённых будущих кандидатов или разметки.',
        acceptance=['Выключенный вариант точно сохраняет исходный XYWH на каждом кадре всех20видео',
            'Включённый вариант даёт точный JSON сохранённого kalman_support на всех фиксированных запросах',
            'Нет повторного TrackEval или нового научного вывода: HOTA относится к прежнему источнику, измеряется перенос исполнения'],
        limitation='Проверка интеграции на тех же20,не независимая научная проверка. Основной стенд не меняется. Точность подтверждается на фиксированных запросах; невидимый для оценки30Гцвыход качества отдельно не измеряет.',
        sha256={name:sha(ROOT/name) for name in sorted(set(names))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Поддержка текущим обнаружением — проверка исполняемого пути',area='flyvis',state='ready',started='2026-10-27',
        question='Воспроизводит ли модуль с текущими входами измеренный сохранённый выход без изменения базы?',hypothesis='Новый способ исполнения должен точно повторить фиксированный результат; новая польза не проверяется.',
        expected=p['action'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],files=names[:3],
        evidence=[dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/protocol.json',claim='Критерии переноса исполнения до проверки')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлена проверка исполняемого пути поддержки детектором','*Постановка.* '+p['action']+' '+p['limitation']+' CPU900с,после первого полного видео2*t*19+30должно помещаться в остаток,иначе остановка без продления. Три критерия точного совпадения вruns/b123_20261027/online_supported_verification/protocol.json. ПовторныйTrackEval не запускается,baselineпоумолчаниюнеизменный.')
    print('Prepared online execution verification')
def run():
    import numpy as np
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from tracking.detection_supported_kalman import DetectionSupportedKalmanOutput
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame,CLASS_NAMES
    p=read(RUN/'protocol.json');began=time.monotonic();deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    exact={};timing={}
    def status(**kw):
        if time.time()>deadline-3 or (RUN/'stop.request').exists():raise TimeoutError('Original stage budget/author stop')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw))
    for i,v in enumerate(p['videos']):
        start=time.monotonic();tracker=XYWHByteTrack(SimpleNamespace(**p['config']));tracker.reset_id()
        enabled=DetectionSupportedKalmanOutput(CLASS_NAMES,enabled=True,low_confidence=p['config']['track_low_thresh']);disabled=DetectionSupportedKalmanOutput(CLASS_NAMES)
        dets=load_detection_cache(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz');matches=read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')['matches']
        anchors={int(x['video_frame_index']):j for j,x in enumerate(matches)};base=read(BASE/'detection_supported_recovery/baseline'/(v+'.json'));target=read(BASE/'detection_supported_recovery/kalman_support'/(v+'.json'))
        raw=[];output=[]
        for frame in range(p['frames'][v]):
            if frame%100==0:status(video=v,index=i,done=frame,total=p['frames'][v])
            xyxy,conf,cls=detections_for_frame(dets,frame);current=np.asarray(tracker.update(_DetSlice(xyxy,conf,cls)),dtype=np.float32).reshape(-1,8)
            assert np.array_equal(disabled.output(tracker,current,xyxy,conf,cls),current)
            if len(current):raw.append(np.column_stack([np.full(len(current),frame),current[:,:7]]).astype(np.float32))
            if frame not in anchors:continue
            j=anchors[frame];out=enabled.output(tracker,current,xyxy,conf,cls)
            assert scalabel_frame(v,base[j],current)==base[j]
            rendered=scalabel_frame(v,base[j],out);assert rendered==target[j],(v,frame)
            output.append(rendered)
        actual=np.concatenate(raw) if raw else np.zeros((0,8),np.float32)
        with np.load(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.npz') as z:assert np.array_equal(actual,z['outputs'])
        write(RUN/'enabled'/(v+'.json'),output);exact[v]=True;timing[v]=time.monotonic()-start
        write(RUN/'completed_videos.json',dict(exact=exact,timing=timing))
        if i==0:
            cost=2*timing[v]*19+30;remaining=deadline-time.time();write(RUN/'price.json',dict(first_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=remaining,proceed=cost<remaining));assert cost<remaining
    write(RUN/'report.json',dict(complete=True,all20_baseline_arrays_exact=exact,all20_enabled_json_exact=exact,disabled_is_exact=True,enabled_by_default=False,
        model_inference_used=False,hota_remeasured=False,prior_hota_source='runs/b123_20261027/detection_supported_recovery/report.json',prior_hota_source_sha256=sha(BASE/'detection_supported_recovery/report.json'),seconds=time.monotonic()-began,timing=timing,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat(),stage='Точное совпадение исполняемого пути сохранено'))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',updated=datetime.now(timezone.utc).isoformat(),stage=repr(e)));raise
