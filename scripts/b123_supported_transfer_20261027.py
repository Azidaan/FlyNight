"""Переносимость фиксированного правила на другой части прежнего локального val."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import hashlib,json,sys,time,contextlib,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'runs/b123_20261027';RUN=BASE/'supported_transfer';IDENT='detection_support_transfer_v1';MSK=timezone(timedelta(hours=3))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def prepare():
    assert not RUN.exists();online=read(BASE/'online_supported_verification/report.json');assert online['complete']
    dev=read(BASE/'detection_supported_recovery/protocol.json');inventory=read(BASE/'unseen_rule_video_inventory.json');assert inventory['remaining_count']==0
    names=sorted([p.stem for p in (ROOT/'data/bdd100k/videos/val').glob('*.mov') if p.stem not in dev['videos']],key=lambda v:hashlib.sha256(('det-support-transfer-20261027:'+v).encode()).hexdigest())[:20]
    assert len(names)==20 and not set(names)&set(dev['videos'])
    weights='weights/baseline_seed42/best.pt';meta=read(ROOT/'runs/z13_val20_matched_20261024/cache'/dev['videos'][0]/'detections/raw_video.json')['identity']
    assert sha(ROOT/weights)==meta['weights_sha256']
    source=['scripts/b123_supported_transfer_20261027.py','tracking/detection_supported_kalman.py','tracking/recovery_detection_support.py',
        'tracking/xywh_bytetrack.py','tracking/offline_association.py','scripts/run_bdd_video_rate_experiment.py','tracking/video_alignment.py','evaluation/hota_eval.py','scripts/z13_continue.py',weights,
        'runs/b123_20261027/online_supported_verification/report.json','runs/b123_20261027/detection_supported_recovery/report.json','runs/b123_20261027/unseen_rule_video_inventory.json']
    images={}
    for v in names:
        paths=sorted((ROOT/f'data/bdd100k/images/track/val/{v}').glob('*.jpg'));assert paths
        images[v]=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in paths]
        source += [f'data/bdd100k/videos/val/{v}.mov',f'data/bdd100k/labels/box_track_20/val/{v}.json']
    p=dict(id=IDENT,created=datetime.now(MSK).isoformat(),budget_seconds=4800,author_stop='2026-10-27T07:20:00+03:00',device='cuda',videos=names,development_videos=dev['videos'],images=images,
        selection='First20 of remaining180 localval by SHA256(det-support-transfer-20261027:videoID), excluding20rule-development videos; no label/image content/outcome selection.',
        weights=weights,imgsz=meta['imgsz'],conf=meta['conf'],iou=meta['iou'],config=read(BASE/'recovery_motion_control/protocol.json')['configs']['xywh'],search_radius=read(ROOT/'runs/z13_val20_matched_20261024/cache'/dev['videos'][0]/'alignment.json')['search_radius'],
        action='Тот же фиксированный Калман+поддержка текущим обнаружением, те же conf>0.1/IoU0.5/возраст1..30,без подбора. '
            'Текущий YOLO26 baseline считает обнаружения на каждом исходном кадре; оба выхода получают ровно один кэш. '
            'Исходные назначения/состояние XYWH не меняются; выключенный выход проверяется на точное совпадение с исходным массивом каждый кадр.',
        cost_gate='Первое полное видео, включая выравнивание/YOLO/оба выхода/TrackEval:2*стоимость*19+120 должны помещаться в остаток4800с идо07:20. Иначе сохранить первый результат и остановить без продления.',
        acceptance=['Все20непересекающихся с разработкой видео, одинаковые детекции/оценка, точный выключенный контроль',
            'HOTA/AssA/IDF1/FP/FN/IDSw и парный95%интервал; для пользы нижняя граница supported−baseline>0',
            'Не менять веса,пороги,правило или выборку после результатов; полнота и независимая сверка сохранённых выходов'],
        limitation='Отдельная часть прежнего val,а не никогда не использованный итоговый тест: у всех200локальныхвидео есть следы прежнихработ. '
            'Выбор не пересекается с20разработки нового правила,но не исключает общего влияния истории проекта/выбора детектора. '
            'Нельзя называть доказательством на внешнем датасете,вкладом FlyVis,маски или точнойLTR. Это полезность фиксированного параллельного модуля всего проекта. '
            'Новый набор не загружается; используются существующие локальные изображения/видео/разметка. Разметка объектов только у оценщика; выравнивание использует изображения для времён запроса.',
        sha256={name:sha(ROOT/name) for name in sorted(set(source))})
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Поддержка текущим обнаружением — перенос на другую часть val',area='flyvis',state='ready',started='2026-10-27',
        question='Сохранится ли польза фиксированного правила на других20видео прежнегоval?',hypothesis='Поддержка текущим обнаружением может перенестись за пределы20разработки;знак неизвестен.',
        expected=p['action'],summary=p['selection'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],files=source[:3],
        evidence=[dict(kind='file',ref=RUN.relative_to(ROOT).as_posix()+'/protocol.json',claim='Неизменные правила, входы, не пересекающаяся выборка и критерии до нового измерения')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлена проверка переносимости фиксированного правила','*Постановка.* '+p['action']+'\n\n*Выборка.* '+p['selection']+'\n\n*Ограничение независимости.* '+p['limitation']+'\n\n*Бюджет.* '+p['cost_gate']+' НовыйGPUтолько после проверенного конца предыдущих деревьев;одинGPU,срок автора прежний. Источник:runs/b123_20261027/supported_transfer/protocol.json. HOTAэтойчастиvalещёнет.')
    print('Prepared fixed-rule transfer on disjoint20 of prior localval, not a new external test')
def run():
    import numpy as np,torch
    torch.set_num_threads(2);torch.manual_seed(7)
    from scripts.run_bdd_video_rate_experiment import build_alignment,build_detection_cache,load_detection_cache,detections_for_frame,scalabel_frame,CLASS_NAMES
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from tracking.detection_supported_kalman import DetectionSupportedKalmanOutput
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');started=time.monotonic();deadline=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    def status(stage,**kw):
        if time.time()>deadline-3 or (RUN/'stop.request').exists():raise TimeoutError('Original stage budget/author stop')
        write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(MSK).isoformat(),**kw))
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    per={arm:{} for arm in ['baseline','supported']};timing={};proof={}
    for index,v in enumerate(p['videos']):
        began=time.monotonic();folder=RUN/'cache'/v;folder.mkdir(parents=True,exist_ok=True)
        status('Временное выравнивание существующих изображений',video=v,index=index,total=20)
        images=[]
        for item in p['images'][v]:
            path=ROOT/item['path'];assert sha(path)==item['sha256'];images.append(path)
        matches,video_meta=build_alignment(ROOT/f'data/bdd100k/videos/val/{v}.mov',images,folder/'alignment.json',p['search_radius'])
        status('YOLO26 на каждом исходном кадре',video=v,index=index,total=20)
        class Progress:
            def write(self,s):
                sys.__stdout__.write(s)
                match=re.search(r'\[YOLO\] (\d+) кадров',s)
                if match:status('YOLO26 на каждом исходном кадре',video=v,index=index,done=int(match.group(1)),total=video_meta['decoded_frame_count'])
            def flush(self):sys.__stdout__.flush()
        with contextlib.redirect_stdout(Progress()):meta=build_detection_cache(ROOT/f'data/bdd100k/videos/val/{v}.mov',ROOT/p['weights'],folder/'raw_video.npz',imgsz=p['imgsz'],conf=p['conf'],iou=p['iou'],device='0',force=False)
        dets=load_detection_cache(folder/'raw_video.npz');assert int(dets['frame_count'])==int(video_meta['decoded_frame_count'])
        tracker=XYWHByteTrack(SimpleNamespace(**p['config']));tracker.reset_id();off=DetectionSupportedKalmanOutput(CLASS_NAMES);on=DetectionSupportedKalmanOutput(CLASS_NAMES,enabled=True,low_confidence=p['config']['track_low_thresh'])
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');assert len(truth)==len(matches);anchors={int(x['video_frame_index']):i for i,x in enumerate(matches)}
        rows={arm:[] for arm in per};raw=[];augmented=[]
        for frame in range(int(dets['frame_count'])):
            if frame%100==0:status('Неизменный XYWH и правило с текущими входами',video=v,index=index,done=frame,total=int(dets['frame_count']))
            xyxy,conf,cls=detections_for_frame(dets,frame);base=np.asarray(tracker.update(_DetSlice(xyxy,conf,cls)),dtype=np.float32).reshape(-1,8)
            assert np.array_equal(off.output(tracker,base,xyxy,conf,cls),base)
            if len(base):raw.append(np.column_stack([np.full(len(base),frame),base[:,:7]]).astype(np.float32))
            if frame not in anchors:continue
            extended=on.output(tracker,base,xyxy,conf,cls);assert np.array_equal(extended[:len(base)],base)
            for arm,output in [('baseline',base),('supported',extended)]:rows[arm].append(scalabel_frame(v,truth[anchors[frame]],output))
            if len(extended):augmented.append(np.column_stack([np.full(len(extended),frame),extended]).astype(np.float32))
        np.savez_compressed(folder/'tracker_arrays.npz',baseline=np.concatenate(raw) if raw else np.zeros((0,8),np.float32),supported_queries=np.concatenate(augmented) if augmented else np.zeros((0,9),np.float32))
        for arm in per:
            assert len(rows[arm])==len(matches);write(RUN/arm/(v+'.json'),rows[arm]);status('Оценка видео',video=v,arm=arm);per[arm][v]=evaluate_hota(RUN/arm,[v],split='val');write(RUN/arm/'per_video.json',per[arm])
        timing[v]=time.monotonic()-began;proof[v]=dict(disabled_exact_every_frame=True,one_shared_detection_cache=True,detector_cache_sha256=sha(folder/'raw_video.npz'),tracker_arrays_sha256=sha(folder/'tracker_arrays.npz'))
        write(RUN/'completed_videos.json',dict(videos=list(timing),timing=timing,proof=proof))
        if index==0:
            cost=2*timing[v]*19+120;available=deadline-time.time();write(RUN/'price.json',dict(first_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available,'Measured remaining cost exceeds original budget'
    pooled={}
    for arm in per:status('Общая оценка другой частиval',arm=arm);pooled[arm]=evaluate_hota(RUN/arm,p['videos'],split='val')
    contrast=paired({v:x['HOTA'] for v,x in per['supported'].items()},{v:x['HOTA'] for v,x in per['baseline'].items()})
    write(RUN/'report.json',dict(complete=True,videos=p['videos'],development_videos=p['development_videos'],pooled=pooled,per_video=per,paired=contrast,proof=proof,timing=timing,seconds=time.monotonic()-started,tracker_state_changed=False,flyvis_core_used=False,sam2_used=False,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',stage='Переносимость на отдельной частиval измерена',updated=datetime.now(MSK).isoformat()))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(MSK).isoformat()));raise
