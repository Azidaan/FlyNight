"""Fixed additional20val input preparation; no training, tuning or held-out75."""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_val20_matched_20261024';STAGE=RUN/'inputs';IDENT='z13_val20_inputs_v1';T0=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as J
    before=J.IDENT
    try:J.IDENT=IDENT;J.journal(title,text)
    finally:J.IDENT=before


def check():
    p=read(STAGE/'protocol.json')
    if (STAGE/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Input preparation budget/deadline')


def status(stage,**kw):
    check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),jobs_done=len(list((STAGE/'jobs').glob('*.json'))),jobs_total=21,**kw))


def prepare():
    import hashlib,cv2
    assert not RUN.exists();auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json');assert now()<datetime.fromisoformat(auth['save_at'])
    old=read(ROOT/'runs/z13_matched_trackers_20261024/protocol.json');excluded=set(old['videos'])
    clips=list((ROOT/'data/bdd100k/videos/val').glob('*.mov'));ordered=sorted((p.stem for p in clips if p.stem not in excluded),key=lambda v:hashlib.sha256(('z13-val20-inputs-v1:42:'+v).encode()).hexdigest());videos=ordered[:20];assert len(videos)==20
    train=ROOT/'training/configs/generated/full_step5_train.txt';training=train.read_text(encoding='utf-8');assert all(v not in training for v in videos)
    files=['scripts/z13_val20_inputs_20261024.py','scripts/run_bdd_video_rate_experiment.py','tracking/video_alignment.py','tracking/gpu_lock.py','weights/baseline_seed42/best.pt',train.relative_to(ROOT).as_posix(),'runs/z13_matched_trackers_20261024/report.json']
    probe=old['videos'][0];files+=['runs/video_rate/'+probe+'/detections/raw_video.npz','runs/video_rate/'+probe+'/detections/raw_video.json'];frames={};imagecounts={}
    for video in [probe,*videos]:
        path=ROOT/f'data/bdd100k/videos/val/{video}.mov';label=ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json';images=sorted((ROOT/f'data/bdd100k/images/track/val/{video}').glob('*.jpg'));assert path.is_file() and label.is_file() and images
        cap=cv2.VideoCapture(str(path));assert cap.isOpened();frames[video]=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();imagecounts[video]=len(images)
        files.extend(p.relative_to(ROOT).as_posix() for p in [path,label,*images])
    p=dict(id=IDENT,created=now().isoformat(),device='cuda',budget_seconds=3600,save_at=auth['save_at'],hard_stop=auth['hard_stop'],stop_request_path=(STAGE/'stop.request').relative_to(ROOT).as_posix(),videos=videos,probe_video=probe,frames=frames,imagecounts=imagecounts,configs=old['configs'],weights='weights/baseline_seed42/best.pt',detector=dict(imgsz=960,conf=.01,iou=.7,device='0'),search_radius=24,
      selection='Exactly20 localval videos excluding previous7, sorted SHA256(z13-val20-inputs-v1:42:+videoid), before any new detections/outputs/metrics. No video selected by tracker performance, lighting or failure. Fixed sample is not enlarged until significance; no final75 opened.',
      purpose='Prepare fixed original-frame detections/alignment for a broader matched benchmark requested by author. This does not improve the detector and is not promotion of failed FlyVis readouts. Core learning absent. Next registered CPU baseline4 trackers uses exactly these inputs; subsequent FlyVis inference needs a separate fixed protocol and measured cost.',
      execution='Separate interpreter per video, one GPU mutex, bounded3600seconds. First replay previous7val probe detection arrays xyxy/conf/cls/frame_idx/frame_count exactly; timing arrays excluded. Then fixed first new video alignment/detections, conservative2x full-video wall*19remaining+120s must fit remaining original stage budget. Completed inputs retained. No HOTA from preparation.',
      limitations='All200val videos were used in older low-frame-rate tracker research; detector val monitored during weight selection. New20 are additional original-frame caches, not untouched independent confirmation or final test. No outcome-dependent exclusion; ambiguous alignment stops preparation, no replacement video.',sha256={n:sha(ROOT/n) for n in sorted(set(files))})
    write(STAGE/'protocol.json',p);write(RUN/'selection.json',dict(videos=videos,excluded_previous7=sorted(excluded),selection=p['selection'],limitations=p['limitations']))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Подготовка20дополнительныхval для общего сравнения исходных кадров',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['benchmark','infrastructure'],question='Можно ли подготовить фиксированные20дополнительныхval в тех же условиях YOLO26s960/conf0,01,без нового обучения и выбора по исходу?',hypothesis='Проверяемый входной этап для более широкой сравнительной оценки;преимуществоFlyVisнепредполагается.',expected=p['purpose']+' '+p['execution'],acceptance=[dict(criterion='Прежний детектор точно воспроизведён на контрольном полном видео;веса/код неизменны.',met=None),dict(criterion='Все20роликов и кадры сопоставлены по содержимому,числа детекционных/декодированных/размеченных кадров согласованы,хеши сохранены.',met=None),dict(criterion='Проверенный консервативный бюджет и полный конец своего дерева;никаких новых научных исходов/HOTAпо подготовке.',met=None)],summary='Фиксированный список и условия до новых детекций;не независимый финальныйтест.',obtained='',interpretation='',decision='',cannot_claim=p['limitations']+' НетHOTAиулучшениядетектора/мухи.',metrics=[],commits=[],files=files[:4],evidence=[dict(kind='file',ref=(STAGE/'protocol.json').relative_to(ROOT).as_posix(),claim='Список,условия и хеши до расчёта')]))
    write(cp,cat);journal('зафиксированы20дополнительныхval для одинаковых условий','*Постановка.* '+p['selection']+'\n\n'+p['purpose']+'\n\nСначала точное воспроизведение прежнихдетекций на одном полномконтрольномролике,затемпервый новыйролик и проверка2xценыоставшихся19+120с. Отдельныйбюджет3600с,одинGPU,исходные скрипты/веса неизменны. Это новая подготовительнаястадия сдругимфиксированнымнабором,непродление остановленногоstock10800с. '+p['limitations']+'\n\nИсточник: runs/z13_val20_matched_20261024/inputs/protocol.json;подготовка неоцениваетHOTAи не меняеттрактFlyVis.')


def job(video,probe=False):
    import cv2,numpy as np,torch
    from scripts import run_bdd_video_rate_experiment as V
    from tracking.gpu_lock import GpuLock
    p=read(STAGE/'protocol.json');assert video in p['videos']+[p['probe_video']];torch.set_num_threads(2);cv2.setNumThreads(2)
    folder=RUN/('probe' if probe else 'cache')/video
    def progress_print(*values,**kwargs):
        print(*values,**kwargs);m=re.match(r'\[YOLO\] (\d+) кадров',' '.join(map(str,values)))
        if m:status('Неизменные детекции каждого исходного кадра',variant=video,done=int(m.group(1)),total=p['frames'][video])
    V.print=progress_print
    for n,h in p['sha256'].items():
        if video in n or n.endswith(('.py','.pt')):assert sha(ROOT/n)==h,n
    status('Контроль воспроизведения' if probe else 'Сопоставление кадров по содержимому',variant=video)
    if not probe:
        images=sorted((ROOT/f'data/bdd100k/images/track/val/{video}').glob('*.jpg'));matches,meta=V.build_alignment(ROOT/f'data/bdd100k/videos/val/{video}.mov',images,folder/'alignment.json',p['search_radius'])
        assert len(matches)==len(read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json'))==p['imagecounts'][video]
        assert len({m['video_frame_index'] for m in matches})==len(matches) and int(meta['decoded_frame_count'])==p['frames'][video]
    with GpuLock('flyvis_fovea_gpu'):result=V.build_detection_cache(ROOT/f'data/bdd100k/videos/val/{video}.mov',ROOT/p['weights'],folder/'detections/raw_video.npz',**p['detector'],force=False)
    assert result['frame_count']==p['frames'][video]
    with np.load(folder/'detections/raw_video.npz') as z:
        assert np.isfinite(z['xyxy']).all() and np.isfinite(z['conf']).all()
        if probe:
            with np.load(ROOT/f'runs/video_rate/{video}/detections/raw_video.npz') as old:
                for key in ('xyxy','conf','cls','frame_idx','frame_count'):assert np.array_equal(z[key],old[key]),('Detector replay mismatch',key)
    if not probe:
        write(folder/'summary.json',dict(protocol=dict(video_id=video,source_fps=meta['fps'],raw_frames=meta['decoded_frame_count'],video_sha256=sha(ROOT/f'data/bdd100k/videos/val/{video}.mov')),alignment=read(folder/'alignment.json'),detector=result))
        write(folder/'motion/flyvis_tiled.json',dict(fps=meta['fps'],metadata_only=True,note='Only fps for fixed signature builder. No full-frame FlyVis motion was computed.'))
    outputs=[folder/'detections/raw_video.npz',folder/'detections/raw_video.json']
    if not probe:outputs += [folder/'alignment.json',folder/'summary.json',folder/'motion/flyvis_tiled.json']
    write(STAGE/'jobs'/('probe.json' if probe else video+'.json'),dict(at=now().isoformat(),video=video,detector_replay=probe,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in outputs}))


def child(video,probe=False):
    marker=STAGE/'jobs'/('probe.json' if probe else video+'.json')
    if marker.exists():
        data=read(marker)
        for n,h in data['outputs'].items():assert sha(ROOT/n)==h
        return data
    log=STAGE/'logs'/('probe' if probe else video);log.parent.mkdir(parents=True,exist_ok=True)
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        start=time.monotonic();proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',video,*(['--probe'] if probe else [])],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        while proc.poll() is None:check();time.sleep(.5)
        assert proc.returncode==0,('Input child failed',video,probe,proc.returncode)
    data=read(marker);data['wall_seconds_with_imports']=time.monotonic()-start;write(marker,data);return data


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(STAGE/'protocol.json');verify_sources(p);child(p['probe_video'],True)
    for i,video in enumerate(p['videos']):
        m=child(video)
        if i==0:
            estimate=2*m['wall_seconds_with_imports']*19+120;available=p['budget_seconds']-(time.monotonic()-T0)
            write(STAGE/'price.json',dict(first_video_seconds=m['wall_seconds_with_imports'],conservative_remaining_seconds=estimate,available_seconds=available,proceed=estimate<available));assert estimate<available,'Input preparation does not fit measured original budget'
    verify_sources(p);jobs=list((STAGE/'jobs').glob('*.json'));assert len(jobs)==21
    write(STAGE/'report.json',dict(videos=p['videos'],total_frames=sum(p['frames'][v] for v in p['videos']),jobs=21,detector_replay_exact=True,metadata_and_detection_hashes={path.name:read(path)['outputs'] for path in jobs},no_hota_or_training=True,limitations=p['limitations']))
    write(STAGE/'status.json',dict(state='completed',stage='20входныхкэшей готовы;HOTAнеизмерялась',updated=now().isoformat(),jobs_done=21,jobs_total=21))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:job(sys.argv[2], '--probe' in sys.argv)
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit input stage mode required')
