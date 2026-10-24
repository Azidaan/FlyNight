"""Ten frozen systems, fixed additional20val, one shared MASA and evaluator."""
from pathlib import Path
from datetime import datetime
import os,sys,time,inspect,subprocess,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_matched_trackers_20261024 import make_tracker,ARMS
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',FOVEA_THREADS='2',FOVEA_GPU_LOCK='flyvis_fovea_gpu')
RUN=ROOT/'runs/z13_val20_matched_20261024';STAGE=RUN/'common';OLD=ROOT/'runs/z13_matched_trackers_20261024';IDENT='z13_val20_common_v1';T0=time.monotonic()
SYSTEMS=('xywh','bytetrack','botsort','tracktrack','legacyD','centerD','legacyC','centerC','legacyB','centerR1')
MODELS={**{a:'runs/z13_shared_masa_hota_v1_20261015/full/'+a[-1]+'/model.pt' for a in ('legacyD','legacyC','legacyB')},**{a:'runs/z13_center_hota_20261021/'+a.removeprefix('center')+'/config.json' for a in ('centerD','centerC','centerR1')}}


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as J
    old=J.IDENT
    try:J.IDENT=IDENT;J.journal(title,text)
    finally:J.IDENT=old


def check():
    p=read(STAGE/'protocol.json')
    if (STAGE/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Common20 own budget/Moscow deadline')


def status(stage,**kw):
    check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),jobs_done=len(list((STAGE/'jobs').glob('*.json'))),jobs_total=231,**kw))


def ended(folder):
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    e=read(folder/'controller_result.json');c=read(folder/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['worker_ended'] and e['all_owned_ended'] and not e['remaining'] and not e['hard_timeout']
    assert ps.creation(c['pid'])!=c['birth'] and all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not STAGE.exists();ended(RUN/'raw');ended(RUN/'inputs_recovery1');base=read(RUN/'raw/protocol.json');verify_sources(base)
    report=read(RUN/'raw/report.json');assert read(RUN/'raw/finish_recorded.json')['report_sha256']==sha(RUN/'raw/report.json')
    hashes=dict(base['sha256'])
    for stage,number in ((RUN/'raw',84),(RUN/'inputs_recovery1',21)):
        markers=list((stage/'jobs').glob('*.json'));assert len(markers)==number
        for f in markers:
            for n,h in read(f)['outputs'].items():assert sha(ROOT/n)==h;hashes[n]=h
    pv={a:{v:x['HOTA'] for v,x in rr.items()} for a,rr in report['per_video'].items()}
    for key,val in report['paired'].items():a,b=key.split('-');assert paired(pv[a],pv[b])==val
    names=['scripts/z13_val20_common_20261024.py','scripts/z13_val20_common_console.ps1','scripts/z13_val20_common_finish_20261024.py','scripts/z13_bounded_recovery_controller_20261022.py','scripts/build_fly_signatures.py','scripts/build_fly_signatures_val.py','scripts/z13_shared_masa.py','scripts/stitch_eval_v2.py','scripts/eval_fly_reid.py','tracking/z13_benchmark_adapter.py','tracking/z13_compiled_backend.py','tracking/z13_spatial_localizer.py','tracking/z13_ridge_localizer.py','tracking/z13_ridge_localizer_v2.py','tracking/z13_readout.py','tracking/masa_r50.py','tracking/gpu_lock.py','runs/z13_center_hota_20261021/masa_head.joblib','weights/masa/masa_r50.pth',*MODELS.values()]
    # Include the complete immutable scientific identity of earlier validated inference.
    prior=read(OLD/'gpu_recovery3/protocol.json');verify_sources(prior);hashes.update(prior['sha256'])
    identity=read(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/inference_identity.json')
    for a in ('legacyD','legacyC','legacyB'):assert sha(ROOT/MODELS[a])==identity[MODELS[a]]
    for arm in SYSTEMS:names.append((OLD/arm/'signatures'/base['probe_video']/'signatures.npz').relative_to(ROOT).as_posix())
    names.append((OLD/'xywh/masa'/(base['probe_video']+'.npz')).relative_to(ROOT).as_posix());names+=['runs/z13_val20_matched_20261024/raw/report.json','runs/z13_val20_matched_20261024/inputs_recovery1/report.json']
    names.remove('scripts/build_fly_signatures_val.py')
    for n in names:hashes[n]=sha(ROOT/n)
    auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    p=dict(id=IDENT,created=now().isoformat(),device='cuda',budget_seconds=86400,save_at=auth['save_at'],hard_stop=auth['hard_stop'],stop_request_path=(STAGE/'stop.request').relative_to(ROOT).as_posix(),videos=base['videos'],probe_video=base['probe_video'],frames=base['frames'],configs=base['configs'],systems=list(SYSTEMS),models=MODELS,sha256=hashes,
      action='Complete a wider fixed matched evaluation of existing frozen readouts/cores with native competitors. No new learned candidate, descriptor, loss or policy is promoted. Four baseline systems plus legacyD/centerD/legacyC/centerC/legacyB/centerR1. Each raw and with identical frozen base+MASA. This is evaluation, not a newly improved FlyVis version.',
      controls='C is the pretrained fixed FlyVis core with task-trained frozen readout; D is one saved adapted core. Their readout/policy weights and training differ. B is small learnedGRU, R1 temporal receptors, parameter counts differ. D-C is functional system comparison, not an isolated backbone/anatomy effect. Same four observed frames followed by continuing recurrent search history, max4eyes, fixed per-arm tau/lambda. Geometry-only tracker without any correction is included.',
      settings='Original-frame960/conf.01/iou.7 detections,high.25/low.1/new.25/buffer30, native matching/GMC defaults,ReIDoff. Same MASA R50weights and base_masa scorer trained earlier on130train,gap150,dist5,threshold.6. Native raw outputs equal independent CPU20 exactly. Strict original annotation mapping from verified cache,including conflict repair retained. No fitting/tuning on20val.',
      primary='Report all10 raw/MASA pooled HOTA and all per-video values. For each concrete benefit claim require paired-video95%CI strict lower>0 for the exact matched contrast. FlyVis-to-stand requires each D versus xywh; modernization versus stock additionally D versus corresponding C, and matched receptor/smallGRU contrast. No secondary comparison rescues a failed primary. Complete20 and all231jobs required; otherwise only technical remainder, no efficacy verdict.',
      gates='First old fixed control video: all10 raw output arrays exactly reproduce stored previous signatures;XYWH feature arrays and MASA baseline embeddings exactly repeat. Fresh process per signature/MASA,one namedGPUmutex. Then first fixed new video all10+MASA retained;2x its full walltime*remaining19+900seconds must fit remaining stage86400s and author deadline. Scientific thresholds unchanged. Stop after600s without frame/stage progress,retain completed jobs,no automatic repeated failed child.',
      budget_reason='New fixed20matched protocol,not extension of terminated7val stock10800s or raw20CPU3600s. Prior measured first6+MASA1247s plus first4controls826s gives conservative first10 upper~2073s;2x*19+900~79674s plus probe fits86400s. Actual first-new-video measured gate required. Authorized weekend endsMon26Oct07:40;absolute save07:35/hard07:39:30. No training/download/heldout75.',
      limitations=base['limitations']+' Additional20 were selected before outcomes, but are not untouched final test. Native algorithms not optimally tuned. MASA scorer trained on XYWH pairs may transfer unequally;show raw and MASA. One saved D is not a seed-independence result. Prior7 subset and these20 remain separate tables;do not compare their raw numbers as changes.')
    write(STAGE/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Общие20val: десять систем с одинаковой MASA и контролями C/B/R1',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['benchmark','fly-problem-P8'],question='Каковы FlyVis и внешние трекеры при одинаковых исходных кадрах,детекциях,оценке и сшивке на фиксированных20val?',hypothesis='Знак добавки D/штатной C и общий рейтинг заранее неизвестны;проверка устраняет смешение условий.',expected=p['action']+' '+p['controls'],acceptance=[dict(criterion=p['gates'],met=None),dict(criterion='Все20×10систем raw/MASA,231задание,первичные строки/хеши и независимая сверка;своё дерево закончено в бюджете.',met=None),dict(criterion=p['primary'],met=None)],summary='Зарегистрировано до общего сравнения;числа CPUraw20 не подменяют общий итог с FlyVis/MASA.',obtained='',interpretation='',decision='',cannot_claim=p['limitations']+' '+p['controls'],metrics=[],commits=[],files=names[:3],evidence=[dict(kind='file',ref=(STAGE/'protocol.json').relative_to(ROOT).as_posix(),claim='Фиксированные условия,критерии,контроли и бюджет до запуска'),dict(kind='file',ref='runs/z13_val20_matched_20261024/raw/report.json',claim='Готовая исходная ступень20val без FlyVis/MASA')]));write(cp,cat)
    journal('зарегистрирована общая проверка десяти систем на20val','*Постановка.* '+p['action']+'\n\n'+p['controls']+'\n\n*Критерии до прогона.* '+p['primary']+'\n\n'+p['gates']+'\n\n*Бюджет.* '+p['budget_reason']+'\n\n*Ограничения.* '+p['limitations']+'\n\nИсходная ступень20val уже завершенаexit0: raw/report.json. Парные интервалы и84готовыхмаркера проверены;эти вычисления не повторяются,кроме обязательного точного контроля выходов при извлечении сигнатур.')


def build_signature(arm,video,probe=False):
    import numpy as np
    from tracking.z13_compiled_backend import setup,enable
    setup()
    import torch
    torch.set_num_threads(2)
    from scripts import build_fly_signatures as B
    from tracking import z13_readout
    from tracking.gpu_lock import GpuLock
    p=read(STAGE/'protocol.json');output=(STAGE/'probe' if probe else RUN)/arm/'signatures';cache=ROOT/'runs/video_rate' if probe else RUN/'cache';argslist=['builder','--video-id',video,'--videos',str(ROOT/'data/bdd100k/videos/val'),'--detections-root',str(cache),'--flow-root',str(cache),'--output',str(output),'--max-eyes','4','--skip-fly']
    if arm in MODELS:
        argslist+=['--z13',str(ROOT/MODELS[arm])]
        if arm.startswith('legacy'):from tracking.z13_spatial_localizer import SpatialLocalizer as Base
        else:from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer as Base
        class Fixed(Base):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                if self.arm!='R1':self.compiler_info=enable(self)
        z13_readout.Localizer=Fixed
    oldargv=sys.argv
    try:sys.argv=argslist;args=B.parse_args()
    finally:sys.argv=oldargv
    source=inspect.getsource(B.build_video);ns=vars(B).copy()
    if arm in ARMS and arm!='xywh':
        from tracking.z13_benchmark_adapter import TrackerView
        ns['_benchmark_factory']=lambda unused:TrackerView(make_tracker(arm,p['configs']));a='from tracking.xywh_bytetrack import XYWHByteTrack';b='output = tracker.update(_DetSlice(xyxy, conf, classes))';assert source.count(a)==source.count(b)==1;source=source.replace(a,'XYWHByteTrack = _benchmark_factory').replace(b,'output = tracker.update(_DetSlice(xyxy, conf, classes), img=frame)')
    def progress(*values,**kw):
        text=' '.join(map(str,values));print(*values,**kw);m=re.match(r'\['+re.escape(video)+r'\] (\d+) ',text)
        if m:status('Контроль прежнего видео' if probe else 'Неизменные сигнатуры',variant=arm+' '+video,done=int(m.group(1)),total=p['frames'][video])
    ns['print']=progress;exec(compile(source,str(ROOT/'scripts/build_fly_signatures.py'),'exec'),ns)
    with GpuLock('flyvis_fovea_gpu'):ns['build_video'](video,args)
    path=output/video/'signatures.npz'
    with np.load(path) as actual:
        assert np.isfinite(actual['outputs']).all() and int(actual['frame_count'])==p['frames'][video]
        if probe:
            with np.load(OLD/arm/'signatures'/video/'signatures.npz') as old:
                keys=('track_id','start','end','first_box','last_box','first_hexin','last_hexin','first_thumb','last_thumb','first_hist','last_hist','outputs') if arm=='xywh' else ('outputs',)
                for key in keys:assert np.array_equal(actual[key],old[key]),('Old control replay',arm,key)
        elif arm in ARMS:
            with np.load(RUN/'raw'/arm/'raw'/(video+'.npz')) as old:assert np.array_equal(actual['outputs'],old['outputs']),('Native CPU20 replay',arm,video)
    return dict(signature_sha256=sha(path),probe=probe,exact_replay=probe or arm in ARMS)


def build_masa(video,probe=False):
    import numpy as np,torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from tracking.masa_r50 import build
    from tracking.gpu_lock import GpuLock
    from scripts.z13_shared_masa import extract_video
    base=STAGE/'probe' if probe else RUN;paths={a:base/a/'signatures'/video/'signatures.npz' for a in SYSTEMS};outputs={a:base/a/'masa'/(video+'.npz') for a in SYSTEMS}
    with GpuLock('flyvis_fovea_gpu'):info=extract_video(build(),ROOT/f'data/bdd100k/videos/val/{video}.mov',paths,outputs,progress=lambda i,n:status('Общая замороженная MASA',variant=video,done=i,total=n))
    if probe:
        with np.load(outputs['xywh']) as actual,np.load(OLD/'xywh/masa'/(video+'.npz')) as old:
            for key in ('first','last'):assert np.array_equal(actual[key],old[key]),('Old MASA replay',key)
    return info


def child(mode,arm,video,probe=False):
    marker=STAGE/'jobs'/f'{"probe_" if probe else ""}{mode}_{arm}_{video}.json'
    if marker.exists():
        m=read(marker)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
        return m
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();log=STAGE/'logs'/marker.stem;log.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',mode,arm,video,*(['--probe'] if probe else [])],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);birth=ps.creation(proc.pid)
        while proc.poll() is None:
            check()
            if (now()-datetime.fromisoformat(read(STAGE/'status.json')['updated'])).total_seconds()>600:
                if ps.creation(proc.pid)==birth:subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                raise TimeoutError('600s without frame/stage progress; no automatic retry')
            time.sleep(.5)
        assert proc.returncode==0,('Common child failed',mode,arm,video,probe,proc.returncode)
    m=read(marker);m['wall_seconds_with_imports']=time.monotonic()-began;write(marker,m);return m


def evaluate():
    import numpy as np,joblib
    from scripts import stitch_eval_v2 as V
    from scripts.eval_fly_reid import stitch_video
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p=read(STAGE/'protocol.json');frozen=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib');model=frozen['model'];cols=frozen['cols'];assert cols==[V.NAMES.index(c) for c in V.HEADS['base_masa']];V._set(150,5.,True);per={};pooled={};hashes={};links={}
    for arm in SYSTEMS:
        per[arm+'_raw']={};per[arm+'_masa']={};links[arm]={};dino=RUN/arm/'dino_compat';dino.mkdir(exist_ok=True);V.WIN['dino_dir']=dino;V.WIN['masa_dir']=RUN/arm/'masa'
        for v in p['videos']:
            status('Общая сшивка и официальная HOTA',variant=arm+' '+v)
            with np.load(RUN/arm/'signatures'/v/'signatures.npz') as z:n=len(z['track_id'])
            compat=dino/(v+'.npz')
            if not compat.exists():np.savez_compressed(compat,first=np.zeros((n,1),np.float16),last=np.zeros((n,1),np.float16))
            d=V.video_pairs((v,str(RUN/arm/'signatures'),str(RUN/'cache'),'val'));scores=model.predict_proba(d['X'][:,cols])[:,1] if len(d['pairs']) else np.zeros(0);assert np.isfinite(scores).all()
            for kind,threshold in (('raw',2.),('masa',.6)):
                folder=RUN/arm/('evaluated_'+kind);r=stitch_video((v,str(RUN/arm/'signatures'),str(RUN/'cache'),'val',d['pairs'],scores,threshold,str(folder)))
                if kind=='masa':links[arm][v]=r['links']
                if arm in ARMS and kind=='raw':assert read(folder/(v+'.json'))==read(RUN/'raw'/arm/'raw'/(v+'.json'))
                per[arm+'_'+kind][v]=evaluate_hota(folder,[v],split='val');hashes[(folder/(v+'.json')).relative_to(ROOT).as_posix()]=sha(folder/(v+'.json'))
        for kind in ('raw','masa'):pooled[arm+'_'+kind]=evaluate_hota(RUN/arm/('evaluated_'+kind),p['videos'],split='val')
        write(STAGE/'evaluated_partial.json',dict(per_video=per,pooled=pooled,links=links))
    pv={k:{v:m['HOTA'] for v,m in rows.items()} for k,rows in per.items()};contrasts={}
    for kind in ('raw','masa'):
        for a in SYSTEMS:
            if a!='xywh':contrasts[a+'_'+kind+'-xywh_'+kind]=paired(pv[a+'_'+kind],pv['xywh_'+kind])
        for a,b in (('legacyD','legacyC'),('centerD','centerC'),('legacyD','legacyB'),('centerD','centerR1'),('legacyC','legacyB'),('centerC','centerR1')):contrasts[a+'_'+kind+'-'+b+'_'+kind]=paired(pv[a+'_'+kind],pv[b+'_'+kind])
    for a in SYSTEMS:contrasts[a+'_masa-'+a+'_raw']=paired(pv[a+'_masa'],pv[a+'_raw'])
    from scripts.z13_localization_labels_20261023 import verify_sources
    verify_sources(p);write(STAGE/'report.json',dict(videos=p['videos'],pooled=pooled,per_video=per,paired=contrasts,output_sha256=hashes,masa_links=links,all_conditions_common=True,independent_final_test=False,limitations=p['limitations']))


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(STAGE/'protocol.json');verify_sources(p)
    for arm in SYSTEMS:status('Начальное точное воспроизведение',variant=arm+' '+p['probe_video']);child('signature',arm,p['probe_video'],True)
    child('masa','all',p['probe_video'],True)
    for i,v in enumerate(p['videos']):
        began=time.monotonic()
        for arm in SYSTEMS:status('Общее сравнение20val',variant=arm+' '+v);child('signature',arm,v)
        child('masa','all',v)
        if i==0:
            measured=time.monotonic()-began;estimate=2*measured*19+900;available=min(p['budget_seconds']-(time.monotonic()-T0),(datetime.fromisoformat(p['save_at'])-now()).total_seconds());write(STAGE/'price.json',dict(first_new_video_seconds=measured,conservative_remaining_seconds=estimate,available_seconds=available,proceed=estimate<available));assert estimate<available,'Common20 price exceeds original budget'
    evaluate();write(STAGE/'status.json',dict(state='completed',stage='10систем×20val,raw/MASAготовы',updated=now().isoformat(),jobs_done=231,jobs_total=231))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:
        mode,arm,v=sys.argv[2:5];probe='--probe' in sys.argv;p=read(STAGE/'protocol.json');assert v in ([p['probe_video']] if probe else p['videos']) and arm in SYSTEMS+('all',)
        result=build_signature(arm,v,probe) if mode=='signature' else build_masa(v,probe);base=STAGE/'probe' if probe else RUN;paths=[base/arm/'signatures'/v/'signatures.npz'] if mode=='signature' else [base/a/'masa'/(v+'.npz') for a in SYSTEMS];write(STAGE/'jobs'/f'{"probe_" if probe else ""}{mode}_{arm}_{v}.json',dict(at=now().isoformat(),result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths}))
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit common20 mode required')
