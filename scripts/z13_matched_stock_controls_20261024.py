"""Frozen stock-core and no-FlyVis controls, serially after the matched benchmark.

This extension never edits the running six-system protocol or its primary report.
"""
from pathlib import Path
from datetime import datetime
import os,sys,time,subprocess,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_trackers_20261024 import RUN,VIDEOS,read,write,sha,now
from scripts import z13_matched_pipeline_20261024 as P
os.environ['CUDA_VISIBLE_DEVICES']='0'
STAGE=RUN/'stock_controls';IDENT='z13_matched_stock_controls_v1';T0=time.monotonic()
SYSTEMS=('legacyC','centerC','legacyB','centerR1')
MODELS={
 'legacyC':'runs/z13_shared_masa_hota_v1_20261015/full/C/model.pt',
 'centerC':'runs/z13_center_hota_20261021/C/config.json',
 'legacyB':'runs/z13_shared_masa_hota_v1_20261015/full/B/model.pt',
 'centerR1':'runs/z13_center_hota_20261021/R1/config.json'}


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as J
    previous=J.IDENT
    try:J.IDENT=IDENT;J.journal(title,text)
    finally:J.IDENT=previous


def check():
    p=read(STAGE/'protocol.json')
    if (STAGE/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Stock controls original budget/Moscow deadline')


def status(stage,**kw):
    check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),jobs_done=len(list((STAGE/'jobs').glob('*.json'))),jobs_total=35,**kw))


def prepare():
    assert not STAGE.exists(),'Existing extension; no duplicate'
    parent=read(P.STAGE/'protocol.json');identity=read(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/inference_identity.json')
    for arm in ('legacyC','legacyB'):assert sha(ROOT/MODELS[arm])==identity[MODELS[arm]]
    for arm in ('centerC','centerR1'):
        config=read(ROOT/MODELS[arm]);assert config['arm']==arm.removeprefix('center') and config['format']=='z13-center-ridge-hota-v1'
    names=['scripts/z13_matched_stock_controls_20261024.py','scripts/z13_matched_stock_queue_20261024.py','scripts/z13_matched_stock_console.ps1','tests/test_z13_matched_stock_controls.py',*MODELS.values()]
    p=dict(id=IDENT,created=now().isoformat(),device='cuda',budget_seconds=10800,save_at=parent['save_at'],hard_stop=parent['hard_stop'],stop_request_path=(STAGE/'stop.request').relative_to(ROOT).as_posix(),videos=list(VIDEOS),systems=list(SYSTEMS),models=MODELS,
      parent_protocol_sha256=sha(P.STAGE/'protocol.json'),base_protocol_sha256=sha(RUN/'protocol.json'),
      authorization='Author asks how stock FlyVis compares, previously requests identical conditions and autonomous necessary runs; frozen inference only, no new fitting or data.',
      method='Same original video builder, four observed frames followed by recurrent search history, max4eyes, retina/lamina, existing frozen per-arm weights/normalization/lambda/tau. Legacy C/B use identical spatial architecture and their saved trained readouts; center C/R1 use identical ridge algorithm with separately preselected training-only configurations. C is pretrained frozen core, not an unadapted complete tracker. B is learned smallGRU, R1 temporal receptor control: neither equals parameter-matched FlyVis. C/D differences include individually trained readout/policy weights; this is not an isolated backbone-only causal effect.',
      execution='Queue waits for parent controller exit0 and all own processes ended. Independent controller then10800seconds or absolute save time, whichever earlier; one GPU child at a time. No completed main six-system jobs repeated or overwritten. First complete fixed video four signatures+commonMASA retained;2x measured first-video wall*7+600 must fit remaining budget. All35jobs and7videos required for efficacy interpretation.',
      contrasts=['legacyD-legacyC','centerD-centerC','legacyD-legacyB','centerD-centerR1','legacyC-legacyB','centerC-centerR1','legacyC-xywh','centerC-xywh'],
      criterion='For each particular positive claim paired-video95%CI lower>0; pooled and mean-video differences reported separately. No tuning based on these seven videos; all outcomes shown.',
      limitations=read(RUN/'protocol.json')['limitations'],sha256={**parent['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(STAGE/'protocol.json',p);write(STAGE/'status.json',dict(state='queued',stage='Ожидание конца основного сравнения; GPU ещё не занята контролями',updated=now().isoformat(),jobs_done=0,jobs_total=35))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: штатная FlyVis C и контроли без мухи в общем сравнении7val',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','benchmark'],question='Каковы результаты штатного ядра C и обученной D при одинаковом окружении и способе считывания; отличается ли добавка мухи от контроля без неё?',hypothesis='Знак отличий заранее неизвестен; прежние35dev не заменяют новое общее сравнение.',expected=p['method'],acceptance=[dict(criterion='Parent six-system run complete exit0; sources,weights,inputs frozen; all35controljobs and7videos complete.',met=None),dict(criterion='Baseline MASA features exactly reproduced for every video; same scorer and annotation projection.',met=None),dict(criterion=p['criterion'],met=None)],summary='Отдельное расширение до измерения C/B/R1; основной протокол и report не меняются.',obtained='',interpretation='',decision='',cannot_claim=p['limitations']+' '+p['method'],metrics=[],commits=[],files=names[:3],evidence=[dict(kind='file',ref=(STAGE/'protocol.json').relative_to(ROOT).as_posix(),claim='Условия и сравнения до новых результатов')]))
    write(cp,cat);journal('добавлены штатное ядро и контроли без мухи','*Постановка.* Автор спрашивает о штатной FlyVis и узком горлышке. '+p['method']+'\n\nОтдельные4контроля на тех же7val; ожидание завершения основного6системного расчёта. До результата критерий: '+p['criterion']+'\n\nБюджет10800с после очереди,абсолютное сохранение07:25и конец07:39:30МСК24.10; цена первого полного ролика до оставшихся6,старые результаты не перезаписываются. Новых данных/обучения/подбора наval нет. Итогов пока нет.')


def build_signature(arm,video):
    import numpy as np
    from tracking.z13_compiled_backend import setup,enable
    setup()
    import torch
    torch.set_num_threads(2)
    from scripts import build_fly_signatures as builder
    from tracking import z13_readout
    from tracking.gpu_lock import GpuLock
    if arm.startswith('legacy'):from tracking.z13_spatial_localizer import SpatialLocalizer as Base
    else:from tracking.z13_ridge_localizer import CenterRidgeLocalizer as Base
    class Fixed(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.arm!='R1':self.compiler_info=enable(self)
    z13_readout.Localizer=Fixed
    old=sys.argv
    try:
        sys.argv=['builder','--video-id',video,'--videos',str(ROOT/'data/bdd100k/videos/val'),'--detections-root',str(ROOT/'runs/video_rate'),'--flow-root',str(ROOT/'runs/video_rate'),'--output',str(RUN/arm/'signatures'),'--max-eyes','4','--skip-fly','--z13',str(ROOT/MODELS[arm])]
        args=builder.parse_args()
    finally:sys.argv=old
    # The original function resolves Localizer dynamically from z13_readout.
    with GpuLock('flyvis_fovea_gpu'):builder.build_video(video,args)
    path=RUN/arm/'signatures'/video/'signatures.npz'
    with np.load(path) as z:assert int(z['frame_count'])==read(RUN/'protocol.json')['frames'][video] and np.isfinite(z['outputs']).all()
    return dict(signature_sha256=sha(path),frozen_model=MODELS[arm])


def build_masa(video):
    import numpy as np,torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from tracking.masa_r50 import build
    from tracking.gpu_lock import GpuLock
    from scripts.z13_shared_masa import extract_video
    inputs={a:RUN/a/'signatures'/video/'signatures.npz' for a in SYSTEMS}
    outputs={a:RUN/a/'masa'/(video+'.npz') for a in SYSTEMS}
    inputs['xywh']=RUN/'xywh/signatures'/video/'signatures.npz';outputs['xywh']=STAGE/'baseline_masa_replay'/(video+'.npz')
    with GpuLock('flyvis_fovea_gpu'):
        info=extract_video(build(),ROOT/f'data/bdd100k/videos/val/{video}.mov',inputs,outputs,progress=lambda i,n:status('Общая MASA, включая контроль воспроизведения',variant=video,done=i,total=n))
    with np.load(outputs['xywh']) as actual,np.load(RUN/'xywh/masa'/(video+'.npz')) as parent:
        for key in ('first','last'):assert np.array_equal(actual[key],parent[key]),('Stock extension baseline MASA replay',video,key)
    info['baseline_embedding_replay']=True;return info


def child(mode,arm,video):
    # Frozen main helper has the same bounded child/marker contract, with paths
    # redirected only to this separate extension and its explicit script.
    marker=STAGE/'jobs'/f'{mode}_{arm}_{video}.json'
    if marker.exists():
        data=read(marker)
        for n,h in data['outputs'].items():assert sha(ROOT/n)==h
        return data
    log=STAGE/'logs'/f'{mode}_{arm}_{video}';log.parent.mkdir(parents=True,exist_ok=True)
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        t=time.monotonic();proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',mode,arm,video],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        while proc.poll() is None:check();time.sleep(.5)
        assert proc.returncode==0,('Stock child failed',mode,arm,video,proc.returncode)
    data=read(marker);data['wall_seconds_with_imports']=time.monotonic()-t;write(marker,data);return data


def evaluate():
    import numpy as np,joblib
    from scripts import stitch_eval_v2 as V
    from scripts.eval_fly_reid import stitch_video
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    parent=read(RUN/'report.json');assert parent['videos']==list(VIDEOS) and parent['all_conditions_common']
    frozen=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib');assert frozen['cols']==[V.NAMES.index(c) for c in V.HEADS['base_masa']]
    per=dict(parent['per_video']);pooled=dict(parent['pooled']);links=dict(parent['masa_links']);hashes=dict(parent['output_sha256']);V._set(150,5.,True)
    for arm in SYSTEMS:
        per[arm+'_raw']={};per[arm+'_masa']={};links[arm]={};dino=RUN/arm/'dino_compat';dino.mkdir(exist_ok=True);V.WIN['dino_dir']=dino;V.WIN['masa_dir']=RUN/arm/'masa'
        for video in VIDEOS:
            status('Общий оценщик HOTA для дополнительных контролей',variant=arm+' '+video)
            with np.load(RUN/arm/'signatures'/video/'signatures.npz') as z:n=len(z['track_id'])
            compat=dino/(video+'.npz')
            if not compat.exists():np.savez_compressed(compat,first=np.zeros((n,1),np.float16),last=np.zeros((n,1),np.float16))
            data=V.video_pairs((video,str(RUN/arm/'signatures'),str(ROOT/'runs/video_rate'),'val'));scores=frozen['model'].predict_proba(data['X'][:,frozen['cols']])[:,1] if len(data['pairs']) else np.zeros(0);assert np.isfinite(scores).all()
            for kind,threshold in (('raw',2.),('masa',.6)):
                folder=RUN/arm/('evaluated_'+kind);result=stitch_video((video,str(RUN/arm/'signatures'),str(ROOT/'runs/video_rate'),'val',data['pairs'],scores,threshold,str(folder)))
                if kind=='masa':links[arm][video]=result['links']
                per[arm+'_'+kind][video]=evaluate_hota(folder,[video],split='val');hashes[(folder/(video+'.json')).relative_to(ROOT).as_posix()]=sha(folder/(video+'.json'))
        for kind in ('raw','masa'):pooled[arm+'_'+kind]=evaluate_hota(RUN/arm/('evaluated_'+kind),list(VIDEOS),split='val')
        write(STAGE/'evaluated_partial.json',dict(per_video=per,pooled=pooled))
    comparisons=dict(parent['paired']);pv={k:{v:m['HOTA'] for v,m in values.items()} for k,values in per.items()}
    for kind in ('raw','masa'):
        for arm in SYSTEMS:comparisons[f'{arm}_{kind}-xywh_{kind}']=paired(pv[arm+'_'+kind],pv['xywh_'+kind])
        for contrast in read(STAGE/'protocol.json')['contrasts']:
            a,b=contrast.split('-');comparisons[f'{a}_{kind}-{b}_{kind}']=paired(pv[a+'_'+kind],pv[b+'_'+kind])
    from scripts.z13_localization_labels_20261023 import verify_sources
    verify_sources(read(STAGE/'protocol.json'))
    write(STAGE/'report.json',dict(videos=list(VIDEOS),pooled=pooled,per_video=per,paired=comparisons,masa_links=links,output_sha256=hashes,parent_report_sha256=sha(RUN/'report.json'),all_conditions_common=True,independent_final_test=False,limitations=read(STAGE/'protocol.json')['limitations']))


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_record_matched_trackers_20261024 import ended
    ended(P.STAGE);p=read(STAGE/'protocol.json');verify_sources(p);assert sha(P.STAGE/'protocol.json')==p['parent_protocol_sha256'];first=time.monotonic()
    for i,video in enumerate(VIDEOS):
        for arm in SYSTEMS:status('Штатная муха и контроли без мухи',variant=arm+' '+video,done=i,total=7);child('signature',arm,video)
        child('masa','all',video)
        if i==0:
            measured=time.monotonic()-first;estimate=2*measured*7+600;available=min(p['budget_seconds']-(time.monotonic()-T0),(datetime.fromisoformat(p['save_at'])-now()).total_seconds());write(STAGE/'price.json',dict(first_complete_video_seconds=measured,conservative_seconds=estimate,available_seconds=available,proceed=estimate<available));assert estimate<available,'Stock full comparison does not fit remaining original window'
    evaluate();write(STAGE/'status.json',dict(state='completed',stage='Все дополнительные контроли и общий отчёт готовы',updated=now().isoformat(),jobs_done=35,jobs_total=35))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:
        mode,arm,video=sys.argv[2:5];assert video in VIDEOS and (arm in SYSTEMS or (mode=='masa' and arm=='all'))
        result=build_signature(arm,video) if mode=='signature' else build_masa(video)
        paths=[RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [RUN/a/'masa'/(video+'.npz') for a in SYSTEMS]+[STAGE/'baseline_masa_replay'/(video+'.npz')]
        write(STAGE/'jobs'/f'{mode}_{arm}_{video}.json',dict(result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths},at=now().isoformat()))
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit extension mode required')
