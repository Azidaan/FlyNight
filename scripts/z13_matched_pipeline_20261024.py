"""One-GPU common signature/MASA pipeline followed by fixed CPU evaluation."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,subprocess,inspect,re,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_trackers_20261024 import RUN,VIDEOS,ARMS,read,write,sha,now,journal,make_tracker
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['FOVEA_THREADS']='2';os.environ['FOVEA_GPU_LOCK']='flyvis_fovea_gpu'
STAGE=RUN/'gpu_stage';SYSTEMS=('legacyD','centerD','xywh','bytetrack','botsort','tracktrack');T0=time.monotonic()
LEGACY=ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/D/model.pt';CENTER=ROOT/'runs/z13_center_hota_20261021/D/config.json'


def check():
    p=read(STAGE/'protocol.json')
    if (STAGE/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Common comparison GPU-stage budget/Moscow deadline')


def status(stage,**kw):
    check();write(STAGE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not STAGE.exists();base=read(RUN/'protocol.json')
    assert len(list((RUN/'xywh/raw').glob('*.done.json')))==7,'First complete raw baseline must exist'
    deadline=datetime.fromisoformat(base['save_at'])-timedelta(minutes=10);remaining=int((deadline-now()).total_seconds());assert remaining>900
    files=['scripts/z13_matched_pipeline_20261024.py','scripts/z13_matched_gpu_console.ps1','tracking/z13_benchmark_adapter.py','tests/test_z13_benchmark_adapter.py','tracking/z13_spatial_localizer.py','tracking/z13_ridge_localizer.py','tracking/z13_compiled_backend.py','tracking/masa_r50.py','tracking/gpu_lock.py','scripts/eval_fly_reid.py','weights/masa/masa_r50.pth',LEGACY.relative_to(ROOT).as_posix(),CENTER.relative_to(ROOT).as_posix()]
    old=read(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/inference_identity.json')
    assert sha(LEGACY)==old[LEGACY.relative_to(ROOT).as_posix()]
    files.append('runs/z13_shared_masa_hota_v1_20261015/full/inference_identity.json')
    p=dict(id=base['id'],created=now().isoformat(),device='cuda',budget_seconds=min(18000,remaining),save_at=deadline.isoformat(),hard_stop=base['hard_stop'],stop_request_path=(STAGE/'stop.request').relative_to(ROOT).as_posix(),videos=list(VIDEOS),systems=list(SYSTEMS),base_protocol_sha256=sha(RUN/'protocol.json'),legacy_checkpoint=LEGACY.relative_to(ROOT).as_posix(),center_config=CENTER.relative_to(ROOT).as_posix(),
      execution='One CUDA child at a time under named GPU mutex. Separate interpreter per signature/MASA job; Idle. Existing frozen compiled Euler adapter and original per-target readouts. All native outputs must equal independent CPU raw runs exactly; baseline also equals old7val. Common unchanged MASA head for every system. No thresholds/weights fitted or selected on7val.',
      technical_price='Fixed first complete video for all6systems and common MASA retained. Conservative2x first-video wall*time7+600seconds must fit stage remaining budget before remaining6videos. This price is not an efficacy decision. Already completed jobs not repeated.',
      source_transform='Native signature extraction only: replace tracker factory by read-only native state view and pass current original BGR frame for GMC. XYAH->XYWH affects only appearance crops, never native state/output; tested. Original XYWH/D builder unchanged. All arrays validated.',
      sha256={**base['sha256'],**{n:sha(ROOT/n) for n in files}})
    write(STAGE/'protocol.json',p);journal('зарегистрированы общие стадии D и MASA','Протокол runs/z13_matched_trackers_20261024/gpu_stage/protocol.json. Все6систем на тех же7val; обе D используют прежние зафиксированные веса/пороги, MASA-голова из прежнего train130 едина для всех. Сначала один полный заранее выбранный ролик всех6систем и общая MASA для цены/воспроизведения; сохраняется. Затем только если2xцена всех7+600с помещается в оставшийся бюджет, остальные6. Один CUDA-процесс через отдельные дочерние интерпретаторы и мьютекс; CPU-оценка после GPU. Нативные выходы совпадают с отдельным CPU-прогоном, состояние XYAH переводится только для вырезок без изменения трекера. Тест адаптера пройден. До07:25сохранениеGPUстадии, до07:40полныйконец; не более18000с. Скорость под конкурирующейCPUнагрузкой не считается сравнением быстродействия.')


def build_signature(arm,video):
    import numpy as np
    from tracking.z13_compiled_backend import setup
    setup()
    import torch
    torch.set_num_threads(2)
    from scripts import build_fly_signatures as builder
    from tracking import z13_readout
    from tracking.gpu_lock import GpuLock
    output=RUN/arm/'signatures';oldargv=sys.argv
    try:
        sys.argv=['builder','--video-id',video,'--videos',str(ROOT/'data/bdd100k/videos/val'),'--detections-root',str(ROOT/'runs/video_rate'),'--flow-root',str(ROOT/'runs/video_rate'),'--output',str(output),'--max-eyes','4','--skip-fly']
        if arm in ('legacyD','centerD'):sys.argv+=['--z13',str(LEGACY if arm=='legacyD' else CENTER)]
        args=builder.parse_args()
    finally:sys.argv=oldargv
    if arm in ('legacyD','centerD'):
        from tracking.z13_compiled_backend import enable
        if arm=='legacyD':from tracking.z13_spatial_localizer import SpatialLocalizer as Base
        else:from tracking.z13_ridge_localizer import CenterRidgeLocalizer as Base
        class Compiled(Base):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);self.compiler_info=enable(self)
        z13_readout.Localizer=Compiled
    source=inspect.getsource(builder.build_video);ns=vars(builder).copy()
    if arm in ARMS and arm!='xywh':
        from tracking.z13_benchmark_adapter import TrackerView
        configs=read(RUN/'protocol.json')['configs'];ns['_benchmark_factory']=lambda unused:TrackerView(make_tracker(arm,configs))
        a='from tracking.xywh_bytetrack import XYWHByteTrack';b='output = tracker.update(_DetSlice(xyxy, conf, classes))'
        assert source.count(a)==source.count(b)==1
        source=source.replace(a,'XYWHByteTrack = _benchmark_factory').replace(b,'output = tracker.update(_DetSlice(xyxy, conf, classes), img=frame)')
    def progress_print(*values,**kw):
        text=' '.join(map(str,values));print(*values,**kw)
        m=re.match(r'\['+re.escape(video)+r'\] (\d+) ',text)
        if m:status('Общие сигнатуры, исходные кадры',variant=arm+' '+video,done=int(m.group(1)),total=read(RUN/'protocol.json')['frames'][video])
    ns['print']=progress_print;exec(compile(source,str(ROOT/'scripts/build_fly_signatures.py'),'exec'),ns)
    with GpuLock('flyvis_fovea_gpu'):ns['build_video'](video,args)
    path=output/video/'signatures.npz'
    with np.load(path) as z:
        assert np.isfinite(z['outputs']).all() and int(z['frame_count'])==read(RUN/'protocol.json')['frames'][video]
        if arm in ARMS:
            while not (RUN/arm/'raw'/(video+'.done.json')).exists():check();time.sleep(1)
            with np.load(RUN/arm/'raw'/(video+'.npz')) as old:assert np.array_equal(z['outputs'],old['outputs']),(arm,video,'CPU/native replay')
        if arm=='xywh':
            with np.load(ROOT/f'runs/fly_signatures_val/{video}/signatures.npz') as old:
                for key in ('track_id','start','end','first_box','last_box','first_hexin','last_hexin','first_thumb','last_thumb','first_hist','last_hist','outputs'):assert np.array_equal(z[key],old[key]),('Baseline signature replay',key,video)
    return dict(signature_sha256=sha(path),raw_replay=arm in ARMS,baseline_feature_replay=arm=='xywh')


def build_masa(video):
    import numpy as np,torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from tracking.masa_r50 import build
    from tracking.gpu_lock import GpuLock
    from scripts.z13_shared_masa import extract_video
    signatures={a:RUN/a/'signatures'/video/'signatures.npz' for a in SYSTEMS};outputs={a:RUN/a/'masa'/(video+'.npz') for a in SYSTEMS}
    with GpuLock('flyvis_fovea_gpu'):
        model=build();info=extract_video(model,ROOT/f'data/bdd100k/videos/val/{video}.mov',signatures,outputs,progress=lambda i,n:status('Единые признаки MASA',variant=video,done=i,total=n))
    with np.load(outputs['xywh']) as actual,np.load(ROOT/f'runs/masa_embeddings/{video}.npz') as old:
        for key in ('first','last'):assert np.array_equal(actual[key],old[key]),('Baseline MASA replay',video,key)
    info['baseline_embedding_replay']=True;return info


def child(mode,arm,video):
    marker=STAGE/'jobs'/(mode+'_'+arm+'_'+video+'.json')
    if marker.exists():
        data=read(marker)
        for n,h in data['outputs'].items():assert sha(ROOT/n)==h,n
        return data
    log=STAGE/'logs'/(mode+'_'+arm+'_'+video);log.parent.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)), '--job',mode,arm,video]
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        began=time.monotonic();p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        while p.poll() is None:check();time.sleep(.5)
        assert p.returncode==0,('GPU child failed',mode,arm,video,p.returncode)
    assert marker.exists();data=read(marker);data['wall_seconds_with_imports']=time.monotonic()-began;write(marker,data);return data


def evaluate():
    import numpy as np,joblib
    from scripts import stitch_eval_v2 as V
    from scripts.eval_fly_reid import stitch_video
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    frozen=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib');model=frozen['model'];cols=frozen['cols'];assert cols==[V.NAMES.index(c) for c in V.HEADS['base_masa']]
    per={};pooled={};links={};hashes={};V._set(150,5.,True)
    for arm in SYSTEMS:
        per[arm+'_raw']={};per[arm+'_masa']={};links[arm]={}
        dino=RUN/arm/'dino_compat';dino.mkdir(exist_ok=True)
        V.WIN['dino_dir']=dino;V.WIN['masa_dir']=RUN/arm/'masa'
        for v in VIDEOS:
            status('Одинаковая сшивка и официальная HOTA',variant=arm+' '+v)
            with np.load(RUN/arm/'signatures'/v/'signatures.npz') as z:n=len(z['track_id'])
            compat=dino/(v+'.npz')
            if not compat.exists():np.savez_compressed(compat,first=np.zeros((n,1),np.float16),last=np.zeros((n,1),np.float16))
            d=V.video_pairs((v,str(RUN/arm/'signatures'),str(ROOT/'runs/video_rate'),'val'));scores=model.predict_proba(d['X'][:,cols])[:,1] if len(d['pairs']) else np.zeros(0)
            assert np.isfinite(scores).all()
            for kind,threshold in (('raw',2.),('masa',.6)):
                folder=RUN/arm/('evaluated_'+kind);result=stitch_video((v,str(RUN/arm/'signatures'),str(ROOT/'runs/video_rate'),'val',d['pairs'],scores,threshold,str(folder)))
                if kind=='masa':links[arm][v]=result['links']
                if arm in ARMS and kind=='raw':assert read(folder/(v+'.json'))==read(RUN/arm/'raw'/(v+'.json')),('Common annotation projection differs',arm,v)
                per[arm+'_'+kind][v]=evaluate_hota(folder,[v],split='val');hashes[(folder/(v+'.json')).relative_to(ROOT).as_posix()]=sha(folder/(v+'.json'))
        for kind in ('raw','masa'):pooled[arm+'_'+kind]=evaluate_hota(RUN/arm/('evaluated_'+kind),list(VIDEOS),split='val')
        write(STAGE/'evaluated_partial.json',dict(per_video=per,pooled=pooled,links=links))
    pv={k:{v:m['HOTA'] for v,m in values.items()} for k,values in per.items()};comparisons={}
    for kind in ('raw','masa'):
        for arm in SYSTEMS:
            if arm!='xywh':comparisons[arm+'_'+kind+'-xywh_'+kind]=paired(pv[arm+'_'+kind],pv['xywh_'+kind])
    for arm in SYSTEMS:comparisons[arm+'_masa-'+arm+'_raw']=paired(pv[arm+'_masa'],pv[arm+'_raw'])
    for arm in ('legacyD','centerD'):
        for rival in ('bytetrack','botsort','tracktrack'):comparisons[arm+'_masa-'+rival+'_masa']=paired(pv[arm+'_masa'],pv[rival+'_masa'])
    from scripts.z13_localization_labels_20261023 import verify_sources
    verify_sources(read(STAGE/'protocol.json'))
    write(RUN/'report.json',dict(videos=list(VIDEOS),pooled=pooled,per_video=per,paired=comparisons,masa_links=links,output_sha256=hashes,all_conditions_common=True,independent_final_test=False,limitations=read(RUN/'protocol.json')['limitations']))


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(STAGE/'protocol.json');verify_sources(p);assert sha(RUN/'protocol.json')==p['base_protocol_sha256'];first=time.monotonic()
    for i,video in enumerate(VIDEOS):
        for arm in SYSTEMS:status('Подготовка общих сигнатур',variant=arm+' '+video,done=i,total=7);child('signature',arm,video)
        child('masa','all',video)
        if i==0:
            measured=time.monotonic()-first;estimate=2*measured*7+600;available=min(p['budget_seconds']-(time.monotonic()-T0),(datetime.fromisoformat(p['save_at'])-now()).total_seconds())
            write(STAGE/'price.json',dict(first_complete_video_seconds=measured,conservative_seconds=estimate,available_seconds=available,proceed=estimate<available,efficacy_not_a_price_criterion=True));assert estimate<available,'Full comparison does not fit measured remaining budget'
    status('Все GPU-задания завершены; CPU-оценка')
    evaluate();write(STAGE/'status.json',dict(state='completed',stage='Все6систем,7видео,обе общие процедуры готовы',updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--job' in sys.argv:
        mode,arm,video=sys.argv[2:5];assert mode in ('signature','masa') and video in VIDEOS and arm in SYSTEMS+('all',)
        result=build_signature(arm,video) if mode=='signature' else build_masa(video)
        paths=[RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [RUN/a/'masa'/(video+'.npz') for a in SYSTEMS]
        write(STAGE/'jobs'/(mode+'_'+arm+'_'+video+'.json'),dict(result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths},at=now().isoformat()))
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(STAGE/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit pipeline mode required')
