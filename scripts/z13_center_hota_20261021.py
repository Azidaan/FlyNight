"""Fixed closed-loop test of the retained centered ridge readout, not rejected pilots."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_center_hota_20261021';LEGACY=ROOT/'runs/z13_shared_masa_hota_v1_20261015/full'
IDENT='z13_center_hota_v1';T0=time.monotonic();LIMIT=1200;ATTEMPT='probe'


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,39,30,tzinfo=MSK):raise TimeoutError('Owned stage deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run; inspect without duplication')
    old=read(ROOT/'runs/z13_target_center_dev_20261018/protocol.json');assert read(ROOT/'runs/z13_target_center_dev_20261018/report.json')['primary_pass']
    videos=sorted(old['videos']);names=['tracking/z13_ridge_localizer.py','tests/test_z13_ridge_localizer.py','scripts/z13_center_hota_20261021.py','scripts/z13_center_hota_controller_20261021.py','tracking/z13_compiled_backend.py','scripts/build_fly_signatures.py','scripts/z13_shared_masa.py','scripts/z13_eval.py','scripts/stitch_eval_v2.py','scripts/eval_fly_reid.py','tracking/masa_r50.py','scripts/freeze_bench.py','tracking/xywh_bytetrack.py','tracking/z13_target_center.py','tracking/z13_readout.py','tracking/foveated_eyes.py','tracking/retina.py','runs/train_cache/selection.json','runs/z13_target_center_dev_20261018/report.json','runs/z13_shared_masa_hota_v1_20261015/full/eval/report.json','weights/masa/masa_r50.pth','runs/flyvis/results/flow/0000/000/best_chkpt','runs/flyvis/results/flow/0000/000/_meta.yaml','runs/z13_spatial_v2_20261013/D/full/model.pt']
    for arm in ('C','D','R1'):
        rows_path=ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json';rows=read(rows_path);assert len(rows)==8326
        from tracking.z13_spatial_localizer import percentile_threshold
        selection=percentile_threshold([r['p_present'] for r in rows],.01)
        normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
        config=dict(format='z13-center-ridge-hota-v1',arm=arm,tau=selection['tau'],lambda_value=lam,normalization=normal,backbone_file='runs/z13_spatial_v2_20261013/D/full/model.pt',policy=selection)
        write(RUN/arm/'config.json',config);write(RUN/arm/'noop_config.json',dict(config,tau=None))
        names.extend([f'runs/z13_target_center_20261018/{arm}/center_rows.json',f'runs/z13_target_center_20261018/{arm}/choice.json',f'runs/z13_target_center_dev_20261018/{arm}/normalization.json',f'runs/z13_center_hota_20261021/{arm}/config.json',f'runs/z13_center_hota_20261021/{arm}/noop_config.json'])
    # Freeze only train130/dev35 inputs. Heldout75 and detector validation are outside this job.
    from scripts.fly_lifecycle_alive import sets
    groups=sets(ROOT/'runs/fly_signatures',ROOT/'runs/fly_signatures_val')
    for v,*_ in groups['train']:names.append(f'runs/stitch_v2/gap150_dist5/cache/{v}.npz')
    for v in videos:
        names.extend([f'data/bdd100k/videos/train/{v}.mov',f'runs/train_cache/{v}/detections/raw_video.npz',f'runs/train_cache/{v}/motion/flyvis_tiled.json',f'runs/train_cache/{v}/alignment.json',f'data/bdd100k/labels/box_track_20/train/{v}.json',f'runs/fly_signatures/{v}/signatures.npz',f'runs/dino_embeddings/{v}.npz',f'runs/masa_embeddings/{v}.npz',f'runs/chain_filter_rank/repro/dev/{v}.json',f'runs/z13_shared_masa_hota_v1_20261015/full/eval/C_masa/{v}.json',f'runs/z13_shared_masa_hota_v1_20261015/full/eval/D_masa/{v}.json'])
        names.append(f'runs/stitch_v2/gap150_dist5/cache/{v}.npz')
    names.extend(['data/bdd100k/labels/bdd100k_labels_images_train.json','scripts/z13_hota_deadline_guard.py','scripts/z13_night_control_20261021.py','runs/z13_triton_hota_v1_20261014/packages/triton_windows-3.6.0.post26.dist-info/RECORD'])
    probe_video=read(ROOT/'runs/z13_candidate_reserve_20261021/protocol.json')['evaluate'][0]
    names.extend([f'data/bdd100k/videos/train/{probe_video}.mov',f'runs/train_cache/{probe_video}/detections/raw_video.npz',f'runs/train_cache/{probe_video}/motion/flyvis_tiled.json',f'runs/fly_signatures/{probe_video}/signatures.npz'])
    p=dict(id=IDENT,created=now().isoformat(),authorization='Explicit author autonomous necessary runs,technical repairs,commits/push until07:40 MSK21.10.2026; partial completed videos retained.',
        action='Apply the retained Gaussian-center ridge readout in the unchanged tracker environment. Rejected candidate rankers, temporal mixtures,quadratic peaks and spatial windows are not included.',
        arms=['C','D','R1'],videos=videos,probe_video=probe_video,threshold_videos=read(ROOT/'runs/z13_candidate_reserve_20261021/protocol.json')['evaluate'],
        policy='Each arm one fixed top1% threshold from ALL8326 natural train26 raw centered-map peak confidences, ties retained; no labels/errors used in threshold selection. Lost tracks only in existing30framebuffer: change mean[:2] only. Size,velocity,covariance,associations and MASA settings unchanged. No new refusal calibration or 90%precision claim.',
        implementation='Same four observed frames, fp16 input quantization, ongoing state/EMA,68 normalized features, observed Gaussian target and saved center lambda. Cache the algebraically identical ridge coefficients once per lost episode. Core frozen. Compiler only after same-input strict center/score/decision parity.',
        technical='Pure regression and incrementalR1 bit-exact tests; saved first8 natural train26 episode controls C/D/R1: center<0.5px,score<0.001,same threshold decisions, same lambda. Compiler/eager same tolerances and decisions. One complete train video tau=None output/frame IDs exactly equal stand. Same old MASA head recipe reproduced and frozen; all35 baseline stitched JSONs must equal prior stand before comparison.',
        comparisons=['Dcenter-stand: pooledHOTA>=+0.10 and pairedvideo95%lower>0','Dcenter-R1center: pairedvideo95%lower>0','Dcenter-legacyD: pairedvideo95%lower>0'],
        secondary='Ccenter versus stand,R1center,legacyC; Dcenter-Ccenter is separate core-training contrast. All/night strata and unfavorable outcomes reported. No primary rescue by C or subgroup.',
        evaluation='Original MASA R50,shared immutable frame features with first-frame parity guard; original greedy stitcher gap150/dist5/tau0.6 and base_masa columns. TrackEval full35 only, pairedvideo5000 bootstrap seed1; pooled separately. LegacyD output reusable only with unchanged source/input hashes and baseline parity. New implementation comparison is not isolation of Gaussian target alone.',
        counterargument='Better episode centers may not improve closed-loop trajectories. Top1% confidence is weak and runtime lost-frame distribution differs from labeled-time train26. Corrections alter future crops and associations; frozen episode results do not prove tracker benefit.',
        decision='One fixed diagnostic policy, no dev/HOTA retuning. After technical gates run sequential per-video C/D/R1 signatures,sharedMASA,stitch. Complete-only markers, stop starting jobs07:35, drain only current job until07:39:30. Partial results: remaining jobs only, no efficacy verdict or partial HOTA.',
        probe_budget_seconds=1200,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',prohibited='No core training,downloads,detector change,protected75 or chat heartbeat automations.',sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',phase='probe',stage='Техническая проверка и замер стоимости ещё не выполнены',updated=now().isoformat()))
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: HOTA сохранённого центрированного считывателя',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8'],question='Даёт ли ранее подтверждённый центрированный считыватель пользу полному стенду сверх отсутствия поправки, R1 и прежней D?',hypothesis='Функциональный выигрыш центров может переноситься в замкнутый трекер; это ещё не измерено.',expected='Одна фиксированная диагностическая политика1% без подбора по dev/HOTA. C/D/R1, прежние35dev,YOLO26/ByteTrack/MASA без изменений. Техническая проба до1200с; далее только оставшееся окно автора, законченные видео сохраняются. Неполная очередь не даёт вердикта об эффективности.',
        acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['comparisons'][0],met=None),dict(criterion=p['comparisons'][1],met=None),dict(criterion=p['comparisons'][2],met=None)],summary='Зарегистрировано подключение принятого центрированного считывателя к стенду; HOTA не измерена.',done='',obtained='',interpretation='',decision='',cannot_claim='Эта политика диагностическая, прежний провал90%/50 не переписывается. Dev35 многократно использован, не независимое подтверждение. Сравнение полной реализации не изолирует только форму регрессионной цели; D−C отдельно, вклад анатомии не доказан. Неполные видео/очередь не измерение HOTA.',metrics=[],commits=[],files=['tracking/z13_ridge_localizer.py','scripts/z13_center_hota_20261021.py','scripts/z13_center_hota_controller_20261021.py'],evidence=[dict(kind='file',ref='runs/z13_center_hota_20261021/protocol.json',claim='Политика/критерии/срок до HOTA'),dict(kind='file',ref='runs/z13_target_center_dev_20261018/report.json',claim='Сохранённая ранее положительная функциональная версия')]))
    write(catpath,cat);journal('постановка проверки стенда с принятой центрированной моделью','*Постановка.* '+p['action']+'\n\n'+p['policy']+'\n\nОсновные критерии до HOTA: '+str(p['comparisons'])+'.\n\n'+p['technical']+'\n\n'+p['decision']+'\n\nЭто самостоятельная проверка ранее прошедшего функциональные критерии узла, а не перенос проваленных моделей карты104/парного ранжирования/окна. Нет новых весов ядра или открытия75. Чистая проверка коэффициентов совпала побитово; у R1 до регистрации исправлен порядок операций division→multiply reciprocal, чтобы воспроизвести исходную EMA точно; тесты2passed. Проба до1200с, остальное только до утра автора; оценка стоимости ещё не получена. Автоматизация чата остаётся PAUSED.')


def build(video,arm,out,noop=False):
    from scripts import build_fly_signatures as builder
    from tracking import z13_readout
    from tracking.z13_ridge_localizer import CenterRidgeLocalizer
    from tracking.z13_compiled_backend import enable
    class CompiledCenter(CenterRidgeLocalizer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.arm in ('C','D'):self.compiler_info=enable(self)
    z13_readout.Localizer=CompiledCenter
    saved=sys.argv
    try:
        sys.argv=['builder','--video-id',video,'--z13',str(RUN/arm/('noop_config.json' if noop else 'config.json')),'--output',str(out),'--max-eyes','4']
        args=builder.parse_args();args.skip_fly=True;return builder.build_video(video,args)
    finally:sys.argv=saved


def replay(loc,hx,masks,labels,data,episodes):
    import numpy as np
    from tracking.z13_readout import to_frame
    eyes=loc.start_sequence(hx[:,:4],masks);answer={};by={}
    for i,l in enumerate(data.labels):
        if not l[8] and l[0] in episodes:by.setdefault((l[0],l[1]+4),[]).append((i,l))
    for t in range(4,hx.shape[1]):
        pairs=loc.advance(eyes,hx[:,t])
        for j,e in enumerate(episodes):
            for i,l in by.get((e,t),[]):answer[i]=dict(center=np.asarray(to_frame(pairs[j][0],l[7],loc.bank.eye_size)),score=pairs[j][1])
    return answer


def environment_head():
    """Reproduce and freeze the existing MASA stitch model, no scientific retuning."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import joblib
    from scripts import stitch_eval_v2 as V
    from scripts.fly_lifecycle_alive import sets
    from scripts.eval_fly_reid import stitch_video
    p=read(RUN/'protocol.json');groups=sets(ROOT/'runs/fly_signatures',ROOT/'runs/fly_signatures_val');V._set(150,5.,True)
    data=[]
    for v,sr,rt,sp in groups['train']:check();data.append(V.cached_pairs((v,str(sr),str(rt),sp)))
    cols=[V.NAMES.index(n) for n in V.HEADS['base_masa']];assert 'dino_cos' not in V.HEADS['base_masa'] and not any(n.startswith('fly_') for n in V.HEADS['base_masa'])
    X=np.concatenate([d['X'][d['y']>=0] for d in data]);y=np.concatenate([d['y'][d['y']>=0] for d in data]);model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)).fit(X[:,cols],y)
    joblib.dump(dict(model=model,cols=cols),RUN/'masa_head.joblib');parity={}
    for v,sr,rt,sp in groups['dev']:
        check();d=V.cached_pairs((v,str(sr),str(rt),sp));prob=model.predict_proba(d['X'][:,cols])[:,1] if len(d['y']) else np.zeros(0)
        stitch_video((v,str(sr),str(rt),sp,d['pairs'],prob,.6,str(RUN/'probe/stand_stitch')))
        actual=read(RUN/'probe/stand_stitch'/(v+'.json'));reference=read(ROOT/'runs/chain_filter_rank/repro/dev'/(v+'.json'));assert actual==reference,v;parity[v]=True
    write(RUN/'probe/stand_parity.json',dict(all35_stitched_json_equal=True,videos=parity,masa_head_sha256=sha(RUN/'masa_head.joblib')))


def probe():
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tracking.gpu_lock import GpuLock
    from tracking.z13_compiled_backend import enable
    from tracking.z13_ridge_localizer import CenterRidgeLocalizer
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.build_fly_signatures import RenderOnlyBank
    from scripts.z13_train import Episodes
    p=read(RUN/'protocol.json');verify(p);data=Episodes(p['threshold_videos']);eps=sorted({l[0] for l in data.labels if not l[8]})[:8];maximum=[max(l[1] for l in data.labels if l[0]==e and not l[8]) for e in eps];repro={};timing={}
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for arm in p['arms']:
            status('Эпизод: исходное/скомпилированное считывание',variant=arm,done=len(repro),total=3)
            est=make_motion_estimator('flyvis_tiled',30.,.02);loc=CenterRidgeLocalizer(RUN/arm/'config.json',RenderOnlyBank(est,max_batch=4));hx,ms=data.batch(torch,loc.dev,eps,maximum);mask=ms.cpu().numpy()
            eager=replay(loc,hx,mask,None,data,eps)
            if arm in ('C','D'):enable(loc)
            compiled=replay(loc,hx,mask,None,data,eps);saved={r['label']:r for r in read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')}
            dc=max(float(np.abs(r['center']-saved[i]['center']).max()) for i,r in compiled.items());ds=max(abs(r['score']-saved[i]['p_present']) for i,r in compiled.items())
            ce=max(float(np.abs(r['center']-eager[i]['center']).max()) for i,r in compiled.items());se=max(abs(r['score']-eager[i]['score']) for i,r in compiled.items());assert dc<.5 and ds<.001 and ce<.5 and se<.001
            assert all((r['score']>=loc.tau)==(saved[i]['p_present']>=loc.tau)==(eager[i]['score']>=loc.tau) for i,r in compiled.items())
            repro[arm]=dict(rows=len(compiled),max_center_px=dc,max_score=ds,compiler_center_px=ce,compiler_score=se,same_threshold_decisions=True);del loc,est,hx,ms;gc.collect();torch.cuda.empty_cache()
        video=p['probe_video'];status('Полное train-видео без вмешательств',variant='R1 noop',done=0,total=4)
        began=time.monotonic();build(video,'R1',RUN/'probe/noop',True)
        with np.load(RUN/'probe/noop'/video/'signatures.npz') as a,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as b:assert np.array_equal(a['outputs'],b['outputs']) and int(a['frame_count'])==int(b['frame_count'])
        timing['noop_seconds']=time.monotonic()-began
        for ai,arm in enumerate(p['arms']):
            status('Полное train-видео для замера стоимости',variant=arm,done=ai,total=3);began=time.monotonic();build(video,arm,RUN/'probe'/arm);timing[arm]=dict(seconds=time.monotonic()-began,events=read(RUN/'probe'/arm/video/'signatures.json')['z13_events'])
        from tracking.masa_r50 import build as masa_model
        from scripts.z13_shared_masa import extract_video
        status('Проверка общей MASA',done=0,total=1);model=masa_model();paths={arm:RUN/'probe'/arm/video/'signatures.npz' for arm in p['arms']};outputs={arm:RUN/'probe/masa'/arm/(video+'.npz') for arm in p['arms']}
        timing['masa']=extract_video(model,ROOT/'data/bdd100k/videos/train'/(video+'.mov'),paths,outputs,check_first_frame=True);del model;gc.collect();torch.cuda.empty_cache()
    status('Воспроизведение прежней сшивки на35видео',done=0,total=35);environment_head();verify(p)
    estimate=35*(sum(timing[a]['seconds'] for a in p['arms'])+timing['masa']['seconds'])
    write(RUN/'technical.json',dict(passed=True,reproduction=repro,no_intervention_outputs_bitwise_equal=True,baseline_stitch35_equal=True,timing=timing,rough_full_seconds=estimate,caveat='One complete train video; dev complexity and compilation/cache differ. Partial queue expected if window insufficient; no HOTA measured.'))


def signature(arm,video):
    import numpy as np
    target=RUN/arm/'signatures'/video
    if (target/'signatures.npz').exists() or (target/'signatures.json').exists():
        try:
            meta=read(target/'signatures.json')
            with np.load(target/'signatures.npz') as z,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as ref:assert int(z['frame_count'])==meta['frames']==int(ref['frame_count']) and np.isfinite(z['outputs']).all()
            return
        except (OSError,ValueError,KeyError,AssertionError,EOFError):preserve_partial([target/'signatures.npz',target/'signatures.json'])
    build(video,arm,RUN/arm/'signatures')
    with np.load(target/'signatures.npz') as z,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as reference:assert int(z['frame_count'])==int(reference['frame_count']) and np.isfinite(z['outputs']).all()


def preserve_partial(paths):
    for path in paths:
        path=Path(path).resolve();path.relative_to(RUN.resolve())
        if path.exists():
            dest=path.with_name(path.name+'.partial_'+now().strftime('%Y%m%d_%H%M%S_%f'));dest.resolve().relative_to(RUN.resolve());assert not dest.exists();path.rename(dest)


def masa(video):
    import torch
    torch.set_num_threads(2)
    from tracking.masa_r50 import build as make
    from scripts.z13_shared_masa import extract_video,tasks
    import numpy as np
    arms=read(RUN/'protocol.json')['arms'];model=make();paths={a:RUN/a/'signatures'/video/'signatures.npz' for a in arms};out={a:RUN/a/'masa'/(video+'.npz') for a in arms}
    for a in arms:
        preserve_partial([out[a].with_name(out[a].name+'.next.npz')])
        if out[a].exists():
            try:
                n,_=tasks(paths[a])
                with np.load(out[a]) as z:assert z['first'].shape==z['last'].shape==(n,256) and np.isfinite(z['first']).all() and np.isfinite(z['last']).all()
            except (OSError,ValueError,KeyError,AssertionError,EOFError):preserve_partial([out[a]])
    pending={a:path for a,path in paths.items() if not out[a].exists()}
    if pending:
        detail=extract_video(model,ROOT/'data/bdd100k/videos/train'/(video+'.mov'),pending,{a:out[a] for a in pending},progress=lambda i,n:status('MASA текущего видео',video=video,done=i,total=n),check_first_frame=True);write(RUN/'masa_details'/(video+'.json'),detail)
    for a in arms:
        n,_=tasks(paths[a])
        with np.load(out[a]) as z:assert z['first'].shape==z['last'].shape==(n,256) and np.isfinite(z['first']).all() and np.isfinite(z['last']).all()
        compat=RUN/a/'dino_compat'/(video+'.npz');compat.parent.mkdir(parents=True,exist_ok=True)
        if not compat.exists():np.savez_compressed(compat,first=np.zeros((n,384),np.float16),last=np.zeros((n,384),np.float16))


def stitch(video):
    import joblib
    import numpy as np
    from scripts import stitch_eval_v2 as V
    from scripts.eval_fly_reid import stitch_video
    pm=joblib.load(RUN/'masa_head.joblib');p=read(RUN/'protocol.json')
    for arm in p['arms']:
        V._set_dirs(150,5.,str(RUN/arm/'dino_compat'),str(RUN/arm/'masa'));sig=RUN/arm/'signatures';rt=ROOT/'runs/train_cache';d=V.video_pairs((video,str(sig),str(rt),'train'));prob=pm['model'].predict_proba(d['X'][:,pm['cols']])[:,1] if len(d['y']) else np.zeros(0)
        stitch_video((video,str(sig),str(rt),'train',d['pairs'],prob,.6,str(RUN/'eval'/(arm+'_masa'))))


def evaluate():
    import numpy as np
    from evaluation.hota_eval import evaluate_hota
    from scripts.buffer_extend_eval import KEYS,per_video_hota
    from scripts.z13_eval import paired
    from datasets.subset_filter import filter_videos_by_subset,load_video_attributes
    p=read(RUN/'protocol.json');videos=p['videos'];dirs={a+'_masa':RUN/'eval'/(a+'_masa') for a in p['arms']}
    dirs.update(A_masa=ROOT/'runs/chain_filter_rank/repro/dev',legacyD=LEGACY/'eval/D_masa',legacyC=LEGACY/'eval/C_masa')
    assert all(all((d/(v+'.json')).exists() for v in videos) for d in dirs.values())
    per={name:{v:per_video_hota((str(d),v,'train'))[1] for v in videos} for name,d in dirs.items()};night=list(filter_videos_by_subset(load_video_attributes(str(ROOT/'data/bdd100k/labels/bdd100k_labels_images_train.json')),videos,'night'));rng=np.random.default_rng(1);report=dict(videos=35,pooled={},diff={},per_video=per,diagnostic_policy=True,independent_confirmation=False,old90_gate_unchanged=True)
    pairs=[('D_masa','A_masa'),('D_masa','R1_masa'),('D_masa','legacyD'),('D_masa','C_masa'),('C_masa','A_masa'),('C_masa','R1_masa'),('C_masa','legacyC')]
    for subset,vv in [('all',videos),('night',night)]:
        for name,d in dirs.items():report['pooled'][name+'|'+subset]={k:v for k,v in evaluate_hota(d,vv,split='train').items() if k in KEYS}
        for a,b in pairs:
            r=paired(per,a,b,vv,rng);r['pooled_gain']=report['pooled'][a+'|'+subset]['HOTA']-report['pooled'][b+'|'+subset]['HOTA'];report['diff'][a+'-'+b+'|'+subset]=r
    diff=report['diff'];report['criteria']=dict(implementation_D_vs_stand=diff['D_masa-A_masa|all']['pooled_gain']>=.1 and diff['D_masa-A_masa|all']['ci95'][0]>0,fly_added_D_vs_R1=diff['D_masa-R1_masa|all']['ci95'][0]>0,readout_D_vs_legacyD=diff['D_masa-legacyD|all']['ci95'][0]>0)
    verify(p);write(RUN/'eval/report.json',report)


def main():
    global LIMIT,ATTEMPT
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--attempt',default='probe');ap.add_argument('--budget',type=int,default=1200);ap.add_argument('--stage',choices=['probe','signatures','masa','stitch','eval']);ap.add_argument('--arm');ap.add_argument('--video');args=ap.parse_args()
    if args.prepare:prepare();return
    ATTEMPT=args.attempt;LIMIT=args.budget;stage=args.stage or 'probe'
    if not args.worker:raise ValueError('Worker required')
    os.environ['FOVEA_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    os.environ['CUDA_VISIBLE_DEVICES']='' if stage in ('stitch','eval') else '0'
    from tracking.z13_compiled_backend import setup
    setup();p=read(RUN/'protocol.json');verify(p)
    from tracking.gpu_lock import GpuLock
    try:
        if stage=='probe':probe()
        elif stage in ('signatures','masa'):
            with GpuLock('flyvis_fovea_gpu'):
                signature(args.arm,args.video) if stage=='signatures' else masa(args.video)
        elif stage=='stitch':stitch(args.video)
        else:evaluate()
        write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Этап '+stage+' сохранён',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
