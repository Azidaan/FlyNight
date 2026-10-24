"""Bounded readout-only template intervention, with identical frozen query states."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse
import gc
import hashlib
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_template_swap_20261025'
W=ROOT/'runs/z13_weekend_20261024'
IDENT='z13_template_swap_v1'
CACHE=ROOT/'runs/z13_filter_descriptor_fixed_20261024/cache'
RESERVE=ROOT/'runs/z13_candidate_reserve_20261021'
ARMS=('D','C','R1')
T0=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as recorder
    previous=recorder.IDENT
    try:
        recorder.IDENT=IDENT
        recorder.journal(title,text)
    finally:
        recorder.IDENT=previous


def check():
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):
        raise TimeoutError('Fixed stage budget or author deadline')


def status(stage,**kw):
    check()
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def observed_metadata(videos,split):
    import numpy as np
    result=[];base=0
    for video in videos:
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:
            masks,identities,pert=z['ex_mask'],z['ep_identity'],z['ep_pert']
        for i in range(len(masks)):
            if not pert[i]:
                result.append(dict(split=split,episode=base+i,video=video,identity=str(identities[i]),mask=np.packbits(masks[i]).tobytes().hex()))
        base+=len(masks)
    return result


def prepare():
    import numpy as np
    from tracking.z13_template_swap import choose_donors
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
    assert not RUN.exists(),'Never overwrite an existing attempt'
    foreign_unchanged()
    base=read(CACHE/'protocol.json')
    verify_sources(base)
    assert read(CACHE/'report.json')['parity_pass']
    assert read(CACHE/'controller_result.json')['exit_code']==0
    videos=base['videos']
    templates=sum([observed_metadata(videos[s],s) for s in ('fit20','fit84')],[])
    recipients=observed_metadata(videos['train26'],'train26')
    mapping={str(x['episode']):choose_donors(x,templates) for x in recipients}
    own={str(x['episode']):x for x in recipients}
    write(RUN/'donors.json',dict(recipient=own,donors=mapping))
    labels=read(RESERVE/'evaluate/labels.json')
    assert len(labels)==8326 and len(recipients)==2188
    eligible=[x for x in labels if mapping[str(x['episode'])]]
    coverage=dict(episodes=len(recipients),eligible_episodes=sum(bool(v) for v in mapping.values()),
        rows=len(labels),eligible_rows=len(eligible),eligible_known_positive=sum(x['kind']==1 for x in eligible),
        videos_with_eligible_positive=sorted({x['video'] for x in eligible if x['kind']==1}),
        exclusion='Unknown observed identity, or fewer than three different known donor identities with the exact observed mask in other fit104 videos. No approximate geometry fallback.')
    assert len(coverage['videos_with_eligible_positive'])==26,'Do not silently reduce the planned paired-video cohort'
    write(RUN/'coverage.json',coverage)
    names=['tracking/z13_template_swap.py','tests/test_z13_template_swap.py',
        'scripts/z13_template_swap_20261025.py','scripts/z13_finish_template_swap_20261025.py',
        'scripts/z13_template_swap_console.ps1','scripts/z13_bounded_recovery_controller_20261022.py',
        'tracking/z13_readout.py','tracking/z13_target_center.py','tracking/z13_ridge_localizer.py',
        'scripts/z13_target_model.py','scripts/z13_train.py','tracking/gpu_lock.py',
        'runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_local_peak_20261021/geometry.json',
        'runs/z13_candidate_reserve_20261021/evaluate/labels.json',
        'runs/z13_filter_descriptor_fixed_20261024/cache/protocol.json',
        'runs/z13_filter_descriptor_fixed_20261024/cache/report.json',
        'runs/z13_filter_descriptor_fixed_20261024/cache/controller_result.json',
        'runs/z13_template_swap_20261025/donors.json','runs/z13_template_swap_20261025/coverage.json']
    names += [f'runs/z13_data/{v}.npz' for v in sum(videos.values(),[])]
    for arm in ARMS:
        names += [f'runs/z13_target_center_dev_20261018/{arm}/normalization.json',
            f'runs/z13_target_center_20261018/{arm}/choice.json',
            f'runs/z13_observed_memory_20261021/{arm}/single_rows.json',
            f'runs/z13_query_memory_20261021/{arm}/evaluate/labels.json']
        names += [f'runs/z13_filter_descriptor_fixed_20261024/cache/{arm}/{s}/descriptors.json' for s in ('fit20','fit84','train26')]
    authorization=read(W/'authorization.json')
    save=min(now()+timedelta(seconds=3600),datetime.fromisoformat(authorization['save_at']))
    hard=min(save+timedelta(seconds=20),datetime.fromisoformat(authorization['hard_stop']))
    p=dict(id=IDENT,created=now().isoformat(),timezone='Europe/Moscow',device='cuda',budget_seconds=3600,
        save_at=save.isoformat(),hard_stop=hard.isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        authorization='Author: Давай пробовать, following the proposed isolated template test; standing autonomous necessary work until26.10.2026 07:40MSK. OneGPU, no core training/dev/HOTA/heldout75/downloads.',
        action='Test whether the existing complete observed-center ridge filter depends usefully on the correct exemplar; an isolated diagnosis before choosing an instance-discrimination training change.',
        evaluate=videos['train26'],donor_fit={s:videos[s] for s in ('fit20','fit84')},arms=list(ARMS),
        intervention='Frozen query activity and recurrent history computed once per arm in original sorted groups of8. Replace all68 filter coefficients with3 distinct fit104 donor filters; exact same observed721-column mask and known different observed identity/video. Full721-column maps and peaks recomputed. No fixed peaks/scores from correct template supplied to wrong condition. Normalization and per-arm lambda unchanged; no fitting or selection on26 outcomes.',
        intercept='Per-model constant is omitted in every condition: standard-deviation-normalized peak height and local weighted center are invariant to a constant offset. A numerical unit check and the complete original own-template replay are required.',
        technical='All8326 own-template rows/arm; centers<0.5pixel and scores<0.001 vs frozen single_rows, all source hashes fixed; same original groups of8. Query arrays unchanged by projection. Foreign model IDs and exact observed mask checked for every eligible episode. All26 videos contain eligible positive rows.',
        primary='On the exact-mask eligible kind1 rows, each video contributes its own-template useful-center fraction minus the mean of the3 foreign-template fractions. Paired-video95%CI: lower>0 detects useful template dependence. Upper<0.02 establishes only an effect below2pp in this diagnostic. Otherwise inconclusive. This is not a criterion for model usefulness or training approval.',
        secondary='Map correlation and normalized-map max difference, center displacement/lattice spacing, confidence differences and candidate behavior by kind. Unknown kind-1 never classified as target absent. Original wrong-template recipients without known observed identity excluded from strict identity cohort, retained in own-template replay.',
        counterargument='Correct observations still enter the frozen recurrent state. Lack of additional filter benefit cannot establish total template neglect or an intrinsic FlyVis defect. Other-video donors are easier than in-scene distractors; geometry matching does not equal class/appearance matching. Gaussian labels and discriminative ridge fitting already condition the model on the observed region. Reused train26, no claim of independent generalization.',
        cost_gate='Retain first2D groups. Twice measured feature-extraction seconds per episode-time multiplied by remaining D/C episode-time, plus180s forR1/report, must fit the original remaining stage budget; otherwise save partial and stop. No completed chunk repeated.',
        next='If filter dependence is established, reject a blanket claim that this readout ignores its exemplar. If weak/uncertain, compare failure subsets and source parity before choosing one functional internal readout correction. No automatic core training or promotion to HOTA.',
        literature='https://arxiv.org/pdf/2001.02593 section4/figure4 and section5: target-branch perturbation distinguishes target-conditioned representation from saliency. Our fixed-state ridge test is not a reproduction of their MobileNet/auxiliary detector.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in sorted(set(names))}})
    write(RUN/'protocol.json',p)
    write(RUN/'status.json',dict(state='ready',stage='Изолированная подмена модели цели',done=0,total=822,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: изолированная подмена образца при прежнем состоянии поиска',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8','engineering-future'],
        question='Использует ли действующий различающий фильтр правильный образец сверх той информации, которая уже содержится в неизменном состоянии мухи?',
        hypothesis='Уверенный отклик может недостаточно зависеть от личности образца; это гипотеза, а не установленная причина. Подмена полного фильтра отделяет его вклад от изменения состояния.',
        expected=p['intervention']+' До3600с, одинGPU; C/D/R1, полная проверка8326строк, одинаковые доноры.',
        acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],
        summary='Протокол изолированной проверки и строгие пары зарегистрированы до вычисления исходов.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=names[:5],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Постановка, три донорских фильтра и правила вывода до результата')]))
    write(cp,cat)
    journal('постановка изолированной зависимости от образца',
        '*Действие.* '+p['action']+'\n\n*Метод.* '+p['intervention']+'\n\n*Критерий до результатов.* '+p['primary']+
        '\n\n*Контраргумент.* '+p['counterargument']+'\n\nИсходная статья: '+p['literature']+
        '\n\nБюджет3600с+20с, срок '+hard.isoformat()+', первые2группы сохраняются и задают проверку стоимости. Нового обучения, dev,HOTA,75 и загрузок нет. Покрытие строгих пар — runs/z13_template_swap_20261025/coverage.json.')


def worker():
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_default_device('cpu')
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_template_swap import project_models
    from tracking.gpu_lock import GpuLock
    p=read(RUN/'protocol.json');verify_sources(p)
    labels=read(RESERVE/'evaluate/labels.json');by_ep={}
    for l in labels:by_ep.setdefault(l['episode'],[]).append(l)
    eps=sorted(by_ep);assert len(eps)==2188
    data=Episodes(p['evaluate']);donors=read(RUN/'donors.json')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    batches=[eps[i:i+8] for i in range(0,len(eps),8)]
    time_weights=[4+max(l['pos'] for e in batch for l in by_ep[e])+1 for batch in batches]
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    done=0;measured=[]
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ARMS:
            status('Загрузка прежних коэффициентов и норм',variant=arm,done=done,total=3*len(batches))
            normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json')
            mu,sd=np.asarray(normal['mean']),np.asarray(normal['sd'])
            weights={s:read(CACHE/arm/s/'descriptors.json') for s in ('fit20','fit84','train26')}
            baseline={l['label']:l for l in read(ROOT/f'runs/z13_observed_memory_20261021/{arm}/single_rows.json')}
            if arm=='R1':
                bank=None;bb=None;dev='cpu';torch.set_default_device('cpu')
            else:
                bank,unused,bb=old.build_arm(arm,torch,ck);dev=bb.dev
                assert str(dev).startswith('cuda')
            for bi,batch in enumerate(batches):
                path=RUN/arm/f'chunk_{bi:05d}.json'
                if path.exists():
                    assert sha(path.with_suffix('.npz'))==read(path)['maps_sha256'];done+=1;continue
                status('Один поиск → свой и три чужих фильтра',variant=arm,done=done,total=3*len(batches))
                begin=time.monotonic()
                mx=[max(l['pos'] for l in by_ep[e]) for e in batch]
                hx,_=data.batch(torch,dev,batch,mx)
                need={4+l['pos'] for e in batch for l in by_ep[e]}
                F=old.episode_features(torch,arm,bb,hx,geom.neighbors,need)
                rows=[];maps=[];dc=ds=0.;feature_hash=hashlib.sha256()
                for j,e in enumerate(batch):
                    selected=donors['donors'][str(e)];eligible=bool(selected)
                    own_weight=weights['train26'][str(e)]
                    all_weights=[own_weight]+[weights[x['split']][str(x['episode'])] for x in selected]
                    recipient=donors['recipient'][str(e)]
                    assert not eligible or (len(selected)==3 and len({x['identity'] for x in selected})==3 and all(x['mask']==recipient['mask'] and x['video']!=recipient['video'] and x['identity']!=recipient['identity'] for x in selected))
                    for label in by_ep[e]:
                        x=(F[4+label['pos']][j].T-mu)/sd
                        before=x.copy();feature_hash.update(x.tobytes())
                        response=project_models(x,all_weights)
                        assert np.array_equal(before,x)
                        output=[]
                        for score in response:
                            center,pres=map_center(score,geom)
                            crop=data.labels[label['label']][7]
                            c=np.asarray(to_frame(center,crop,g['eye_size']))
                            r=dict(center=c.tolist(),p_present=pres)
                            if label['kind']==1:
                                err=float(np.linalg.norm(c-label['gt']));prior=float(np.linalg.norm(np.asarray(label['pred'])-label['gt']))
                                r.update(err=err,useful=bool(err<=.25*label['scale'] and err<prior))
                            output.append(r)
                        ref=baseline[label['label']]
                        dc=max(dc,float(np.abs(np.asarray(output[0]['center'])-ref['center']).max()))
                        ds=max(ds,abs(output[0]['p_present']-ref['p_present']))
                        centered=response-response.mean(1,keepdims=True)
                        standardized=centered/(response.std(1,keepdims=True)+1e-9)
                        stats=[]
                        for k in range(1,len(output)):
                            corr=float(np.dot(standardized[0],standardized[k])/len(x))
                            stats.append(dict(map_correlation=corr,map_z_max_abs=float(np.abs(standardized[0]-standardized[k]).max()),
                                center_displacement_lattice=float(np.linalg.norm(np.asarray(output[0]['center'])-output[k]['center']) *g['eye_size']/max(crop[2]-crop[0],1)/geom.spacing),
                                own_minus_foreign_score=output[0]['p_present']-output[k]['p_present']))
                        rows.append(dict(label=label['label'],episode=e,video=label['video'],kind=label['kind'],eligible=eligible,outputs=output,comparison=stats))
                        padded=np.zeros((4,721));padded[:len(response)]=response;maps.append(padded)
                assert dc<.5 and ds<.001,dict(arm=arm,batch=bi,center_error=dc,score_error=ds)
                output_path=path.with_suffix('.npz');temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,maps=np.asarray(maps),labels=np.asarray([r['label'] for r in rows]),eligible=np.asarray([r['eligible'] for r in rows]));temp.replace(output_path)
                seconds=time.monotonic()-begin
                write(path,dict(rows=rows,max_center_px=dc,max_score=ds,maps_sha256=sha(output_path),query_features_sha256=feature_hash.hexdigest(),seconds=seconds,query_features_unmodified=True))
                done+=1
                if arm=='D' and len(measured)<2:
                    measured.append((seconds,time_weights[bi]))
                    if len(measured)==2:
                        per_time=sum(v[0] for v in measured)/sum(v[1] for v in measured)
                        remaining_weight=2*sum(time_weights)-sum(time_weights[:2])
                        estimate=2*per_time*remaining_weight+180
                        available=min(p['budget_seconds']-(time.monotonic()-T0),(datetime.fromisoformat(p['save_at'])-now()).total_seconds())
                        write(RUN/'price.json',dict(first_two_groups=measured,conservative_remaining_seconds=estimate,available_seconds=available,proceed=estimate<available))
                        if estimate>=available:
                            write(RUN/'report.json',dict(complete=False,reason='Measured cost exceeds original budget',completed_groups=done))
                            write(RUN/'status.json',dict(state='completed',stage='Сохранён замер; полный проход не помещается в бюджет',done=done,total=3*len(batches),updated=now().isoformat()))
                            return
                del F,hx
            write(RUN/arm/'done.json',dict(at=now().isoformat(),rows=sum(len(read(RUN/arm/f'chunk_{bi:05d}.json')['rows']) for bi in range(len(batches))),groups=len(batches)))
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    report()
    verify_sources(p)
    write(RUN/'status.json',dict(state='completed',stage='Три варианта и изолированные сравнения сохранены',done=done,total=3*len(batches),updated=now().isoformat()))


def report():
    import numpy as np
    from scripts.z13_continue import paired
    from tracking.z13_template_swap import interpretation
    p=read(RUN/'protocol.json');out={};manifest={}
    for arm in ARMS:
        assert read(RUN/arm/'done.json')['rows']==8326
        rows=[];dc=ds=0.
        for path in sorted((RUN/arm).glob('chunk_*.json')):
            job=read(path);assert sha(path.with_suffix('.npz'))==job['maps_sha256']
            rows.extend(job['rows']);dc=max(dc,job['max_center_px']);ds=max(ds,job['max_score']);manifest[path.relative_to(ROOT).as_posix()]=sha(path)
        assert len(rows)==8326 and len({r['label'] for r in rows})==8326
        assert dc<.5 and ds<.001
        per_video={};counts={}
        for v in p['evaluate']:
            selected=[r for r in rows if r['video']==v and r['eligible'] and r['kind']==1]
            assert selected
            a=float(np.mean([r['outputs'][0]['useful'] for r in selected]))
            b=float(np.mean([float(np.mean([s['useful'] for s in r['outputs'][1:]])) for r in selected]))
            per_video[v]=dict(own=a,foreign_mean=b,n=len(selected))
        pair=paired({v:x['own'] for v,x in per_video.items()},{v:x['foreign_mean'] for v,x in per_video.items()})
        for kind in (-1,0,1):
            selected=[r for r in rows if r['eligible'] and r['kind']==kind]
            vals=[x for r in selected for x in r['comparison']]
            counts[str(kind)]=dict(rows=len(selected),mean_map_correlation=float(np.mean([x['map_correlation'] for x in vals])) if vals else None,
                mean_score_difference=float(np.mean([x['own_minus_foreign_score'] for x in vals])) if vals else None,
                median_center_displacement_lattice=float(np.median([x['center_displacement_lattice'] for x in vals])) if vals else None)
        out[arm]=dict(rows=len(rows),eligible=sum(r['eligible'] for r in rows),per_video=per_video,paired=pair,interpretation=interpretation(pair),by_kind=counts,max_center_px=dc,max_score=ds)
    write(RUN/'report.json',dict(complete=True,arms=out,chunk_sha256=manifest,
        limits=p['counterargument'],decision='Readout-only diagnostic; no efficacy claim or automatic training. Correct-template information remains in the shared recurrent state.'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--report',action='store_true');x=a.parse_args()
    if x.prepare:prepare()
    elif x.report:report()
    elif x.worker:
        try:worker()
        except Exception as exc:
            write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
