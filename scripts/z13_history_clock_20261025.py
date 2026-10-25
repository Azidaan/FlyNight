"""Frozen-core physical-frame clock -> fixed regional readout -> matched assignment gate."""
from pathlib import Path
from datetime import datetime,timedelta
import os
PHASE=os.environ.get('Z13_HISTORY_CLOCK_PHASE','')
if PHASE!='features':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import gc
import hashlib
import time
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_readout_20261025 as decoder
from scripts import z13_region_readout_recovery1_20261025 as decoder_recovery
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_history_clock_20261025'
IDENT='z13_history_clock_v1'
ARMS=tuple(a+'_'+v for a in ('D','C','R1') for v in ('old','clock'))+('geometry',)
ALPHAS=(0.,.25)
BUDGETS={'features':1800,'replay':900}


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def status(stage,**kw):
    p=read(RUN/PHASE/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/PHASE/'stop.request').exists():raise TimeoutError('Original history-clock phase budget ended')
    write(RUN/PHASE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def comparisons(values):
    from scripts.z13_continue import paired
    videos=list(values['D_clock']);out={'Dclock-baseline':paired(values['D_clock'],{v:0. for v in videos}),'Dclock-Dold':paired(values['D_clock'],values['D_old'])}
    for a in ('C_clock','R1_clock','geometry'):out['Dclock-'+a]=paired(values['D_clock'],values[a])
    out['clock_effect_D-R1']=paired({v:values['D_clock'][v]-values['D_old'][v] for v in videos},{v:values['R1_clock'][v]-values['R1_old'][v] for v in videos})
    return out


def primary(comp):
    return comp['Dclock-baseline']['mean_per_video']>=.02 and comp['Dclock-Dold']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comp.values())


def replay_worker_source():
    code=decoder_recovery.worker_source()
    changes=[("p=read(RUN/'protocol.json');verify_sources(p)","p=read(RUN/'protocol.json');verify_sources(p);clock_maps=load_clock_maps()"),
        ("for split in ('fit','evaluate'):","for split in ('evaluate',):"),
        ("field=geometry if arm=='geometry' else maps[split,arm][l['label']]","if arm=='geometry':field=geometry\n            else:\n                name,mode=arm.rsplit('_',1)\n                field=(maps[split,name] if mode=='old' else clock_maps[name])[l['label']]"),
        ("choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}","choices={a:1 for a in ARMS} # Previous fixed alpha0.25, no retuning"),
        ("contrasts={'D-baseline':paired(chosen['D'],{v:0. for v in p['evaluate26']})}\n    for arm in ('C','R1','geometry'):contrasts['D-'+arm]=paired(chosen['D'],chosen[arm])\n    passed=contrasts['D-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in contrasts.values())",'contrasts=comparisons(chosen)\n    passed=primary(contrasts)'),
        ('exact_baseline_videos=46','exact_baseline_videos=26')]
    for old,new in changes:assert code.count(old)==1,old;code=code.replace(old,new)
    return code.replace('total=46','total=26')


def load_clock_maps():
    report=read(RUN/'features/report.json');assert report['complete'];result={a:{} for a in ('D','C','R1')}
    for name,h in report['manifest'].items():
        path=ROOT/name;assert sha(path)==h and sha(path.with_suffix('.npz'))==read(path)['sha256']
        arm=path.parent.name
        with np.load(path.with_suffix('.npz')) as z:
            for label,score in zip(z['labels'],z['scores']):assert int(label) not in result[arm];result[arm][int(label)]=score
    assert all(len(v)==8326 for v in result.values())
    return result


def feature_worker():
    # Some frozen CPU helper imports deliberately hide CUDA. Restore visibility
    # before Torch/model initialization, without changing any helper source.
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from scripts.z13_continue import paired,per_video_useful
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_target_center import observed_center_target,ridge_scores
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_history_clock import clock_features
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.set_default_device('cpu')
    folder=RUN/'features';p=read(folder/'protocol.json');verify_sources(p)
    data=Episodes(p['evaluate26']);labels={a:read(decoder.MAPS/a/'evaluate/labels.json') for a in ('D','C','R1')}
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    frames=[]
    for video in p['evaluate26']:
        with np.load(ROOT/'runs/z13_data'/(video+'.npz')) as z:frames.extend(z['fr_frame'][int(z['ep_offset'][e]):int(z['ep_offset'][e+1])].copy() for e in range(len(z['ep_offset'])-1))
    byep={}
    for l in labels['D']:byep.setdefault(l['episode'],[]).append(l)
    eps=sorted(byep);assert len(eps)==2188
    batches=[eps[i:i+8] for i in range(0,len(eps),8)];weights=[]
    for batch in batches:
        upto=5+max(l['pos'] for e in batch for l in byep[e]);ff=np.stack([frames[e][np.minimum(np.arange(upto),len(frames[e])-1)] for e in batch])
        weights.append(int(np.diff(ff,axis=1).max(0).sum()))
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    rows_byarm={};manifest={};timing=[];done=0;parity={};start=time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ('D','C','R1'):
            normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu,sd=np.asarray(normal['mean']),np.asarray(normal['sd'])
            lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            if arm=='R1':bank=bb=None;dev='cpu'
            else:bank,_,bb=old.build_arm(arm,torch,ck);dev=bb.dev;assert str(dev).startswith('cuda') and torch.cuda.is_available()
            references={l['label']:l for l in labels[arm]};canonical={r['label']:r for r in read(ROOT/f'runs/z13_observed_memory_20261021/{arm}/single_rows.json')}
            collected=[];dc=ds=0.
            for bi,batch in enumerate(batches):
                path=folder/arm/f'maps_{bi:05d}.npz';marker=path.with_suffix('.json')
                if marker.exists():
                    record=read(marker);assert sha(path)==record['sha256'];collected.extend(record['rows']);manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);done+=1;continue
                before=time.monotonic();upto=[max(l['pos'] for l in byep[e]) for e in batch];hx,mask=data.batch(torch,dev,batch,upto)
                ff=np.stack([frames[e][np.minimum(np.arange(hx.shape[1]),len(frames[e])-1)] for e in batch]);need={3}|{4+l['pos'] for e in batch for l in byep[e]}
                # The actual core/readout path with a unit clock must reproduce
                # its retained old centers/scores before accepting changed time.
                reference=old.episode_features(torch,arm,bb,hx,geom.neighbors,need) if bi<2 else None
                F=clock_features(torch,arm,bb,hx,geom.neighbors,need,ff)
                maps=[];ids=[];rows=[]
                for j,e in enumerate(batch):
                    group=byep[e];xe=(F[3][j].T-mu)/sd;xs=np.stack([(F[4+l['pos']][j].T-mu)/sd for l in group])
                    target=observed_center_target(mask[j].cpu().numpy(),geom.points,geom.spacing);scores=ridge_scores(xe,target,xs,[lam])[0]
                    if reference is not None:
                        a=(reference[3][j].T-mu)/sd;b=np.stack([(reference[4+l['pos']][j].T-mu)/sd for l in group]);old_scores=ridge_scores(a,target,b,[lam])[0]
                    for k,l in enumerate(group):
                        ref=references[l['label']];assert all(ref[key]==l[key] for key in ('video','kind','pos','episode'))
                        score=scores[k];center,pres=map_center(score,geom);point=np.asarray(to_frame(center,l['crop'],geom.eye_size))
                        row={key:l[key] for key in ('label','video','kind','pos','pred')};row.update(center=point.tolist(),p_present=pres,useful=False)
                        if l['kind']==1:
                            error=float(np.linalg.norm(point-l['gt']));prior=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']));row.update(err=error,err_pred=prior,scale=l['scale'],useful=bool(error<=.25*l['scale'] and error<prior))
                        rows.append(row);maps.append(score);ids.append(l['label'])
                        if reference is not None:
                            ce,pr=map_center(old_scores[k],geom);pt=np.asarray(to_frame(ce,l['crop'],geom.eye_size));baseline=canonical[l['label']]
                            dc=max(dc,float(np.abs(pt-baseline['center']).max()));ds=max(ds,abs(pr-baseline['p_present']))
                assert dc<.5 and ds<.001,(arm,bi,dc,ds)
                path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,scores=np.asarray(maps),labels=np.asarray(ids,np.int64))
                record=dict(sha256=sha(path),rows=rows,episodes=batch,virtual_frame_weight=weights[bi],seconds=time.monotonic()-before,old_center_px=dc,old_score_difference=ds)
                write(marker,record);manifest[marker.relative_to(ROOT).as_posix()]=sha(marker);collected.extend(rows);done+=1
                status('Frozen recurrent clock at actual frame intervals',variant=arm,done=done,total=3*len(batches),virtual_frames=weights[bi],elapsed_seconds=time.monotonic()-start)
                if arm=='D' and len(timing)<2:
                    timing.append((record['seconds'],weights[bi]))
                    if len(timing)==2:
                        rate=sum(x for x,w in timing)/sum(w for x,w in timing);estimate=2*rate*(2*sum(weights)-sum(w for x,w in timing))+240;remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
                        write(folder/'price.json',dict(estimate=estimate,remaining=remaining,proceed=estimate<remaining));assert estimate<remaining,'Preserve2groups; actual-clock extraction exceeds original budget'
                del F,reference,hx
            assert len(collected)==8326;write(folder/arm/'rows.json',collected);rows_byarm[arm]=collected;parity[arm]=dict(first_two_groups_old_center_px=dc,old_score_difference=ds)
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    values={a:per_video_useful(rows_byarm[a]) for a in rows_byarm};comp={}
    for a in ('D','C','R1'):
        old_values=per_video_useful(read(ROOT/f'runs/z13_observed_memory_20261021/{a}/single_rows.json'));comp[a+'clock-'+a+'old']=paired(values[a],old_values)
    verify_sources(p);write(folder/'report.json',dict(complete=True,manifest=manifest,parity=parity,center_paired=comp,rows_per_arm=8326,clock_rule='Actual integer frame gaps; linear stimulus interpolation; zero gap no state evolution; original nominal30Hz/substep unchanged',hota_measured=False,seconds=time.monotonic()-start))


def replay_worker():
    p=read(RUN/'replay/protocol.json');code=replay_worker_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_worker_sha256']
    ns={**vars(decoder),'RUN':RUN/'replay','ARMS':ARMS,'ALPHAS':ALPHAS,'load_clock_maps':load_clock_maps,'comparisons':comparisons,'primary':primary}
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['worker']()


def prepare():
    assert not RUN.exists();gate=read(ROOT/'runs/z13_history_clock_gate_20261025/report.json');assert gate['complete'] and gate['admission_pass']
    assert read(ROOT/'runs/z13_history_clock_gate_20261025/publication.json')['push_completed']
    assert sum(x['gaps_known'] for x in gate['per_video'].values())==26600
    base=read(decoder_recovery.RUN/'protocol.json');old=read(ROOT/'runs/z13_history_clock_gate_20261025/protocol.json');start=now();stop=min(start+timedelta(seconds=2820),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_history_clock.py','tests/test_z13_history_clock.py','scripts/z13_history_clock_gate_20261025.py','runs/z13_history_clock_gate_20261025/report.json','scripts/z13_target_model.py','tracking/z13_target_center.py','tracking/z13_ridge_localizer.py','tracking/gpu_lock.py']
    names += [f'runs/z13_target_center_dev_20261018/{a}/normalization.json' for a in ('D','C','R1')]
    names += [f'runs/z13_target_center_20261018/{a}/choice.json' for a in ('D','C','R1')]
    p=dict(base,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),phase_budgets=BUDGETS,
        videos=base['evaluate26'],fit20=[],
        action='Change only the internal temporal resampling rule: a stored interval delta_video_frames advances the existing frozen dynamics by delta nominal30Hz steps, with linear stimulus interpolation; delta0 advances neither neural nor rich-retinal EMA state. Keep all genuine samples, same crops, weights, normalization, observed-center ridge target, saved lambda, readout and alpha0.25.',
        reason='Prior clock inventory found a gap>1 in all26600known queries, often the first search frame starts2video frames after last observation. Stored entries were nevertheless each integrated as one frame. Duplicate leading padding affects mean3.954% of known eval queries. This is a testable model-time mismatch, not established performance harm or a service-only bug.',
        distinction='The gate admitted skip-padding; before any response outcomes this new protocol chooses the single complete delta-frame clock rule because the same inventory exposed systematic gaps as well. Gaps and zero-time duplicates are both handled by that rule, not separately tuned interventions. Unlike old efference models it never warps state or removes true image motion.',
        controls='C/D/R1 use the identical virtual interpolated timeline; retained original full maps are controls. The regional cost decoder remains the original mean pooling, null0, fixedalpha0.25 and alpha0exact baseline; no fit20/calibration/refitting of global parameters in this trial.7matched arms:old/clock forC/D/R1 plusunchanged geometry. Ridge weights adapt to observed template only by the same formula.',
        primary='Dclock-baseline and Dclock-Dold net-beneficial mean-video frequencies both>=2pp; all6paired95%lower>0 including Cclock,R1clock,geometry and (Dclock-Dold)-(R1clock-R1old). Conditional center shifts are secondary and cannot rescue a failed assignment gate.',
        technical='Unit-clock features exactly equal original with tested mock backbone and real rich retina; mixed zero/gap node+edge state masking tested. First2groups of each C/D/R1 old path must reproduce saved centers<0.5px/scores<0.001. Every8326map/label/crop binding retained. First2Dgroups conservative actual virtual-step rate fits1800remaining, or preserve partial and stop. Subsequent26CPUreplays all old branch outcomes exactly equal previous mean decoder atalpha0.25; rows/hashes/intervals independently verified.',
        counterargument='Linear interpolation cannot recover the actual missing images, and more realistic elapsed time can be worse for D trained on the shortened timeline. Old normalization/lambda remain fixed and may not suit changed state distributions. The first search delay is a data/runtime design choice, not proven scientific failure. No claim of HOTA, cause or benefit from gap counts alone; repeatedtrain26 exploratory.',
        next='Pass -> separately budgeted frozen-runtime clock parity/price and matched closed-loop HOTA. Fail -> preserve this actual-clock implementation and allC/R1 controls; do not retune substep/time constants/lambda on26 or start core training by default. Choose next internal input/state node only with a new measured prerequisite.',
        literature='Lappalainen et al.,Nature2024 https://www.nature.com/articles/s41586-024-07939-3 Methods/task-optimization dataset: finer integration uses temporal resampling of sampled frames. Our variable episode timeline and tracking gate are separate; published task gains are not transferred.',
        limits='OneGPU;features1800s -> CPUreplay900s, independent phase controllers, hidden execution. Sunday25Oct has no07:40restriction; nearest author deadlineMonday26Oct07:40, save07:35. New phase budgets and global2820s remain smaller and must not extend. No core/readout training, downloads,dev,HOTA orheldout75.',
        executed_worker_sha256=hashlib.sha256(replay_worker_source().encode()).hexdigest(),sha256={**old['sha256'],**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    compile(replay_worker_source(),str(Path(__file__)),'exec');write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Actual-frame neural clock: frozenGPU then fixedCPUassignment',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Временное состояние FlyVis по реальному интервалу между кадрами',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P3','fly-problem-P8'],
        question='Улучшит ли реальное прошедшее время внутри состояния мухи полезные назначения при том же считывателе?',hypothesis='Рабочая гипотеза: сжатие интервалов и добавление времени на повторённые записи могут искажать динамические признаки; это не доказанная причина ошибок.',expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новый временной узел, равные контроли, фиксированные параметры и бюджеты до откликов')],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка реального времени внутри состояния мухи','*Постановка.* '+p['action']+'\n\n*Основание.* '+p['reason']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Контроли и критерии.* '+p['controls']+' '+p['primary']+' '+p['technical']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Бюджет.* '+p['limits']+' Общий save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. Источник runs/z13_history_clock_20261025/protocol.json. '+p['literature'])


def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']));extra={}
    if phase=='replay':extra[(RUN/'features/report.json').relative_to(ROOT).as_posix()]=sha(RUN/'features/report.json')
    q=dict(p,device='cuda' if phase=='features' else 'cpu',budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);write(RUN/phase/'status.json',dict(state='ready',stage=phase,updated=now().isoformat()));return q


def finish():
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'replay/report.json');assert r['complete'] and r['exact_baseline_videos']==26
    values={a:{} for a in ARMS};oldreport=read(decoder_recovery.RUN/'report.json');assert all(v==.25 for v in oldreport['choices'].values())
    for video in p['videos']:
        path=RUN/'replay/videos'/(video+'.json');assert sha(path)==r['video_sha256'][video];record=read(path);assert record['checks']['raw_replay_exact']
        old={x['uid']:x for x in read(decoder_recovery.RUN/'videos'/(video+'.json'))['rows']};known=[]
        for row in record['rows']:
            ref=old[row['uid']];assert all(row[k]==ref[k] for k in ('uid','video','frame','tid','kind','label','split'))
            for arm in ('D','C','R1'):assert row['arms'][arm+'_old'][1]==ref['arms'][arm][1]
            assert row['arms']['geometry'][1]==ref['arms']['geometry'][1]
            if row['kind']>=0:known.append(row)
        for arm in ARMS:values[arm][video]=float(np.mean([int(x['arms'][arm][1]['beneficial'])-int(x['arms'][arm][1]['harmful']) for x in known]))
    comp=comparisons(values);assert values==r['per_video'] and comp==r['paired'] and primary(comp)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all26_old_outputs_and_mean_actions_exact=True,all_intervals_verified=True,report_sha256=sha(RUN/'replay/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT);passed=r['primary_pass']
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Реальное время состояния: критерий '+('пройден.' if passed else 'не пройден.'),
        obtained=(RUN/'replay/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(comp)+'; центры отдельно runs/z13_history_clock_20261025/features/report.json.',
        interpretation='Изменён только ход внутреннего времени при тех же весах, данных и считывателе. Это последствия первых назначений на повторно использованных train26, не HOTA; частный эффект центров не спасает провал назначения.',decision=p['next']+' Основной критерий: '+str(passed))
    for i,value in enumerate((True,passed)):card['acceptance'][i].update(met=value,evidence=(RUN/('independent_audit.json' if i==0 else 'replay/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог реального времени внутри состояния','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    from scripts import z13_region_assignment_recovery2_20261025 as original
    original.namespace()['handoff']('Полный z13_history_clock_v1 окончен. '+card['obtained']+' '+card['decision']+' Публикация runs/z13_history_clock_20261025/publication.json. Проверить частные confirmed_results/архитектурную веху семантики и временного узла. Все консоли скрыто;57чужихправок/.gitignore сохранять.')
    publish('final')


def publish(stage):
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_history_clock.py','-q']]):
        with (RUN/f'{stage}_checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/(stage+'_owned_graph.json');write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/('vault_'+stage+'.json')),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_history_clock.py','tests/test_z13_history_clock.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    paths=list(RUN.rglob('*.json')) if stage=='final' else [RUN/n for n in ('protocol.json','launch.json','pipeline.json','features/protocol.json','features/controller.json') if (RUN/n).exists()]
    names += [p.relative_to(ROOT).as_posix() for p in paths if p.name not in ('publication.json','start_publication.json')];names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()];names=sorted(set(names))
    assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists() and subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):
        for attempt in range(3):
            result=subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+25]],cwd=ROOT,capture_output=True)
            if result.returncode==0:break
            write(RUN/f'git_add_failure_{i}_{attempt}.json',dict(at=now().isoformat(),stderr=result.stderr.decode('utf-8',errors='replace'),paths=names[i:i+25]));time.sleep(.5)
        result.check_returncode()
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True);subprocess.run(['git','commit','--quiet','-m',('Register' if stage=='start' else 'Evaluate')+' physical-frame temporal resampling inside frozen FlyVis','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True);foreign_unchanged();write(RUN/('start_publication.json' if stage=='start' else 'publication.json'),dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        for phase in BUDGETS:
            folder=RUN/phase;phase_protocol(phase)
            with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
                child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,env={**os.environ,'Z13_HISTORY_CLOCK_PHASE':phase},stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    if (folder/'status.json').exists():
                        try:write(RUN/'status.json',dict(read(folder/'status.json'),phase=phase))
                        except (OSError,ValueError):pass
                    time.sleep(2)
                child.wait()
            c=read(folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
            assert read(folder/'report.json')['complete']
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:{'features':feature_worker,'replay':replay_worker}[PHASE]()
    else:raise RuntimeError('Explicit mode required')
