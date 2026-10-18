"""Bounded train26-to-dev35 transfer of the already specified CPU utility heads."""
from datetime import datetime, timedelta, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha

RUN = ROOT / 'runs/z13_map_transfer_20261018'
TRAIN = ROOT / 'runs/z13_map_utility_20261018'
DEV = ROOT / 'runs/z13_target_center_dev_20261018'
ARMS = ('C', 'D', 'R1')
HEADS = ('geometry', 'limited', 'map_utility')
OWN = ['scripts/z13_map_transfer.py', 'scripts/z13_map_transfer_progress.ps1',
       'tracking/z13_utility_transfer.py', 'tests/test_z13_utility_transfer.py']
CONTEXT = {}
START = time.monotonic()
DEADLINE = None


def msk():
    return datetime.now(timezone(timedelta(hours=3)))


def verify(protocol):
    for path, digest in protocol['sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('Frozen source changed: ' + path)


def status(**values):
    if (RUN / 'stop.request').exists() or (DEADLINE and time.time() >= DEADLINE):
        raise TimeoutError('Deadline reached; completed models and videos retained')
    write(RUN / 'status.json', dict(CONTEXT, state='running',
          updated=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-START,
          **values))


def metadata(videos, folder, canonical=False):
    crops = {}
    offset = 0
    for video in videos:
        with np.load(folder / (video + '.npz')) as data:
            for i, crop in enumerate(data['lab_crop']):
                label = f'{video}:{i}' if canonical else str(offset+i)
                crops[video, label] = crop.copy()
            offset += len(data['lab_kind'])
    return crops


def prepare():
    if RUN.exists():
        raise RuntimeError('Already prepared; do not overwrite or duplicate')
    now = msk()
    if now.date().isoformat() != '2026-10-18':
        raise RuntimeError('Authorization is bounded to18.10.2026')
    old = read(TRAIN / 'protocol.json')
    dev = read(DEV / 'protocol.json')
    assert not set(old['videos']) & set(dev['videos'])
    assert old['fit20'] == dev['fit']
    assert not set(dev['videos']) & set(old['fit20'])
    end = read(TRAIN / 'controller_result.json')
    assert end['worker_ended'] and end['exit_code'] == 0
    paths = set(old['sha256']) | set(dev['sha256']) | set(OWN)
    paths.update(['tracking/z13_utility.py', 'scripts/z13_continue.py',
                  'scripts/z13_score_swap.py', 'scripts/z13_parallel_next.py',
                  'scripts/z13_record_next.py', 'scripts/z13_build_vault_preserving_edits.py',
                  'runs/z13_map_utility_20261018/report.json',
                  'runs/z13_target_center_dev_20261018/protocol.json'])
    for arm in ARMS:
        paths.add(f'runs/z13_map_utility_20261018/{arm}/stats.npz')
        paths.add(f'runs/z13_target_center_dev_20261018/{arm}/normalization.json')
        paths.add(f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json')
        for video in old['videos']:
            paths.add(f'runs/z13_map_utility_20261018/{arm}/{video}.json')
        for video in dev['videos']:
            paths.add(f'runs/z13_target_center_dev_20261018/{arm}/{video}/center_rows.json')
    paths.update(f'runs/z13_target_center_dev_20261018/data/{v}.npz' for v in dev['videos'])
    stop = min(now + timedelta(hours=3), now.replace(hour=23, minute=50, second=0, microsecond=0))
    if now >= stop:
        raise RuntimeError('No time left in the authorized day')
    protocol = dict(id='z13_map_transfer_v1', created_at=now.isoformat(),
        authorization='Author approved transfer plan and continuous justified work on18.10.2026.',
        train_videos=old['videos'], videos=dev['videos'], fit20=old['fit20'], arms=ARMS,
        save_stop_at=stop.isoformat(), hard_stop_at=(stop+timedelta(seconds=30)).isoformat(),
        sha256={p: sha(ROOT/p) for p in sorted(paths)},
        action='Transfer the unchanged utility estimator; all heads fit once on train26 known outcomes before any dev evaluation. Fixed original C/D/R1 centers and scores.',
        fitting='StandardScaler + LogisticRegression C1 lbfgs max_iter500 seed0; geometry6, limited7, map14 features, exactly as train26. Unknowns excluded from fitting, never relabeled.',
        primary='At1%: C map minus original, geometry and limited each lower95%paired-video CI>0. Separate FlyVis gate: C map minus R1 map lower95%CI>0. Both required to continue. D and2/5% secondary, cannot rescue primary.',
        reproduction='Every dev row keeps video/label/kind/useful; unchanged center difference<0.5px and score difference<0.001; fit20 normalization identical to saved dev. Use saved centers/outcomes for comparisons.',
        technical='Capture one complete training video for C and reproduce saved map features with absolute tolerance1e-10, centers<0.5px and score<0.001 before dev.',
        deployment='Global1% rank is diagnostic and uses all scores; not an online threshold or risk guarantee. No tracker application in this protocol.',
        counterargument='C was selected after inspecting train26; sparse selection and unknown outcomes may prevent transfer or a safe intervention even when rank quality improves.',
        decision_value='Determine whether existing C readout merits one working tracker intervention, rather than another utility architecture or longer core training.',
        limitations='C selected from auxiliary results. Reused dev35 is not blind. Core D fit104; head train26; evaluation dev35 disjoint. One seed, no safety guarantee for unknowns, no HOTA, no heldout75.',
        device='cpu', gpu=False, core_training=False, hota=False, heldout75=False,
        chat_automation=False, literature=['https://arxiv.org/html/2003.12565',
          'https://proceedings.mlr.press/v97/geifman19a.html'])
    RUN.mkdir()
    write(RUN/'protocol.json', protocol)
    from scripts.z13_build_vault_preserving_edits import snapshot_local_edits
    protected = snapshot_local_edits(ROOT)
    write(RUN/'initial_foreign_graph.json', {p.relative_to(ROOT).as_posix(): h for p,h in protected.items()})
    write(RUN/'status.json', dict(state='ready', stage='registered before transfer', done=0, total=3))
    catalog = read(ROOT/'experiments/project_history.json')
    assert not any(c['id']==protocol['id'] for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=protocol['id'],
        title='З13: перенос оценки полезности C/D/R1 с train26 на dev35',
        area='flyvis', state='ready', started=now.date().isoformat(),
        question='Переносится ли преимущество оценки карты C сверх геометрии на другие видео?',
        hypothesis='Вспомогательный положительный результат C на train26 может переноситься; это гипотеза, выбор C сделан после просмотра train26.',
        expected=protocol['primary'], acceptance=[dict(criterion=c,met=None) for c in
          (protocol['reproduction'],protocol['primary'])], tags=['Z13','fly-problem-P8'],
        runs=['runs/z13_map_transfer_20261018'], files=OWN,
        evidence=[dict(kind='json',ref='runs/z13_map_transfer_20261018/protocol.json',claim='Критерии, ограничения и исходные хеши до переноса')]))
    (ROOT/'experiments/project_history.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    from scripts.z13_record_next import journal
    journal('постановка z13_map_transfer_v1',
      'Автор согласовал перенос и непрерывную работу. *Основание.* runs/z13_map_utility_20261018/report.json, arms.C.paired_selected_success.0.01: C с картой против геометрии +0,65п.п.,95%ДИ[+0,05;+1,45]; результат вспомогательный, основной критерий D не выполнен. '
      '*Постановка.* Все головы C/D/R1 однократно обучаются на известных исходах train26 до оценки dev35; признаки, центры, λ и логистическая модель прежние. Главный кандидат C, сравнения с исходным баллом, геометрией и ограниченной головой; отдельно контроль R1. Основной отбор1%,2/5% вторичные. Неизвестные исходы отдельно. До3ч CPU и30с завершения, в пределах18.10.2026; эпизоды готовы, видео и ядро не обучаются. Протокол runs/z13_map_transfer_20261018/protocol.json. Повторный dev не слепой; отбор1% ещё не рабочий порог вмешательства. HOTA и75 не запускаются. Критерии согласованного шага записаны до прогона.')
    command=f'@echo off\r\nstart "FlyVis transfer" powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "{ROOT / OWN[1]}" -Run "{RUN}"\r\n'
    (RUN/'open_progress.cmd').write_text(command,encoding='utf-8')
    print('Prepared',len(protocol['sha256']),'frozen sources')


def fit_all(protocol):
    import joblib
    from tracking.z13_utility import observable_features
    from tracking.z13_utility_transfer import fit_training_only
    crops = metadata(protocol['train_videos'], ROOT/'runs/z13_data')
    for arm in ARMS:
        folder=RUN/arm;folder.mkdir(exist_ok=True)
        if (folder/'fit.json').exists():
            saved=read(folder/'fit.json')
            assert all(sha(folder/(name+'.joblib'))==saved['models'][name] for name in HEADS)
            continue
        status(stage='fit on training videos only',variant=arm,done=0,total=3)
        rows=read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')
        maps={(v,str(item['label'])):item['values'] for v in protocol['train_videos']
              for item in read(TRAIN/arm/(v+'.json'))['features']}
        x={name:[] for name in HEADS}
        for r in rows:
            pair=(r['video'],str(r['label']))
            geometry,limited=observable_features(r,crops[pair])
            x['geometry'].append(geometry);x['limited'].append(limited)
            x['map_utility'].append(np.r_[geometry,maps[pair]])
        y=[int(bool(r.get('useful'))) if r['kind']>=0 else -1 for r in rows]
        fitted={}
        for i,name in enumerate(HEADS):
            model,info=fit_training_only(x[name],y,[r['video'] for r in rows],protocol['videos'])
            temporary=folder/(name+'.joblib.next')
            joblib.dump(model,temporary);temporary.replace(folder/(name+'.joblib'))
            fitted[name]=sha(folder/(name+'.joblib'))
            status(stage='fit on training videos only',variant=arm,done=i+1,total=3)
        write(folder/'fit.json',dict(info=info,models=fitted,rows=len(rows),
            evaluation_outcomes_used=False,fit_complete_at=datetime.now(timezone.utc).isoformat()))
    write(RUN/'all_heads_fitted.json',dict(arms=list(ARMS),dev_evaluation_started=False,
          models={arm:read(RUN/arm/'fit.json')['models'] for arm in ARMS}))


def capture_video(torch,arm,bb,geom,eye_size,video,folder,stats,choices,data_folder,reference,canonical=True,offset=0):
    from scripts import z13_target_model as old, z13_target_center_probe as capture, z13_train as train
    from tracking import z13_target_center as original
    from tracking.z13_response_utility import response_features
    from scripts.z13_score_swap import key
    saved_function=original.ridge_scores;saved_modes=capture.MODES;saved_status=capture.status;saved_data=train.DATA
    captured=[]
    def tapping(*args,**kwargs):
        result=saved_function(*args,**kwargs)
        assert result.shape[0]==1
        captured.extend(response_features(score,geom).tolist() for score in result[0])
        return result
    def progress(**kw):
        status(variant=arm,video=video,**kw)
    try:
        train.DATA=data_folder;data=train.Episodes([video])
        original.ridge_scores=tapping;capture.MODES=('center',);capture.status=progress
        rows,timing=capture.paired_rows(torch,arm,bb,geom,eye_size,data,
          {'center':[choices[arm]['center']['lambda_value']]},stats,'fixed features for transfer')
    finally:
        train.DATA=saved_data;original.ridge_scores=saved_function;capture.MODES=saved_modes;capture.status=saved_status
    rows=old.useful_rows(rows['center'],0)
    assert len(rows)==len(captured)
    refs={key(r):r for r in reference};features=[];dc=dp=0.
    for row,values in zip(rows,captured):
        label=f'{video}:{row["label"]}' if canonical else row['label']+offset
        row=dict(row,label=label,video=video);ref=refs[key(row)]
        assert bool(row.get('useful'))==bool(ref.get('useful'))
        dc=max(dc,float(np.max(np.abs(np.asarray(row['center'])-ref['center']))))
        dp=max(dp,abs(row['p_present']-ref['p_present']))
        features.append(dict(label=label,values=values))
    assert {key(dict(row,label=f'{video}:{row["label"]}' if canonical else row['label']+offset,video=video)) for row in rows}==set(refs)
    assert dc<.5 and dp<.001,(arm,video,dc,dp)
    value=dict(features=features,max_center_difference=dc,max_peak_difference=dp,
               known_outcomes_match=True,timing=timing)
    write(folder,value)
    return value


def worker(technical_only=False):
    global DEADLINE
    p=read(RUN/'protocol.json');verify(p);DEADLINE=datetime.fromisoformat(p['save_stop_at']).timestamp()
    if (RUN/'report.json').exists():raise RuntimeError('Completed run must not repeat')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    import torch
    torch.cuda.is_available=lambda:False;torch.set_default_device('cpu');torch.set_num_threads(2)
    assert torch.empty(1).device.type=='cpu' and not torch.cuda.is_initialized()
    fit_all(p)
    from scripts import z13_target_model as old
    import joblib
    from tracking.z13_utility import observable_features
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    choices=read(ROOT/'runs/z13_target_center_20261018/report.json')['choices']
    crops=metadata(p['videos'],DEV/'data',canonical=True)
    for ai,arm in enumerate(ARMS):
        CONTEXT.update(variant=arm,arms_done=ai,arms_total=3,videos_done=0,videos_total=len(p['videos']))
        if (RUN/arm/'done.json').exists():continue
        status(stage='build frozen CPU arm',done=0,total=0)
        bank,geom,bb=old.build_arm(arm,torch,ck)
        assert bb is None or str(bb.dev)=='cpu'
        with np.load(TRAIN/arm/'stats.npz') as z:stats=z['mu'],z['sd']
        saved=read(DEV/arm/'normalization.json')
        assert np.array_equal(stats[0],saved['mean']) and np.array_equal(stats[1],saved['sd'])
        if not (RUN/'technical.json').exists():
            assert arm=='C'
            video=p['train_videos'][0]
            reference=[r for r in read(ROOT/f'runs/z13_target_center_20261018/C/center_rows.json') if r['video']==video]
            got=capture_video(torch,arm,bb,geom,bank.eye_size,video,RUN/'technical_capture.json',stats,choices,
                              ROOT/'runs/z13_data',reference,canonical=False)
            before={str(x['label']):x['values'] for x in read(TRAIN/arm/(video+'.json'))['features']}
            diff=max(float(np.max(np.abs(np.asarray(x['values'])-before[str(x['label'])]))) for x in got['features'])
            assert diff<1e-10,diff
            write(RUN/'technical.json',dict(passed=True,video=video,map_feature_max_error=diff,
                  max_center_difference=got['max_center_difference'],max_peak_difference=got['max_peak_difference'],
                  timing=got['timing'],cuda_initialized=torch.cuda.is_initialized()))
        if technical_only:
            write(RUN/'status.json',dict(state='ready',stage='technical control passed; all heads fixed',done=0,total=3,
                  updated=datetime.now(timezone.utc).isoformat()))
            return
        features={};reference=[]
        for vi,video in enumerate(p['videos']):
            CONTEXT.update(videos_done=vi,video=video)
            status(stage='capture unchanged dev map',done=0,total=0)
            refs=read(DEV/arm/video/'center_rows.json');reference.extend(refs)
            path=RUN/arm/'videos'/(video+'.json')
            value=read(path) if path.exists() else capture_video(torch,arm,bb,geom,bank.eye_size,video,path,stats,choices,DEV/'data',refs)
            for x in value['features']:features[video,str(x['label'])]=x['values']
        del bb,bank
        assert len(features)==len(reference)
        x={name:[] for name in HEADS}
        for r in reference:
            pair=r['video'],str(r['label']);geometry,limited=observable_features(r,crops[pair])
            x['geometry'].append(geometry);x['limited'].append(limited)
            x['map_utility'].append(np.r_[geometry,features[pair]])
        predictions={}
        for name in HEADS:
            status(stage='predict using fixed training head',done=0,total=len(reference))
            fit=read(RUN/arm/'fit.json');assert sha(RUN/arm/(name+'.joblib'))==fit['models'][name]
            model=joblib.load(RUN/arm/(name+'.joblib'))
            predictions[name]=[dict(r,p_present=float(s)) for r,s in zip(reference,model.predict_proba(x[name])[:,1])]
        write(RUN/arm/'predictions.json',predictions)
        write(RUN/arm/'done.json',dict(videos=len(p['videos']),rows=len(reference),source_rows_unchanged=True))
    summarize(p)
    assert not torch.cuda.is_initialized();verify(p)
    write(RUN/'status.json',dict(state='completed',stage='transfer report saved',done=3,total=3,
          updated=datetime.now(timezone.utc).isoformat()))


def summarize(p):
    from scripts.z13_continue import coverage_table,per_video_useful,paired
    from scripts.z13_score_swap import selected_keys,key
    results={};pv={}
    for arm in ARMS:
        variants=read(RUN/arm/'predictions.json')
        variants['original']=[r for v in p['videos'] for r in read(DEV/arm/v/'center_rows.json')]
        coverage={name:coverage_table(rows) for name,rows in variants.items()};comp={}
        for q in (.01,.02,.05):
            for name,rows in variants.items():
                chosen=selected_keys(rows,q)
                pv[arm,name,q]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows])
            comp[f'{q:.2f}']={'map_utility-'+name:paired(pv[arm,'map_utility',q],pv[arm,name,q])
                             for name in ('original','geometry','limited')}
        results[arm]=dict(coverage=coverage,paired_selected_success=comp,
            mean_selected_success={name:float(np.mean(list(pv[arm,name,.01].values()))) for name in variants})
    cross={f'C_map_utility-{arm}_map_utility':paired(pv['C','map_utility',.01],pv[arm,'map_utility',.01]) for arm in ('R1','D')}
    primary=all(x['ci95'][0]>0 for x in results['C']['paired_selected_success']['0.01'].values())
    fly=cross['C_map_utility-R1_map_utility']['ci95'][0]>0
    verify(p)
    write(RUN/'report.json',dict(arms=results,paired=cross,primary_pass=primary,fly_control_pass=fly,
        proceed=primary and fly,decision='ranking_transfer_confirmed; freeze_application_policy_before_HOTA' if primary and fly else
        'ranking_transfer_not_confirmed; no_dev_retuning_or_tracker_application',
        dev_evaluated=True,hota_measured=False,cuda_initialized=False,limitations=p['limitations'],
        deployment_caveat=p['deployment']))


def record():
    end=read(RUN/'controller_result.json');assert end['worker_ended']
    from scripts import z13_record_next as rec
    if not (RUN/'recorded.json').exists():
        cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']=='z13_map_transfer_v1')
        report=read(RUN/'report.json') if (RUN/'report.json').exists() and end['exit_code']==0 and not end['hard_timeout'] else None
        source='runs/z13_map_transfer_20261018/'+('report.json' if report else 'controller_result.json')
        if report:
            obtained=source+': '+json.dumps({k:v for k,v in report.items() if k!='arms'},ensure_ascii=False)
            obtained+='; arms.C.paired_selected_success.0.01: '+json.dumps(report['arms']['C']['paired_selected_success']['0.01'],ensure_ascii=False)
            interpretation='Перенос ранжирования C '+('подтверждён по заданным критериям; правило применения ещё не проверено.' if report['proceed'] else
               'не подтверждён по заданным критериям; интервал с нулём означает неопределённость, а не равенство. Новый подбор поdev не выполняется.')
            state='completed_positive' if report['proceed'] else 'inconclusive';decision=report['decision'];met=[True,report['proceed']]
        else:
            obtained=source+': '+json.dumps(end);interpretation='Сохранены завершённые головы и видео; при частичном результате вывода об эффективности нет.'
            state='inconclusive';decision='save_partial_without_efficiency_verdict';met=[None,None]
        rec.update(card,dict(state=state,done=msk().date().isoformat(),obtained=obtained,interpretation=interpretation,
          decision=decision,cannot_claim='Нет HOTA, проверки75, слепогоdev или гарантии безопасности неизвестных исходов. Глобальный отбор1% не является рабочим онлайн-порогом.'),
          'Завершён разрешённый перенос оценщика')
        for item,value in zip(card['acceptance'],met):item.update(met=value,evidence=source)
        card['evidence'].append(dict(kind='json',ref=source,claim='Полный либо частичный первичный итог переноса'))
        (ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        rec.journal('итог z13_map_transfer_v1','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'. HOTA не измерялась.')
        write(RUN/'recorded.json',dict(recorded=True))
    with (RUN/'transfer_tests.log').open('w',encoding='utf-8') as out:
        test=subprocess.run([sys.executable,'-m','pytest','tests/test_z13_utility_transfer.py','-q'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=120)
    assert test.returncode==0
    rec.BASE=RUN;rec.OWN_FILES=OWN
    rec.validate_and_publish(RUN,True,False)


def controller():
    import psutil
    p=read(RUN/'protocol.json');verify(p)
    if (RUN/'controller.json').exists():raise RuntimeError('Existing controller; inspect identity, do not duplicate')
    family={os.getpid()}|{x.pid for x in psutil.Process().parents()}
    for proc in psutil.process_iter(['cmdline']):
        if proc.pid not in family and any(Path(x).name==Path(__file__).name for x in proc.info['cmdline'] or []):
            raise RuntimeError('Duplicate transfer process: '+str(proc.pid))
    soft=datetime.fromisoformat(p['save_stop_at']).timestamp();hard=datetime.fromisoformat(p['hard_stop_at']).timestamp()
    if time.time()>=soft:raise RuntimeError('No new calculation after deadline')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker']
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born=psutil.Process(child.pid).create_time()
        write(RUN/'controller.json',dict(pid=child.pid,create_time=born,command=cmd,controller_pid=os.getpid(),
          started_at=datetime.now(timezone.utc).isoformat(),soft_deadline=p['save_stop_at'],hard_deadline=p['hard_stop_at'],chat_automation=False))
        timeout=False
        while child.poll() is None:
            if time.time()>=soft and not (RUN/'stop.request').exists():
                (RUN/'stop.request').write_text('Save completed outputs; time budget ended\n',encoding='utf-8')
            if time.time()>=hard:
                proc=psutil.Process(child.pid);assert abs(proc.create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=20);timeout=True;break
            time.sleep(2)
    write(RUN/'controller_result.json',dict(exit_code=child.returncode,worker_ended=True,hard_timeout=timeout,
           ended=datetime.now(timezone.utc).isoformat()))
    with (RUN/'postprocess.stdout.log').open('w',encoding='utf-8') as out,(RUN/'postprocess.stderr.log').open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'record'],cwd=ROOT,env=env,stdout=out,stderr=err)
    write(RUN/'postprocess_result.json',dict(exit_code=result.returncode,chat_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','technical','worker','controller','record']);args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='technical':worker(True)
        elif args.mode=='worker':worker()
        elif args.mode=='controller':controller()
        else:record()
    except BaseException as exc:
        if args.mode in ('technical','worker') and RUN.exists():
            write(RUN/'status.json',dict(state='paused' if isinstance(exc,TimeoutError) else 'failed',stage=str(exc),
                updated=datetime.now(timezone.utc).isoformat()))
        raise
