"""Frozen-core directional readout, with an equal-dimension shuffled control."""
from datetime import datetime,timedelta,timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_directional_context_20261018'
BASE=ROOT/'runs/z13_map_utility_20261018'
ARMS=('D','C','R1');MODES=('mean','directional','shuffled')
OWN=['scripts/z13_directional_context.py','tracking/z13_directional_context.py',
     'tests/test_z13_directional_context.py','scripts/z13_context_progress.ps1']
DEADLINE=None;CONTEXT={};START=time.monotonic()


def verify(p):
    for rel,digest in p['sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen source changed: '+rel)


def status(**values):
    if (RUN/'stop.request').exists() or (DEADLINE and time.time()>=DEADLINE):
        raise TimeoutError('Authorized window ended; completed videos retained')
    write(RUN/'status.json',dict(CONTEXT,state='running',updated=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-START,**values))


def prepare():
    if RUN.exists():raise RuntimeError('Existing run; do not duplicate')
    now=datetime.now(timezone(timedelta(hours=3)))
    if now.date().isoformat()!='2026-10-18':raise RuntimeError('Outside approved day')
    old=read(BASE/'protocol.json');files=set(old['sha256'])|set(OWN)
    files.update(['scripts/z13_map_transfer.py','tracking/z13_response_utility.py',
        'scripts/z13_score_swap.py','scripts/z13_continue.py','tracking/z13_utility.py',
        'scripts/z13_record_next.py','scripts/z13_build_vault_preserving_edits.py'])
    for arm in ARMS:
        files.update(f'runs/z13_map_utility_20261018/{arm}/{name}' for name in ('stats.npz','predicted_scores.json'))
        files.update(f'runs/z13_map_utility_20261018/{arm}/{v}.json' for v in old['videos'])
    stop=min(now+timedelta(hours=2),now.replace(hour=23,minute=50,second=0,microsecond=0))
    if now>=stop:raise RuntimeError('No new computation after cutoff')
    p=dict(id='z13_directional_context_v1',created_at=now.isoformat(),videos=old['videos'],
      sha256={x:sha(ROOT/x) for x in sorted(files)},save_stop_at=stop.isoformat(),hard_stop_at=(stop+timedelta(seconds=30)).isoformat(),
      authorization='Author approved justified runs during18.10 and said continue nonstop; this is one new readout action after failed utility transfer.',
      action='Retain original68 standardized center/neighbor-mean features; append5 zero-sum angular contrasts per34celltypes (238 total). Fixed Helmert basis divided bysqrt5. Same Gaussian target and original per-arm lambda. No core changes.',
      modes='mean68 is exact old control; directional238; shuffled238 has independent fixed seed0 direction permutation per column, same local values and dimension. D primary; C and R1 same procedures.',
      geometry='Angles assigned from actual hex point geometry, not storage order. Missing border neighbors use the center as before.',
      utility='Unchanged geometry6 and map14 logistic estimators with whole-video exclusion on reused train26; same14 map/geometry inputs and fitting settings as prior experiment. No head parameter/feature tuning.',
      reproduction='For every mean row: same key/kind/useful, center<0.5px,score<0.001; mean-map feature max difference<1e-10; mean utility OOF scores match saved scores<1e-10.',
      localization_primary='D directional minus D mean: lower95%video CI>0 and mean>=0.02. Also D directional minus D shuffled, C mean and R1 directional each lower95%CI>0.',
      selection_primary='At1%: D directional map utility minus D mean map utility, D directional geometry, D shuffled map utility and R1 directional map utility all lower95%video CIs>0. Both localization and selection gates required.2/5% and C analyses secondary, not rescue.',
      counterargument='Motion features may lack instance identity; a one-ring pattern may be insufficient. More dimensions can overfit one observed frame; shuffled control tests direction beyond added capacity.',
      decision_value='One readout implementation that changes centers and their selection; if joint gate fails, record this implementation as unconfirmed, do not tune dev or launch HOTA.',
      limitations='Exploratory reused train26, one fixed contrast scale and original lambda; no proof of unique failure cause, no blind evaluation or safety guarantee on unknowns.',
      device='cpu',core_training=False,dev=False,hota=False,heldout75=False,chat_automation=False,
      literature=['https://arxiv.org/html/1904.07220v2#S3.SS6','https://arxiv.org/html/2103.16556v2#S3.SS4'])
    RUN.mkdir();write(RUN/'protocol.json',p)
    write(RUN/'status.json',dict(state='ready',stage='directional readout registered',done=0,total=3))
    from scripts.z13_record_next import journal
    cat=read(ROOT/'experiments/project_history.json')
    cat['experiments'].append(dict(id=p['id'],title='З13: направленный пространственный контекст считывателя',area='flyvis',state='ready',started=now.date().isoformat(),
      question='Помогает ли сохранение расположения соседних сигналов найти цель и выбрать полезные поправки сверх геометрии?',
      hypothesis='Среднее соседей удаляет направления (свойство реализации); направленные контрасты могут улучшить различение цели и фона (гипотеза).',
      expected=p['localization_primary']+' '+p['selection_primary'],acceptance=[dict(criterion=p[x],met=None) for x in ('reproduction','localization_primary','selection_primary')],
      tags=['Z13','fly-problem-P8'],runs=['runs/z13_directional_context_20261018'],files=OWN,
      evidence=[dict(kind='json',ref='runs/z13_directional_context_20261018/protocol.json',claim='Условия и критерии до расчёта')]))
    (ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('постановка z13_directional_context_v1','*Основание.* Текущий tracking/z13_target_center.py использует колонки и среднее соседей; направление соседних сигналов потеряно. DiMP§3.6 использует пространственный фильтр4×4; это основание проверки, не прогноз эффекта. '
      '*Постановка.* Одна замена узла считывания: добавить5направленных контрастов на тип, прежние68компонент сохраняются, всего238. Контроль той же размерности перемешивает направления отдельно у каждой колонки, seed0. Ядра D/C/R1, данные, цель, нормировка и λ прежние. Главная D; одновременно проверяются полезные центры и выбор1% с прежней процедурой исключения видео, без подбора поdev. Оба критерия обязательны (protocol.json). До2ч CPU в пределах18.10, сохранение каждого видео, отдельная консоль. Не повторяем подбор порогов наdev; HOTA и75не запускаются. Неуспех конкретной замены не закрывает FlyVis.')
    print('Prepared',len(p['sha256']),'frozen sources; cutoff',p['hard_stop_at'])


def capture_one(torch,arm,bb,geom,bank,video,offset,stats,choices):
    from tracking import z13_target_center as target
    from tracking.z13_directional_context import directional_neighbors,shuffled_neighbors,expand_features
    from tracking.z13_response_utility import response_features
    from tracking.z13_readout import to_frame
    from scripts.z13_target_model import center_and_presence
    from scripts.z13_train import success
    from scripts import z13_map_transfer as service
    slots=directional_neighbors(geom.points,geom.neighbors);shuffled=shuffled_neighbors(slots)
    refs=[r for r in read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json') if r['video']==video]
    original=target.ridge_scores;extra={name:[] for name in MODES[1:]}
    def tapping(xe,y,xs,lambdas):
        baseline=original(xe,y,xs,lambdas)
        for name,indices in (('directional',slots),('shuffled',shuffled)):
            scores=original(expand_features(xe,indices),y,expand_features(xs,indices),lambdas)[0]
            for score in scores:
                center,peak=center_and_presence(score,geom)
                extra[name].append(dict(center_eye=center.tolist(),peak=peak,map_features=response_features(score,geom).tolist()))
        return baseline
    old_status=service.status;service.status=status;target.ridge_scores=tapping
    try:
        got=service.capture_video(torch,arm,bb,geom,bank.eye_size,video,RUN/arm/'baseline_capture'/(video+'.json'),
            stats,choices,ROOT/'runs/z13_data',refs,canonical=False,offset=offset)
    finally:
        service.status=old_status;target.ridge_scores=original
    old_features={str(x['label']):x['values'] for x in read(BASE/arm/(video+'.json'))['features']}
    feature_error=max(float(np.max(np.abs(np.asarray(x['values'])-old_features[str(x['label'])]))) for x in got['features'])
    assert feature_error<1e-10,feature_error
    by_label={str(r['label']):r for r in refs};variants={m:[] for m in MODES}
    with np.load(ROOT/f'runs/z13_data/{video}.npz') as data:
        for i,item in enumerate(got['features']):
            label=str(item['label']);base=by_label[label]
            variants['mean'].append(dict(base,map_features=item['values']))
            local=int(item['label'])-offset
            for mode in MODES[1:]:
                value=extra[mode][i];center=to_frame(value['center_eye'],data['lab_crop'][local],bank.eye_size)
                row=dict(base,center=list(center),p_present=value['peak'],map_features=value['map_features'],useful=False)
                if row['kind']==1:
                    row['err']=float(np.linalg.norm(np.asarray(center)-data['lab_center'][local]))
                    row['useful']=bool(success(row))
                variants[mode].append(row)
    report=dict(variants=variants,reproduction={k:got[k] for k in ('max_center_difference','max_peak_difference','known_outcomes_match')},
                mean_map_feature_max_error=feature_error,timing=got['timing'])
    write(RUN/arm/'videos'/(video+'.json'),report)
    return report


def worker(technical=False):
    global DEADLINE
    p=read(RUN/'protocol.json');verify(p);DEADLINE=datetime.fromisoformat(p['save_stop_at']).timestamp()
    if (RUN/'report.json').exists():raise RuntimeError('Do not repeat completed experiment')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    import torch
    torch.cuda.is_available=lambda:False;torch.set_default_device('cpu');torch.set_num_threads(2)
    from scripts import z13_target_model as old
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    choices=read(ROOT/'runs/z13_target_center_20261018/report.json')['choices']
    for ai,arm in enumerate(ARMS):
        CONTEXT.update(variant=arm,arms_done=ai,arms_total=3,videos_total=len(p['videos']))
        if (RUN/arm/'done.json').exists():continue
        status(stage='build frozen core',done=0,total=0)
        bank,geom,bb=old.build_arm(arm,torch,ck)
        assert bb is None or str(bb.dev)=='cpu'
        with np.load(BASE/arm/'stats.npz') as z:stats=z['mu'],z['sd']
        offset=0
        for vi,video in enumerate(p['videos']):
            CONTEXT.update(videos_done=vi,video=video)
            status(stage='evaluate mean/directional/shuffled together',done=0,total=0)
            path=RUN/arm/'videos'/(video+'.json')
            result=read(path) if path.exists() else capture_one(torch,arm,bb,geom,bank,video,offset,stats,choices)
            with np.load(ROOT/f'runs/z13_data/{video}.npz') as data:offset+=len(data['lab_kind'])
            if technical:
                write(RUN/'technical.json',dict(passed=True,reproduction=result['reproduction'],
                    feature_error=result['mean_map_feature_max_error'],timing=result['timing'],cuda_initialized=torch.cuda.is_initialized()))
                write(RUN/'status.json',dict(state='ready',stage='full training-video control passed; cached for full run',done=0,total=3,
                    updated=datetime.now(timezone.utc).isoformat()))
                return
        write(RUN/arm/'done.json',dict(videos=len(p['videos']),modes=list(MODES)))
        del bank,bb
    analyze(p);verify(p);assert not torch.cuda.is_initialized()
    write(RUN/'status.json',dict(state='completed',stage='directional context report saved',done=3,total=3,
          updated=datetime.now(timezone.utc).isoformat()))


def analyze(p):
    from tracking.z13_utility import observable_features,leave_video_out
    from scripts.z13_map_transfer import metadata
    from scripts.z13_continue import per_video_useful,paired,coverage_table
    from scripts.z13_score_swap import key,selected_keys
    crops=metadata(p['videos'],ROOT/'runs/z13_data');allrows={};pv={};gated={};summaries={}
    for arm in ARMS:
        references=read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')
        loaded=[read(RUN/arm/'videos'/(v+'.json')) for v in p['videos']]
        for mode in MODES:
            name=arm+'_'+mode;index={key(r):r for x in loaded for r in x['variants'][mode]}
            rows=[index[key(r)] for r in references];allrows[name]=rows;pv[name]=per_video_useful(rows)
            geometry=[];features=[]
            for row in rows:
                g,_=observable_features(row,crops[row['video'],str(row['label'])])
                geometry.append(g);features.append(np.r_[g,row['map_features']])
            targets=[int(bool(r.get('useful'))) if r['kind']>=0 else -1 for r in rows];variants={'original':rows};folds={}
            for head,x in (('geometry',geometry),('map_utility',features)):
                status(stage='fixed utility crossfit',variant=name,done=0,total=26)
                scores,folds[head]=leave_video_out(x,targets,[r['video'] for r in rows])
                if mode=='mean' and head=='map_utility':
                    saved={(r['video'],str(r['label'])):r['score'] for r in read(BASE/arm/'predicted_scores.json')}
                    error=max(abs(float(s)-saved[r['video'],str(r['label'])]) for r,s in zip(rows,scores))
                    assert error<1e-10,(arm,error)
                    write(RUN/arm/'utility_reproduction.json',dict(max_score_difference=error,passed=True))
                variants[head]=[dict(r,p_present=float(s)) for r,s in zip(rows,scores)]
            summaries[name]=dict(useful=sum(bool(r.get('useful')) for r in rows),mean_video_useful=float(np.mean(list(pv[name].values()))),
                 coverage={head:coverage_table(values) for head,values in variants.items()})
            for head,values in variants.items():
                for q in (.01,.02,.05):
                    selected=selected_keys(values,q)
                    gated[name,head,q]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in selected) for r in rows])
            write(RUN/arm/(mode+'_predictions.json'),variants)
            write(RUN/arm/(mode+'_folds.json'),folds)
    comparisons={f'D_directional-{b}':paired(pv['D_directional'],pv[b]) for b in ('D_mean','D_shuffled','C_mean','R1_directional')}
    chosen=gated['D_directional','map_utility',.01]
    selectors={label:paired(chosen,gated[name,head,.01]) for label,name,head in
       [('D_context_map-D_mean_map','D_mean','map_utility'),('D_context_map-D_context_geometry','D_directional','geometry'),
        ('D_context_map-D_shuffled_map','D_shuffled','map_utility'),('D_context_map-R1_context_map','R1_directional','map_utility')]}
    local=all(x['ci95'][0]>0 for x in comparisons.values()) and comparisons['D_directional-D_mean']['mean_per_video']>=.02
    selection=all(x['ci95'][0]>0 for x in selectors.values())
    verify(p)
    write(RUN/'report.json',dict(summary=summaries,localization=comparisons,selection_at1percent=selectors,
        localization_pass=local,selection_pass=selection,proceed=local and selection,
        decision='candidate_for_separate_transfer' if local and selection else 'directional_readout_not_confirmed; do_not_tune_dev_or_apply_tracker',
        dev_evaluated=False,hota_measured=False,limitations=p['limitations']))


def record():
    from scripts import z13_record_next as rec
    end=read(RUN/'controller_result.json');assert end['worker_ended']
    if not (RUN/'recorded.json').exists():
        cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']=='z13_directional_context_v1')
        good=end['exit_code']==0 and not end['hard_timeout'] and (RUN/'report.json').exists()
        report=read(RUN/'report.json') if good else None
        source='runs/z13_directional_context_20261018/'+('report.json' if good else 'controller_result.json')
        result={k:v for k,v in report.items() if k!='summary'} if good else end
        obtained=source+': '+json.dumps(result,ensure_ascii=False)
        interpretation=('Оба критерия конкретной пространственной замены '+('выполнены.' if report['proceed'] else
          'не выполнены. Неуспех не определяет единственную причину и не закрывает FlyVis.') if good else
          'Сохранены завершённые видео; при частичном результате вывода об эффективности нет.')
        met=[True,report['localization_pass'],report['selection_pass']] if good else [None]*3
        rec.update(card,dict(state='completed_positive' if good and report['proceed'] else 'inconclusive',
          done=datetime.now().date().isoformat(),obtained=obtained,interpretation=interpretation,
          decision=report['decision'] if good else 'save_partial_without_efficiency_verdict',
          cannot_claim='Нетdev,HOTA,75отложенных, доказательства бесполезности представления или гарантии риска неизвестных исходов. Train26 повторно использован.'),'Завершена одна пространственная замена считывателя')
        for item,value in zip(card['acceptance'],met):item.update(met=value,evidence=source)
        card['evidence'].append(dict(kind='json',ref=source,claim='Первичный полный либо частичный итог'))
        (ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        rec.journal('итог z13_directional_context_v1','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+card['decision']+'. HOTA не измерялась.')
        write(RUN/'recorded.json',dict(recorded=True))
    with (RUN/'context_tests.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','pytest','tests/test_z13_directional_context.py','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0
    rec.BASE=RUN;rec.OWN_FILES=OWN;rec.validate_and_publish(RUN,True,False)


def controller():
    import psutil
    p=read(RUN/'protocol.json');verify(p)
    if (RUN/'controller.json').exists():raise RuntimeError('Existing controller; no duplicate')
    family={os.getpid()}|{x.pid for x in psutil.Process().parents()}
    for proc in psutil.process_iter(['cmdline']):
        if proc.pid not in family and any(Path(x).name==Path(__file__).name for x in proc.info['cmdline'] or []):
            raise RuntimeError('Duplicate context process:'+str(proc.pid))
    soft=datetime.fromisoformat(p['save_stop_at']).timestamp();hard=datetime.fromisoformat(p['hard_stop_at']).timestamp()
    if time.time()>=soft:raise RuntimeError('No new computation after deadline')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker']
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born=psutil.Process(child.pid).create_time()
        write(RUN/'controller.json',dict(pid=child.pid,create_time=born,command=cmd,controller_pid=os.getpid(),
            started_at=datetime.now(timezone.utc).isoformat(),soft_deadline=p['save_stop_at'],hard_deadline=p['hard_stop_at'],chat_automation=False))
        timeout=False
        while child.poll() is None:
            if time.time()>=soft and not (RUN/'stop.request').exists():(RUN/'stop.request').write_text('Save completed videos; budget ended\n',encoding='utf-8')
            if time.time()>=hard:
                proc=psutil.Process(child.pid);assert abs(proc.create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=20);timeout=True;break
            time.sleep(2)
    write(RUN/'controller_result.json',dict(exit_code=child.returncode,worker_ended=True,hard_timeout=timeout,ended=datetime.now(timezone.utc).isoformat()))
    with (RUN/'postprocess.stdout.log').open('w',encoding='utf-8') as out,(RUN/'postprocess.stderr.log').open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'record'],cwd=ROOT,env=env,stdout=out,stderr=err)
    write(RUN/'postprocess_result.json',dict(exit_code=result.returncode,chat_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','technical','worker','controller','record']);a=parser.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='technical':worker(True)
        elif a.mode=='worker':worker()
        elif a.mode=='controller':controller()
        else:record()
    except BaseException as exc:
        if a.mode in ('technical','worker') and RUN.exists():
            write(RUN/'status.json',dict(state='paused' if isinstance(exc,TimeoutError) else 'failed',stage=str(exc),updated=datetime.now(timezone.utc).isoformat()))
        raise
