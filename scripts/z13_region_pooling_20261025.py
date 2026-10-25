"""One CPU paired intervention: mean versus max on saved full response fields."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import inspect
import hashlib
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_readout_20261025 as base
from scripts import z13_region_readout_recovery1_20261025 as recovery
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_region_pooling_20261025'
IDENT='z13_region_pooling_v1'
ARMS=tuple(a+'_'+m for a in ('D','C','R1','geometry') for m in ('mean','max'))
ALPHAS=(0.,.25)


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def comparisons(values):
    from scripts.z13_continue import paired
    videos=list(values['D_max'])
    out={'Dmax-baseline':paired(values['D_max'],{v:0. for v in videos}),'Dmax-Dmean':paired(values['D_max'],values['D_mean'])}
    for arm in ('C','R1','geometry'):out['Dmax-'+arm+'max']=paired(values['D_max'],values[arm+'_max'])
    return out


def primary(comp):
    return comp['Dmax-baseline']['mean_per_video']>=.02 and comp['Dmax-Dmean']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())


def worker_source():
    code=recovery.worker_source()
    changes=[('from tracking.z13_region_readout import region_probabilities,cost_row','from tracking.z13_region_pooling import region_probabilities\n    from tracking.z13_region_readout import cost_row'),
        ("field=geometry if arm=='geometry' else maps[split,arm][l['label']]","original_arm,mode=arm.rsplit('_',1)\n            field=geometry if original_arm=='geometry' else maps[split,original_arm][l['label']]"),
        ('region_probabilities(field,geom.points,projected,supported)','region_probabilities(field,geom.points,projected,supported,reduction=mode)'),
        ("choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}","choices={a:1 for a in ARMS} # Preserved old alpha0.25; no new selection"),
        ("contrasts={'D-baseline':paired(chosen['D'],{v:0. for v in p['evaluate26']})}\n    for arm in ('C','R1','geometry'):contrasts['D-'+arm]=paired(chosen['D'],chosen[arm])\n    passed=contrasts['D-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in contrasts.values())",
         'contrasts=comparisons(chosen)\n    passed=primary(contrasts)')]
    for old,new in changes:assert code.count(old)==1,old;code=code.replace(old,new)
    return code


def worker():
    p=read(RUN/'protocol.json');code=worker_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_worker_sha256']
    ns={**vars(base),'RUN':RUN,'ARMS':ARMS,'ALPHAS':ALPHAS,'comparisons':comparisons,'primary':primary}
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['worker']()


def prepare():
    assert not RUN.exists()
    previous=ROOT/'runs/z13_region_product_20261025'
    assert read(previous/'publication.json')['push_completed'] and not read(previous/'training/report.json')['primary_pass']
    c=read(previous/'recovery1/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
    old=read(recovery.RUN/'report.json');assert old['complete'] and all(v==.25 for v in old['choices'].values())
    p=read(recovery.RUN/'protocol.json');verify_sources(p)
    start=now();stop=min(start+timedelta(seconds=1200),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_pooling.py','tests/test_z13_region_pooling.py',
        'scripts/z13_region_readout_20261025.py','scripts/z13_region_readout_recovery1_20261025.py',
        'runs/z13_region_readout_20261025/recovery1/report.json','runs/z13_region_product_20261025/training/report.json']
    protocol=dict(p,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=1200,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Replace spatial averaging of the same standardized target-response field inside each unchanged detection region by its maximum. All8conditions D/C/R1/geometry timesmean/max replay simultaneously on the same46original videos and queries.',
        held_fixed='Identical full fields, retinal support, null logit0, detector confidence, original full assignment, alpha0.25 retained from the previous mean prototype for every arm. Alpha0 gives exact untouched baseline; no threshold or alpha retuning. Fit20 recorded for parity/price only, evaluate26 for preregistered comparisons.',
        primary='Dmax-baseline and Dmax-Dmean means both>=2pp with positive lower95%CI; Dmax-Cmax,R1max,geometrymax all lower95%CI>0. All conditions; no selection of a favorable arm or subset.',
        technical='All46baseline outputs float32-exact; all prior peak centers<0.5px/scores<0.001. All mean outcomes must exactly reproduce the previous mean prototype atalpha0.25 on every saved query. Source/rows hashes and paired-video intervals independently checked. First2fit replays conservative2*mean_seconds*44+120 fit remaining1200s or preserve partial and stop.',
        distinction='This changes the spatial aggregation node, not the loss, number of channels, embeddings, thresholds or target/history. Direct mean was tried but max on the same true regions was not. It is not max over one globally moved center.',
        literature='Girshick,ICCV2015,FastR-CNN,section2.1 https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf : region-wise max pooling preserves local activations. Our single scalar target-response maximum differs from the learned multichannel7x7 RoI layer; this motivates a test, not predicted tracking gain.',
        counterargument='Max can amplify a distractor/noise peak and favor large/overlapping boxes. A positive comparison of this scalar readout does not prove that all mean-pooled learned representations failed because of averaging. C/R1/geometry controls and exact mean replay are mandatory; repeatedtrain26 and one-step labels are exploratory, not HOTA.',
        next='Pass -> separately budgeted frozen-policy closed-loop parity/price/HOTA with matched controls. Fail -> retain this max-pooling implementation and stop this fixed pooling branch. No extra scalar threshold/loss trials. Select another internal spatial/state node only from measured evidence.',
        limits='No new training or GPU/extraction/dev/HOTA/heldout75/download. Known/unknown labels unchanged. Prior smaller budgets are complete; this is a new1200sCPU intervention, not an extension.',
        executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',protocol);write(RUN/'status.json',dict(state='ready',stage='Paired mean/max region readout; CPU saved fields',done=0,total=46,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Пространственное считывание: среднее против максимума внутри той же рамки',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Поможет ли сохранение локального максимума внутри реальной области при неизменном назначении и коэффициенте применения?',
        hypothesis='Рабочая гипотеза: усреднение может скрывать локальный отклик цели; максимум может его сохранить, но также усилить помеху. Причина не установлена.',
        expected=protocol['action'],acceptance=[dict(criterion=protocol['technical'],met=None),dict(criterion=protocol['primary'],met=None)],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Парные8условий, прежнийalpha, критерии и бюджет до расчёта')],obtained='',interpretation='',decision='',cannot_claim=protocol['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка парной проверки пространственного усреднения',
        '*Постановка.* '+protocol['action']+' '+protocol['held_fixed']+'\n\n*Критерии.* '+protocol['primary']+' '+protocol['technical']+'\n\n*Отличие.* '+protocol['distinction']+'\n\n*Возражение.* '+protocol['counterargument']+'\n\n*Бюджет.* CPU1200с+20с, скрытый запуск, save '+protocol['save_at']+' / hard '+protocol['hard_stop']+'. '+protocol['limits']+' Источник runs/z13_region_pooling_20261025/protocol.json. '+protocol['literature'])


def audit():
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p);report=read(RUN/'report.json');assert report['complete'] and report['exact_baseline_videos']==46
    values={a:{} for a in ARMS};count=0
    for video in p['videos']:
        path=RUN/'videos'/(video+'.json');assert sha(path)==report['video_sha256'][video]
        data=read(path);assert data['checks']['raw_replay_exact']
        previous=read(recovery.RUN/'videos'/(video+'.json'));old={r['uid']:r for r in previous['rows']}
        for row in data['rows']:
            ref=old[row['uid']];assert all(row[k]==ref[k] for k in ('uid','video','frame','tid','kind','label','split'))
            for arm in ('D','C','R1','geometry'):
                assert row['arms'][arm+'_mean'][1]==ref['arms'][arm][1]
                assert not row['arms'][arm+'_mean'][0]['beneficial'] and not row['arms'][arm+'_max'][0]['harmful']
            count+=1
        if video in p['evaluate26']:
            known=[r for r in data['rows'] if r['kind']>=0]
            for arm in ARMS:values[arm][video]=float(np.mean([int(r['arms'][arm][1]['beneficial'])-int(r['arms'][arm][1]['harmful']) for r in known]))
    comp=comparisons(values);assert comp==report['paired'] and values==report['per_video'] and primary(comp)==report['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all46_original_means_and_baselines_exact=True,all8_conditions_checked=True,query_rows=count,all_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    return report


def handoff(text):
    from scripts import z13_region_assignment_recovery2_20261025 as original
    original.namespace()['handoff'](text)


def finish():
    ps=WindowsProcesses();c=read(RUN/'controller_result.json')
    assert c['exit_code']==0 and c['worker_ended'] and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    r=audit();p=read(RUN/'protocol.json');passed=r['primary_pass'];cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Среднее против максимума: критерий '+('пройден.' if passed else 'не пройден.'),obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → paired: '+str(r['paired'])+';8парных условий, прежнийalpha0,25,46точныхповторов.',
        interpretation='Изменено только пространственное агрегирование той же карты внутри тех же областей. Это сравнение конкретных считывателей; знак не доказывает причину всех отказов и не является HOTA.',decision=p['next']+' Основной критерий: '+str(passed),cannot_claim=p['counterargument'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Проверены46видео и исходные средние.'))
        card[k]=v
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for i,value in enumerate((True,passed)):card['acceptance'][i].update(met=value,evidence=(RUN/('independent_audit.json' if i==0 else 'report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог парного пространственного считывания','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision'])
    handoff('Полный z13_region_pooling_v1 окончен. '+fields['obtained']+' '+fields['decision']+' Публикация runs/z13_region_pooling_20261025/publication.json. Проверить confirmed_results и общую архитектурную веху context/product/pooling. Все консоли скрыто по решению25.10; автор ушёл до утра.')
    publish('final')


def publish(stage):
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_region_pooling.py','-q']]):
        with (RUN/f'{stage}_checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/(stage+'_owned_graph.json');write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/('vault_'+stage+'.json')),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_pooling.py','tests/test_z13_region_pooling.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name not in ('publication.json','start_publication.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m',('Register' if stage=='start' else 'Evaluate')+' paired spatial pooling of FlyVis target responses','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/('start_publication.json' if stage=='start' else 'publication.json'),dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
            child.wait()
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--publish-start' in sys.argv:publish('start')
    else:raise RuntimeError('Explicit mode required')
