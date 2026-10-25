"""One changed condition: equal total fit weight of actionable/no-action queries."""
from pathlib import Path
from datetime import timedelta,datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import ast
import hashlib
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts import z13_localization_labels_20261023 as notes
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
from tracking.z13_region_balanced_loss import balanced_action_loss
BASE=original.RUN
RUN=ROOT/'runs/z13_region_balanced_20261025'
IDENT='z13_region_balanced_v1'


def journal(title,text):
    notes.IDENT=IDENT;notes.journal(title,text)


def training_source():
    source=original.source();tree=ast.parse(source)
    code=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_worker'))
    replacements=[
      ("known=np.flatnonzero(data['fit84']['kind'].numpy()>=0);assert len(known)>128",
       "known=np.flatnonzero(data['fit84']['kind'].numpy()>=0);assert len(known)>128\n        positives=int(data['fit84']['good'][known].any(1).sum());assert 0<positives<len(known)\n        positive_weight=(len(known)-positives)/positives;assert positives==p['fit_positive'] and len(known)==p['fit_known']"),
      ("loss=marginal_choice_loss(torch,scores,d['good'][idx],d['mask'][idx],1.)",
       "loss=balanced_action_loss(torch,scores,d['good'][idx],d['mask'][idx],positive_weight)"),
      ("train_known=len(known),cuda_initialized=", "train_known=len(known),positive_weight=positive_weight,train_positive=positives,cuda_initialized="),
      ("passed=comp['D-baseline']['mean_per_video']>=.02", "comp['D-balanced-D-uniform']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])\n    passed=comp['D-baseline']['mean_per_video']>=.02")]
    for old,new in replacements:
        assert code.count(old)==1,old;code=code.replace(old,new)
    return code


def status(phase,stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original CPU budget ended')
    write(RUN/'status.json',dict(state='running',phase='training',stage=stage,updated=now().isoformat(),**kw))


def worker():
    ns=original.namespace();load=ns['load_split']
    ns={**ns,'RUN':RUN,'BASE':BASE,'load_split':load,'balanced_action_loss':balanced_action_loss,'status':status}
    p=read(RUN/'protocol.json');assert hashlib.sha256(training_source().encode()).hexdigest()==p['executed_training_sha256']
    exec(compile(training_source(),str(Path(__file__)),'exec'),ns)
    ns['train_worker']()
    write(RUN/'status.json',dict(state='completed',stage='All12 balanced readouts saved',updated=now().isoformat()))


def prepare():
    assert not RUN.exists()
    e=read(BASE/'training/controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    old=read(BASE/'training/report.json');assert old['complete'] and not old['primary_pass']
    audit=read(ROOT/'runs/z13_region_fit_audit_20261025/report.json');assert audit['complete'] and not audit['cuda_initialized']
    feature=read(BASE/'features/report.json');assert feature['complete']
    for name,h in feature['manifest'].items():
        marker=ROOT/name;assert sha(marker)==h and sha(marker.with_suffix('.npz'))==read(marker)['sha256']
    positive=known=0
    for path in (BASE/'features/D/fit84').glob('chunk_*.npz'):
        with np.load(path) as z:
            k=z['kind']>=0;known+=int(k.sum());positive+=int((k&z['good'].any(1)).sum())
    assert (known,positive)==(17859,1356)
    start=now();stop=min(start+timedelta(seconds=1800),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    sources=['tracking/z13_region_balanced_loss.py','tests/test_z13_region_balanced_loss.py',Path(__file__).relative_to(ROOT).as_posix(),
       'scripts/z13_region_balanced_console.ps1','scripts/z13_bounded_recovery_controller_20261022.py',
       (BASE/'features/report.json').relative_to(ROOT).as_posix(),(BASE/'training/report.json').relative_to(ROOT).as_posix(),
       'runs/z13_region_fit_audit_20261025/report.json']
    p=dict(read(BASE/'protocol.json'),at=start.isoformat(),budget_seconds=1800,device='cpu',timezone='Europe/Moscow',save_at=stop.isoformat(),
       hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
       action='Only query-class weights change: positive=Nnegative/Npositive fitted on fit84; normalize by actual batch weight sum. Same UID minibatches, model initialization,1150steps/128,three seeds, fit20 threshold grid and train26 evaluation.',
       fit_known=known,fit_positive=positive,positive_weight=(known-positive)/positive,
       primary='D-baseline mean>=2pp and all five paired-video lower95% bounds>0: baseline,C,R1,geometry,old uniform D.',
       distinction='Earlier balanced_ridge balanced retinal columns inside/outside an observed frame. This balances whole assignment queries with/without any beneficial action; frozen core, inputs and actions stay unchanged.',
       literature='Lin et al., ICCV2017, Focal Loss for Dense Object Detection, section3.1 balanced cross entropy: https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf . Mechanistic analogy, not the same tracking task or full focal loss.',
       counterargument='Forced recovery is also high for geometry: rare-class weighting may only increase false interventions or help the external geometric rule. Positive D-C/R1/geometry contrasts remain mandatory; missing global assignment context is unresolved.',
       next='Pass -> separate gated closed-loop runtime parity/price/HOTA protocol. Fail -> retain outcome; no HOTA for this version, inspect missing readout context rather than weakening criteria.',
       executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),
       sha256={**read(BASE/'protocol.json')['sha256'],**{n:sha(ROOT/n) for n in sources}})
    write(RUN/'protocol.json',p);write(RUN/'training/protocol.json',p)
    cat=read(ROOT/'experiments/project_history.json');parent=next(c for c in cat['experiments'] if c['id']=='z13_region_assignment_v1')
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Равный вес запросов с полезным действием и без него',area=parent['area'],state='ready',
       question='Поможет ли уравнивание классов вмешательства в обучении внутреннего регионального считывателя?',
       hypothesis='Рабочая гипотеза: преобладание запросов без полезного действия подавляет обучение вмешательства. Не доказанная причина; признаки областей могут быть недостаточны.',
       expected='Увеличить полезные назначения при контроле вредных; D должна превосходить C/R1/геометрию и прежнюю D при равном обучении.',
       acceptance=[dict(criterion='Все12 моделей, неизменные признаки/пакеты/пороги выбора, первичные строки и хеши подтверждены',met=None),
                   dict(criterion=p['primary'],met=None)],started=start.strftime('%Y-%m-%d'),evidence=[(RUN/'protocol.json').relative_to(ROOT).as_posix(),'runs/z13_region_fit_audit_20261025/report.json'],
       tags=['fly-problem-P8'],obtained='',interpretation='',decision='',cannot_claim='Однокадровые назначения на многократно использованных train26; не HOTA. Вклад изменения веса не доказывает причину всех отказов мухи.'))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка равного веса запросов вмешательства',
       '*Постановка.* Меняется только взвешивание ошибки целого запроса: '+str(known-positive)+' без полезного действия и '+str(positive)+' с ним на fit84; вес положительного класса '+str(p['positive_weight'])+'. Нормировка по сумме весов пакета. '
       'Прежние признаки/пакеты/инициализации/1150шагов/три сида, fit20-порог и train26. Это не балансировка колонок balanced_ridge. '
       '*Основание.* D/fit84/seed0: принудительно942 полезных из1356 доступных, при обычном выборе24; fit20:96 из160 и0. Источник runs/z13_region_fit_audit_20261025/report.json. '
       'Причина дисбаланса остаётся гипотезой; у геометрии принудительный выбор тоже часто удачен. '+p['primary']+' CPU1800с с независимым контроллером, без GPU и повторного извлечения. Источник '+(RUN/'protocol.json').relative_to(ROOT).as_posix()+'. '+p['literature'])


def finish():
    import torch
    from scripts.z13_continue import paired
    from tracking.z13_filter_descriptor import make_ranker
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p);e=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    r=read(RUN/'training/report.json');assert r['complete'] and r['models']==12 and not r['cuda_initialized']
    original_ns=original.namespace();models={};means={}
    for variant in ('D','C','R1','geometry'):
        data=original_ns['load_split'](torch,'D' if variant=='geometry' else variant,'train26')
        models[variant]={}
        for seed in p['seeds']:
            dest=RUN/'training'/variant/f'seed{seed}';done=read(dest/'done.json');m=read(dest/'report.json')
            assert sha(dest/'model.pt')==done['model_sha256'] and sha(dest/'report.json')==done['report_sha256']
            assert sha(dest/'rows.json')==m['rows_sha256'] and m['positive_weight']==p['positive_weight'] and m['steps']==p['steps']
            assert read(dest/'calibration.json')['tau']==m['tau']
            model=make_ranker(torch,activity=variant!='geometry').cpu()
            model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
            rows,per,count=original_ns['policy_rows'](data,original_ns['predict'](torch,model,data),m['tau'],p['evaluate26'])
            assert rows==read(dest/'rows.json') and per==m['per_video'] and count==m['counts']
            models[variant][seed]=m
        means[variant]={v:float(np.mean([models[variant][s]['per_video'][v] for s in p['seeds']])) for v in p['evaluate26']}
    comp={'D-baseline':paired(means['D'],{v:0. for v in p['evaluate26']})}
    for variant in ('C','R1','geometry'):comp['D-'+variant]=paired(means['D'],means[variant])
    comp['D-balanced-D-uniform']=paired(means['D'],read(BASE/'training/report.json')['mean_seed_per_video']['D'])
    assert comp==r['paired']
    passed=comp['D-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in comp.values())
    assert passed==r['primary_pass'];assert not torch.cuda.is_initialized()
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all12_models_repredicted_on_all8326_queries=True,
        rows_counts_intervals_recomputed=True,report_sha256=sha(RUN/'training/report.json'),complete=True))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Равный вес запросов: основной критерий '+('пройден.' if passed else 'не пройден.'),
       obtained=(RUN/'training/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(comp)+';12моделей,3сида, признаки прежние, fit84/fit20/train26.',
       interpretation='Изменён только вес классов запроса. Результат не доказывает причину всех отказов; качество принудительного выбора не равно качеству допуска. Однокадровые последствия, не замкнутый трекер.',
       decision=p['next']+' Измеренный основной критерий: '+str(passed),cannot_claim=p['counterargument']+' HOTA не измерялась; доля потолка стенда неизвестна.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Проверен полный пилот взвешивания классов.'))
        card[key]=value
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for index,met in enumerate((True,passed)):card['acceptance'][index].update(met=met,evidence=(RUN/('independent_audit.json' if index==0 else 'training/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat)
    journal('итог равного веса запросов вмешательства','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision'])
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    original.namespace()['handoff']('Балансировка классов регионального узла закончена. '+fields['obtained']+' '+fields['decision']+' Проверить отдельные подтверждённые сравнения и архитектурную веху; публикация '+(RUN/'publication.json').relative_to(ROOT).as_posix()+'.')
    publish()


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for index,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_region_balanced_loss.py','-q']]):
        with (RUN/f'checks_{index}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph_before.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=['tracking/z13_region_balanced_loss.py','tests/test_z13_region_balanced_loss.py',Path(__file__).relative_to(ROOT).as_posix(),
       'scripts/z13_region_balanced_console.ps1','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names+=[x.relative_to(ROOT).as_posix() for x in RUN.rglob('*.json') if x.name!='publication.json']
    names+=[n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Evaluate balanced action supervision of regional FlyVis readout','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def monitor():
    try:
        p=read(RUN/'protocol.json')
        while not (RUN/'controller_result.json').exists():
            if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=30):raise TimeoutError('Controller completion missing')
            time.sleep(2)
        finish()
    except Exception as error:
        write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--monitor' in sys.argv:monitor()
    else:raise RuntimeError('Explicit mode required')
