"""Fresh author-requested bounded pilot; preserve and reuse expired earlier work."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import inspect
import hashlib
import time
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_full_box_supervision_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired
from tracking.z13_action_feasibility import helpful_box_mask
RUN=ROOT/'runs/z13_box_supervision_followup_20261025'
OLD=base.RUN
IDENT='z13_box_supervision_followup_v1'
BUDGETS={'cache':3600,'training':1800}
PHASE=os.environ.get('Z13_BOX_FOLLOWUP_PHASE','')


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def status(phase,stage,**kw):
    p=read(RUN/phase/'protocol.json')
    if (RUN/phase/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Fresh followup phase deadline')
    write(RUN/phase/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def cache_source():
    s=inspect.getsource(base.cache_worker)
    s=s.replace('from tracking.z13_box_action_labels import full_box_rows','from tracking.z13_box_action_batch import FullBoxCosts\n    full_box_rows=FullBoxCosts()')
    old="dest=RUN/'cache/videos'/(v+'.json');assert not dest.exists()"
    new="""dest=RUN/'cache/videos'/(v+'.json')
        if dest.exists():
            assert sha(dest)==p['reused_completed'][v]
            record=read(dest);assert record['complete'] and record['checks']['raw_replay_exact']
            entries=record['entries'];manifest[v]=sha(dest);total+=len(entries);known+=sum(x['kind']>=0 for x in entries.values())
            continue"""
    assert s.count(old)==1;s=s.replace(old,new)
    s=s.replace('if i==1:','if len(times)==2:').replace('2*float(np.mean(times))*128+60','2*float(np.mean(times))*(len(videos)-i-1)+60')
    old='assert total==37296 and known==26600';assert s.count(old)==1
    s=s.replace(old,old+"\n    assert full_box_rows.checked>=128\n    write(RUN/'cache/native_cost_parity.json',dict(rows_checked=full_box_rows.checked,batch_scalar_bit_exact=True))")
    return s


def prepare():
    assert not RUN.exists()
    math=read(ROOT/'runs/z13_math_priority_20261025/report.json');assert math['admit_strict_box_label_pilot']
    old=read(OLD/'protocol.json');verify_sources(old)
    partial=read(OLD/'partial_recorded.json');assert partial['trained_models']==0 and partial['scientific_verdict'] is None
    ps=WindowsProcesses()
    for path in (OLD/'cache/controller_result.json',OLD/'cache/recovery1/controller_result.json'):
        e=read(path);assert e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    complete={f.stem:sha(f) for f in (OLD/'cache/videos').glob('*.json')};assert len(complete)==4
    for video,h in complete.items():
        source=OLD/'cache/videos'/(video+'.json');assert partial['source_hashes'][source.relative_to(ROOT).as_posix()]==h
    start=now();auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    stop=min(start+timedelta(seconds=sum(BUDGETS.values())+600),datetime.fromisoformat(auth['save_at']))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_full_box_supervision_20261025.py',
        'scripts/z13_math_priority_20261025.py','tracking/z13_action_feasibility.py','tracking/z13_box_action_batch.py',
        'scripts/z13_bounded_recovery_controller_20261022.py','runs/z13_math_priority_20261025/report.json',
        (OLD/'partial_recorded.json').relative_to(ROOT).as_posix()]
    names += [(OLD/'cache/videos'/(v+'.json')).relative_to(ROOT).as_posix() for v in complete]
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',phase_budgets=BUDGETS,
        pipeline_save_at=stop.isoformat(),pipeline_hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(auth['hard_stop'])).isoformat(),
        budget_seconds=sum(BUDGETS.values())+600,save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(auth['hard_stop'])).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        fresh_author_request='25.10: KeepTrack вроде плохо подходил; распределить приоритеты, провести мат. тесты, далее настоящий прогон.',
        fresh_budget_reason='New explicitly requested scientific pilot after the completed mathematical gate, not a technical retry or extension of the expired1200s stage. Separate run/protocol/controller. Preserve old sources, failures and4completed videos; no old process resumes.',
        reused_completed=complete,actual_action_gate='Before any model: full observed box actions excluding explicit NULL have oracle mean>=.02 and paired-video95CI lower>0 on26; otherwise stop before training. This strict support gate is narrower than the parent geometry oracle.',
        technical='Same original130raw outputs bit-exact;37296queries/26600known; exact regional choices/context/feature bindings. Full-box batch distances checked against native scalar for>=128actions. Reuse4immutable completed videos. Price after2new videos:2*mean*remaining+60 within fresh cache budget3600;training firstmodel2*time*23+60 within1800. All24model predictions independently reproduced from saved weights, labels and thresholds; three seeds and controls unchanged.',
        new_executed_cache_sha256=hashlib.sha256(cache_source().encode()).hexdigest(),
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    for v,h in complete.items():
        dest=RUN/'cache/videos'/(v+'.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((OLD/'cache/videos'/(v+'.json')).read_bytes());assert sha(dest)==h
    write(RUN/'status.json',dict(state='ready',phase='cache',stage='Новая проверка последствий рамки;4видео сохранены',done=4,total=130,updated=start.isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Повторная постановка полного действия FlyVis после математического допуска',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],
        question='Поможет ли неизменный внутренний считыватель при обучении последствиям именно полной рамки?',hypothesis=next(x for x in cat['experiments'] if x['id']=='z13_full_box_supervision_v1')['hypothesis'],
        expected=p['action']+' '+p['fixed'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],
        obtained='',interpretation='',decision='',cannot_claim=p['counterargument']+' Это адаптация цели действия, не воспроизведение полной пространственной сети SiamMOT.',
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новая просьба автора, отдельный бюджет и прежняя научная постановка'),dict(kind='json',ref='runs/z13_math_priority_20261025/report.json',claim='Пройденная предварительная математика, не прогноз модели')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('новый допущенный опыт после математической проверки',p['fresh_budget_reason']+'\n\n*Действие.* '+p['action']+'\n\n*Фиксировано.* '+p['fixed']+'\n\n*Критерии.* '+p['actual_action_gate']+' '+p['primary']+'\n\n*Бюджет.* CPU3600с метки и1800с обучение, каждый+20с, общий срок '+p['pipeline_hard_stop']+'. '+p['counterargument'])


def cache_worker():
    p=read(RUN/'cache/protocol.json');s=cache_source();assert hashlib.sha256(s.encode()).hexdigest()==p['new_executed_cache_sha256']
    ns={**vars(base),'RUN':RUN,'status':status};exec(compile(s,__file__,'exec'),ns);ns['cache_worker']()


def train_worker():
    s=base.loader_source();p=read(RUN/'training/protocol.json');assert hashlib.sha256(s.encode()).hexdigest()==p['executed_loader_sha256']
    ns={**vars(base.context),**vars(base),'RUN':RUN,'status':status,'original':base.context.original,'signed_action_loss':base.context.signed_action_loss}
    exec(compile(s,__file__,'exec'),ns);ns['train_worker']()


def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']));assert now()<stop
    extra={}
    if phase=='training':extra[(RUN/'cache/report.json').relative_to(ROOT).as_posix()]=sha(RUN/'cache/report.json')
    q=dict(p,budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);write(RUN/phase/'status.json',dict(state='ready',stage=phase,updated=now().isoformat()))


def strict_gate():
    p=read(RUN/'protocol.json');values={};counts={}
    for video in p['evaluate26']:
        rows=read(RUN/'cache/videos'/(video+'.json'))['entries'].values();known=[x for x in rows if x['kind']>=0]
        n=sum(helpful_box_mask(x['choices'],x['good']) for x in known);values[video]=n/len(known);counts[video]=dict(helpful_box=n,known=len(known))
    comp=paired(values,{v:0. for v in values});passed=comp['mean_per_video']>=.02 and comp['ci95'][0]>0
    write(RUN/'strict_action_gate.json',dict(complete=True,passed=passed,oracle_only=True,not_model_prediction=True,box_only_excludes_null=True,paired=comp,counts=counts))
    return passed


def prediction_audit():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');p=read(RUN/'protocol.json')
    contexts={}
    for video,h in read(RUN/'cache/report.json')['manifest'].items():
        path=RUN/'cache/videos'/(video+'.json');assert sha(path)==h
        for uid,e in read(path)['entries'].items():assert int(uid) not in contexts;contexts[int(uid)]=e
    ns=base.context.original.namespace();load=ns['load_split'];total=0
    from tracking.z13_region_assignment_context import make_context_ranker
    for arm in ('D','C','R1','geometry'):
        data=load(torch,'D' if arm=='geometry' else arm,'train26')
        for i,uid in enumerate(data['uid'].tolist()):
            n=int(data['mask'][i].sum());entry=contexts[uid];assert len(entry['context'])==n
            data['good'][i,:n]=torch.tensor(entry['good'],dtype=torch.bool);data['harm'][i,:n]=torch.tensor(entry['harm'],dtype=torch.bool)
        for mode in ('context','blank'):
            extra=torch.zeros((*data['mask'].shape,6),dtype=torch.float32)
            if mode=='context':
                for i,uid in enumerate(data['uid'].tolist()):extra[i,:int(data['mask'][i].sum())]=torch.tensor(contexts[uid]['context'])
            d=dict(data,geo=torch.cat([data['geo'],extra],-1))
            for seed in p['seeds']:
                dest=RUN/'training'/(arm+'_'+mode)/f'seed{seed}'
                model=make_context_ranker(torch,activity=arm!='geometry');model.load_state_dict(torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
                logits=ns['predict'](torch,model,d);rec=read(dest/'report.json')
                rows,per,counts=ns['policy_rows'](d,logits,rec['tau'],p['evaluate26'])
                assert rows==read(dest/'rows.json') and per==rec['per_video'] and counts==rec['counts'];total+=1
    assert total==24 and not torch.cuda.is_initialized()
    write(RUN/'prediction_reproduction.json',dict(at=now().isoformat(),models=total,all_saved_predictions_exact=True,cuda_initialized=False))


def finish():
    assert read(RUN/'prediction_reproduction.json')['all_saved_predictions_exact']
    s=inspect.getsource(base.finish)
    ns={**vars(base),'RUN':RUN,'IDENT':IDENT,'BUDGETS':BUDGETS,'journal':journal,'handoff':handoff}
    exec(compile(s,__file__,'exec'),ns);ns['finish']()


def handoff(text):
    base.handoff(text+' Новый прогон по явной просьбе автора; старый full_box_supervision не продлён. Публикация runs/z13_box_supervision_followup_20261025/publication.json. Перед следующей дорогой ступенью критерии обязательны.')


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');next(x for x in cat['experiments'] if x['id']==IDENT)['state']='running';write(ROOT/'experiments/project_history.json',cat)
    phase='cache'
    try:
        for phase in BUDGETS:
            phase_protocol(phase);folder=RUN/phase
            with (folder/'controller.stdout.log').open('w',encoding='utf8') as out,(folder/'controller.stderr.log').open('w',encoding='utf8') as err:
                child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',str(Path(__file__))],cwd=ROOT,
                    env={**os.environ,'Z13_BOX_FOLLOWUP_PHASE':phase},stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
                write(folder/'launch.json',dict(pid=child.pid,birth=ps.creation(child.pid),at=now().isoformat()))
                while child.poll() is None:
                    try:write(RUN/'status.json',dict(read(folder/'status.json'),phase=phase,observer_at=now().isoformat()))
                    except (OSError,ValueError):pass
                    time.sleep(2)
                child.wait()
            e=read(folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'],e
            assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()) and read(folder/'report.json')['complete']
            if phase=='cache' and not strict_gate():
                write(RUN/'stopped_at_gate.json',dict(at=now().isoformat(),reason='Strict box-only reserve did not meet unchanged admission criterion',trained_models=0));return
        prediction_audit();finish()
    except Exception as error:
        write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:
        try:
            {'cache':cache_worker,'training':train_worker,'pipeline':pipeline}[PHASE]()
            write((RUN if PHASE=='pipeline' else RUN/PHASE)/'status.json',dict(state='completed',stage='Фаза сохранена',updated=now().isoformat()))
        except Exception as e:
            write(RUN/PHASE/'failure.json',dict(at=now().isoformat(),error=repr(e)));raise
