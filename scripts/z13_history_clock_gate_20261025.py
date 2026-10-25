"""Bounded prerequisite: quantify artificial elapsed time from duplicate exemplar padding."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import subprocess
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_history_clock_gate_20261025'
IDENT='z13_history_clock_gate_v1'


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def prepare():
    assert not RUN.exists()
    previous=ROOT/'runs/z13_region_semantic_20261025'
    assert read(previous/'publication.json')['push_completed'] and not read(previous/'training/report.json')['primary_pass']
    c=read(previous/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(previous/'protocol.json');start=now();stop=start+timedelta(seconds=120)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_data.py','scripts/z13_train.py','scripts/z13_target_model.py','tracking/z13_readout.py','tracking/flyvis_motion.py','runs/z13_region_semantic_20261025/training/report.json']
    p=dict(old,id=IDENT,created=start.isoformat(),budget_seconds=120,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Candidate internal timing change: do not advance recurrent state on zero-elapsed duplicate exemplar-padding entries. Preserve every genuine observed/search frame, state across genuine steps, eye geometry, learned parameters and old normalization/readout. This gate only checks whether the trigger has enough exposure before any GPU pilot.',
        question='How many unchanged natural queries inherit duplicate leading exemplar timestamps or missing frame intervals, and are zero-time entries exact leading copies with the same crop?',
        acceptance='Each query/episode/frame/crop binding valid; nondecreasing actual timestamps. Count duplicates only before last exemplar and verify identical frame,crop,hex data. Admission for a skip-padding prototype only if mean-video fraction of known evaluation queries with such duplicates>=0.02 and paired lower95%CI>0; this is an affected-subset upper-bound inventory, not evidence of improvement. Other gaps reported but not silently repaired.',
        distinction='Earlier efference_eye transported state/reference or stabilized image coordinates to estimate decoded flow. This does not warp the retina or cancel real image motion. It changes only whether a repeated storage entry with the same video timestamp consumes artificial neural time. Prior bad results are preserved.',
        literature='Lappalainen et al.,Nature2024,Connectome-constrained networks predict neural activity across the fly visual system,Methods/task-optimization dataset: https://www.nature.com/articles/s41586-024-07939-3 . Temporal resampling respects physical sampling intervals. This supports examining simulated time, not a published tracking gain or a claim that intentional padding is a service bug.',
        counterargument='The initial fade-in may already be close to a steady state, so extra constant-input steps can have negligible influence or even help initialization. Changing timing can shift the distribution learned by D. Frequent padding alone proves neither failure causality nor a useful replacement. Thresholds and core weights are not changed by this gate.',
        next='If enough exposure and exact leading-copy semantics are verified, separately register a bounded frozen-core skip-padding prototype with same-history C/D/R1 controls and saved old responses; no new core training. Otherwise preserve the specific insufficient trigger and choose another state/input node. No HOTA from counts.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Actual exemplar timestamps and duplicate padding',done=0,total=130,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Временное состояние FlyVis: мишень искусственных повторов образца',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P3','fly-problem-P8'],
        question=p['question'],hypothesis='Рабочая гипотеза: повтор кадров для заполнения четырёх мест образца может добавлять состояние без нового времени видео. Наличие этого механизма и влияние на выход — разные вопросы.',expected=p['action'],acceptance=[dict(criterion=p['acceptance'],met=None)],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие, дешёвая мишень и критерий до измерения')],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка мишени временного состояния после отрицательного считывателя',
        '*Действие.* '+p['action']+'\n\n*Проверка до GPU.* '+p['acceptance']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Возражение.* '+p['counterargument']+'\n\nCPU120с+20с, скрыто, save '+p['save_at']+' / hard '+p['hard_stop']+'. НикакихGPU/обучения/dev/HOTA/75/загрузок. Источник runs/z13_history_clock_gate_20261025/protocol.json. '+p['literature'])


def worker():
    p=read(RUN/'protocol.json');verify_sources(p);queries,_=original.namespace()['queries_and_crops']();start=time.monotonic();records={}
    for i,(video,items) in enumerate(queries.items()):
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original120s clock gate deadline')
        with np.load(ROOT/'runs/z13_data'/(video+'.npz')) as z:
            offsets=z['ep_offset'];frames=z['fr_frame'];crops=z['fr_crop'];hexes=None
            cases=[]
            for q in items:
                e=q['episode_local'];end=4+q['label']['pos'];lo=int(offsets[e]);ff=frames[lo:lo+end+1];cc=crops[lo:lo+end+1]
                assert len(ff)==end+1 and ff[-1]==q['frame'] and np.array_equal(cc[-1],q['label']['crop'])
                delta=np.diff(ff);assert np.all(delta>=0)
                zero=np.flatnonzero(delta==0)
                if len(zero):
                    assert np.array_equal(zero,np.arange(len(zero))) and int(zero[-1])<3
                    if hexes is None:hexes=z['fr_hex']
                    for j in zero:assert np.array_equal(cc[j],cc[j+1]) and np.array_equal(hexes[lo+j],hexes[lo+j+1])
                cases.append(dict(uid=q['uid'],kind=q['label']['kind'],duplicate_steps=len(zero),missing_steps=int(np.maximum(delta-1,0).sum()),max_delta=int(delta.max(initial=0))))
        write(RUN/'videos'/(video+'.json'),dict(video=video,cases=cases,exact_leading_copies=True))
        records[video]=dict(rows=len(cases),known=sum(r['kind']>=0 for r in cases),duplicates_known=sum(r['kind']>=0 and r['duplicate_steps']>0 for r in cases),gaps_known=sum(r['kind']>=0 and r['missing_steps']>0 for r in cases),duplicate_steps=sum(r['duplicate_steps'] for r in cases),missing_steps=sum(r['missing_steps'] for r in cases),max_delta=max((r['max_delta'] for r in cases),default=0))
        write(RUN/'status.json',dict(state='running',stage='Actual frame intervals; no neural execution',variant=video,done=i+1,total=130,updated=now().isoformat(),elapsed_seconds=time.monotonic()-start))
    from scripts.z13_continue import paired
    values={v:records[v]['duplicates_known']/records[v]['known'] for v in p['evaluate26']};effect=paired(values,{v:0. for v in values});passed=effect['mean_per_video']>=.02 and effect['ci95'][0]>0
    write(RUN/'report.json',dict(complete=True,admission_pass=passed,affected_query_fraction=effect,per_video=records,all_zero_time_entries_exact_leading_copies=True,
        total_queries=sum(r['rows'] for r in records.values()),known_queries=sum(r['known'] for r in records.values()),seconds=time.monotonic()-start,
        video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in records},hota_measured=False,claim='Timing-trigger inventory only; no neural-response or tracking benefit measured.'))
    write(RUN/'status.json',dict(state='completed',stage='Timestamp gate saved',done=130,total=130,updated=now().isoformat()))


def finish():
    ps=WindowsProcesses();c=read(RUN/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'report.json');assert r['complete'] and r['total_queries']==37296 and r['known_queries']==26600
    totals={}
    for v,h in r['video_sha256'].items():
        assert sha(RUN/'videos'/(v+'.json'))==h
        rows=read(RUN/'videos'/(v+'.json'))['cases'];assert len(rows)==r['per_video'][v]['rows']
        totals[v]=sum(x['kind']>=0 and x['duplicate_steps']>0 for x in rows)/sum(x['kind']>=0 for x in rows)
    from scripts.z13_continue import paired
    effect=paired({v:totals[v] for v in p['evaluate26']},{v:0. for v in p['evaluate26']});assert effect==r['affected_query_fraction']
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    card.update(state='completed_positive' if r['admission_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),
        summary='Мишень повторного времени '+('достаточна для отдельной пробы.' if r['admission_pass'] else 'не прошла критерий.'),
        obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → affected_query_fraction: '+str(effect)+';37296запросов,26600известных; все нулевые интервалы — точные ведущие копии.',
        interpretation='Измерена только частота триггера искусственного времени. Не измерены ошибка активности, улучшение представления или HOTA. Положительный статус относится только к наличию мишени.',
        decision=p['next']+' Admission: '+str(r['admission_pass']))
    card['acceptance'][0].update(met=bool(r['admission_pass']),evidence=(RUN/'report.json').relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,admission_pass=r['admission_pass']))
    journal('итог частоты искусственного времени образца','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    original.namespace()['handoff']('Завершён z13_history_clock_gate_v1. '+card['obtained']+' '+card['decision']+' Источник только частота триггера, не улучшение/HOTA. Предыдущий семантический узел отрицателенf0a68099; архитектурную ветвь сохранить при следующей объединённой вехе. Все консоли скрыто,57чужихправок/.gitignore сохранять.')
    publish()


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (RUN/f'checks_{i}.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name!='publication.json']
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()];names=sorted(set(names))
    assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists() and subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Measure duplicate-time exposure before a FlyVis state intervention','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',str(Path(__file__))],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);child.wait()
        finish()
    except Exception as error:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(error)));raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
