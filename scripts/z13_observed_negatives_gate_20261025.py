"""Deferred CPU prerequisite: reliable same-class negatives in the observed exemplar."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,inspect,hashlib,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts import z13_history_clock_gate_20261025 as template
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
PARENT=ROOT/'runs/z13_ridge_uncertainty_20261025'
RUN=ROOT/'runs/z13_observed_negatives_gate_20261025'
IDENT='z13_observed_negatives_gate_v1'
ACCEPTANCE='Exact130video/37296query/26600known bindings, nonempty observed mask, every negative uses only last-observed frame/crop, same tracker class, conf>=0.25, IoU<=0.1 with the lattice-reconstructed observed box, >=1column outside its1.25expanded guard, supported center by the unchanged lattice rule. Admission requires lower paired-video95%CI of known evaluate26 queries with such an observed negative AND a reachable beneficial region action>0.02; upper<0.02 insufficient, otherwise inconclusive. This is only causal-input availability and oracle exposure, not an achieved benefit. No relaxing this criterion after result.'

def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)

def intent():
    assert not (PARENT/'next_observed_negatives_intent.json').exists()
    write(PARENT/'next_observed_negatives_intent.json',dict(at=now().isoformat(),condition='Only after full negative uncertainty result, primary audit and successful publication; positive/error -> no next scientific calculation.',action='Candidate target-specific ridge update: weight4on separable same-class detector-negative columns in the last OBSERVED exemplar, weight1elsewhere; same Gaussian target/core/readout and matched unweightedC/D/R1 controls. This queue permits only the cheap availability prerequisite, not extraction or fitting.',acceptance=ACCEPTANCE,source_sha256=sha(Path(__file__)),budget_seconds=180,author_save_at='2026-10-26T07:35:00+03:00',claim='Registered before the uncertainty response result; criterion not selected from its success/failure magnitudes.'))

def prepare():
    assert not RUN.exists();assert read(PARENT/'publication.json')['push_completed'] and not read(PARENT/'replay/report.json')['primary_pass']
    queued=read(PARENT/'next_observed_negatives_intent.json');assert queued['source_sha256']==sha(Path(__file__)) and queued['acceptance']==ACCEPTANCE
    ps=WindowsProcesses()
    for phase in ('features','replay'):
        c=read(PARENT/phase/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(original.RUN/'protocol.json');verify_sources(old);start=now();stop=min(start+timedelta(seconds=180),datetime.fromisoformat(queued['author_save_at']));assert (stop-start).total_seconds()>=60
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_data.py','tracking/z13_readout.py','scripts/z13_history_clock_gate_20261025.py','runs/z13_region_assignment_20261025/labels/report.json','runs/z13_ridge_uncertainty_20261025/next_observed_negatives_intent.json']
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),device='cpu',budget_seconds=180,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),action=queued['action'],acceptance=ACCEPTANCE,
        literature='Zhu et al.,Distractor-aware Siamese Networks for Visual Object Tracking,ECCV2018,section3.2semantic negative pairs and3.3incremental learning https://arxiv.org/html/1808.06048 . Our observed-mask weighting is a limited ridge-target adaptation, not DaSiamRPN or its published gains. Existing Gaussian targets already include ordinary background; this tests separately identified same-class observed neighbors.',
        counterargument='A nearby detector box may be a duplicate or a wrong identity, while conservative overlap exclusion can discard the hard overlapping distractors. Observable negatives may be common but uninformative; weighting can suppress useful target features. A full prototype must use identical causal masks forC/D/R1, oldmean parity, fixed before-evaluation settings and actual assignment controls; no presumed cause or HOTA.',
        next='Admission -> separate preregistered bounded weighted/unweighted target-model prototype; no automatic core training orHOTA. Insufficient/inconclusive -> retain this exact definition and do not change IoU/expansion/2pp based on the result.',sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Only past observed distractors; CPU',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Модель конкретной цели: доступность наблюдённых отрицательных соседей',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Хватает ли причинно доступных отрицательных соседей образца для отдельной проверки их веса в модели цели?',hypothesis='Рабочая гипотеза: одинаковые объекты рядом с целью могут требовать отдельной подгонки к конкретному треку. Обычные отрицательные колонки уже есть; их отсутствие не заявляется.',expected=p['action'],acceptance=[dict(criterion=ACCEPTANCE,met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Отложенная мишень и критерий из зарегистрированной заранее очереди')]))
    write(ROOT/'experiments/project_history.json',cat);journal('постановка наблюдённых отрицательных соседей','*Действие.* '+p['action']+'\n\n*Критерий до расчёта.* '+ACCEPTANCE+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Источник.* '+p['literature']+'\n\n*Бюджет.* CPU180с+20с, безGPU/обучения/dev/HOTA/75/загрузок, скрыто, save '+p['save_at']+' / hard '+p['hard_stop']+'. Источник runs/z13_observed_negatives_gate_20261025/protocol.json.')

def worker():
    from tracking.z13_readout import to_eye
    p=read(RUN/'protocol.json');verify_sources(p);ns=original.namespace();queries,_=ns['queries_and_crops']();geom=ns['geometry']();records={}
    labelroot=ROOT/'runs/z13_region_assignment_20261025/labels';manifest=read(labelroot/'report.json')['video_sha256']
    for i,(video,items) in enumerate(queries.items()):
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original observed-negatives deadline')
        path=labelroot/'videos'/(video+'.json');assert sha(path)==manifest[video];labels={x['uid']:x for x in read(path)['rows']}
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:off=z['ep_offset'];frames=z['fr_frame'];crops=z['fr_crop'];masks=z['ex_mask']
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz') as z:d={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
        cache={};cases=[]
        for q in items:
            row=labels[q['uid']];ep=q['episode_local'];category=row['regions']['track_class'];key=(ep,category)
            assert row['video']==video and row['kind']==q['label']['kind'] and row['frame']==q['frame']
            if key not in cache:
                k=int(off[ep])+3;f=int(frames[k]);crop=crops[k];assert f<q['frame'];mask=masks[ep];assert mask.any()
                selected=geom.points[mask];lo=selected.min(0)-geom.spacing/2;hi=selected.max(0)+geom.spacing/2;mid=(lo+hi)/2;half=(hi-lo)*1.25/2
                guard=((geom.points>=mid-half)&(geom.points<=mid+half)).all(1);union=np.zeros(len(mask),bool);count=0
                ix=(d['frame_idx']==f)&(d['conf']>=.25)&(d['cls']==category)
                for b in d['xyxy'][ix]:
                    a=np.asarray(to_eye(b[:2],crop,geom.eye_size));bb=np.asarray(to_eye(b[2:],crop,geom.eye_size));center=(a+bb)/2
                    supported=bool(np.all(center>=0) and np.all(center<=geom.eye_size) and np.linalg.norm(geom.points[geom.nearest(center)]-center)<=.6*geom.spacing)
                    inter=float(np.maximum(np.minimum(hi,bb)-np.maximum(lo,a),0).prod());area=float(np.maximum(bb-a,0).prod());targetarea=float((hi-lo).prod());iou=inter/max(area+targetarea-inter,1e-9)
                    neg=((geom.points>=a)&(geom.points<=bb)).all(1)&~guard
                    if supported and iou<=.1 and neg.any():union|=neg;count+=1
                assert not (union&guard).any();cache[key]=dict(observed_frame=f,negative_regions=count,negative_columns=int(union.sum()))
            available=cache[key]['negative_regions']>0;beneficial=any(a['beneficial'] for a in row['arms']['regions'][1:])
            cases.append(dict(uid=q['uid'],kind=row['kind'],observed=cache[key],negative_available=available,reachable=bool(available and beneficial)))
        write(RUN/'videos'/(video+'.json'),dict(video=video,cases=cases,past_observed_only=True))
        records[video]=dict(rows=len(cases),known=sum(x['kind']>=0 for x in cases),available=sum(x['kind']>=0 and x['negative_available'] for x in cases),reachable=sum(x['kind']>=0 and x['reachable'] for x in cases))
        write(RUN/'status.json',dict(state='running',stage='Отрицательные соседи последнего наблюдения',done=i+1,total=130,variant=video,updated=now().isoformat()))
    from scripts.z13_continue import paired
    effect=paired({v:records[v]['reachable']/records[v]['known'] for v in p['evaluate26']},{v:0. for v in p['evaluate26']})
    state='admit' if effect['ci95'][0]>.02 else ('insufficient_exposure' if effect['ci95'][1]<.02 else 'inconclusive')
    verify_sources(p);write(RUN/'report.json',dict(complete=True,admission_pass=state=='admit',gate_state=state,observed_negative_oracle_exposure=effect,per_video=records,total_queries=sum(x['rows'] for x in records.values()),known_queries=sum(x['known'] for x in records.values()),video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in records},hota_measured=False))
    write(RUN/'status.json',dict(state='completed',stage=state,updated=now().isoformat()))

def finish():
    ps=WindowsProcesses();c=read(RUN/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'report.json');assert r['complete'] and r['total_queries']==37296 and r['known_queries']==26600;values={}
    for v,h in r['video_sha256'].items():
        path=RUN/'videos'/(v+'.json');assert sha(path)==h;rows=read(path)['cases'];values[v]=sum(x['kind']>=0 and x['reachable'] for x in rows)/sum(x['kind']>=0 for x in rows)
    from scripts.z13_continue import paired
    effect=paired({v:values[v] for v in p['evaluate26']},{v:0. for v in p['evaluate26']});assert effect==r['observed_negative_oracle_exposure']
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    card.update(state={'admit':'completed_positive','insufficient_exposure':'closed','inconclusive':'inconclusive'}[r['gate_state']],done=now().strftime('%Y-%m-%d'),summary='Наблюдённые отрицательные соседи: '+r['gate_state'],obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → observed_negative_oracle_exposure: '+str(effect)+';37296запросов/26600известных, только прошлый наблюдённый кадр.',interpretation='Измерена доступность причинных отрицательных регионов и оракульная достижимость. Не доказаны различимость объектов, правильность весового обновления или польза стенду.',decision=p['next']+' Gate: '+r['gate_state'])
    card['acceptance'][0].update(met=bool(r['admission_pass']),evidence=(RUN/'report.json').relative_to(ROOT).as_posix());write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,gate_state=r['gate_state']))
    journal('итог доступности наблюдённых отрицательных соседей','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    original.namespace()['handoff']('Завершён z13_observed_negatives_gate_v1 после полного отрицательногоuncertainty. '+card['obtained']+' '+card['decision']+' Полный weighted/unweighted исполнитель ещё не создан. Не ослаблять критерий, не путать доступность с улучшением. Всё скрыто,57чужихправок/.gitignore сохранять; архитектурную вехуuncertainty проверить.')
    publish()

def publish():
    code=inspect.getsource(template.publish).replace('Measure duplicate-time exposure before a FlyVis state intervention','Check causal observed distractors before target-specific adaptation')
    ns=dict(globals());exec(compile(code,__file__,'exec'),ns);ns['publish']()

def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',__file__],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);child.wait()
        finish()
    except Exception as e:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(e)));raise

def watch_parent():
    intent_path=PARENT/'next_observed_negatives_intent.json';assert read(intent_path)['source_sha256']==sha(Path(__file__))
    ps=WindowsProcesses();write(PARENT/'next_gate_watcher.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),next_script=Path(__file__).relative_to(ROOT).as_posix()))
    while now()<datetime.fromisoformat('2026-10-26T07:34:00+03:00'):
        if (PARENT/'publication.json').exists():
            assert read(PARENT/'publication.json')['push_completed']
            if read(PARENT/'replay/report.json')['primary_pass']:write(PARENT/'next_gate_result.json',dict(at=now().isoformat(),state='not_started_parent_positive'));return
            prepare();pipeline();write(PARENT/'next_gate_result.json',dict(at=now().isoformat(),state='next_gate_finished'));return
        c=read(PARENT/'pipeline.json')
        if ps.creation(c['pid'])!=c['birth']:
            write(PARENT/'next_gate_result.json',dict(at=now().isoformat(),state='parent_ended_without_publication_no_next_calculation'));return
        time.sleep(5)
    write(PARENT/'next_gate_result.json',dict(at=now().isoformat(),state='author_window_ended_no_next_calculation'))

if __name__=='__main__':
    if '--intent' in sys.argv:intent()
    elif '--watch-parent' in sys.argv:watch_parent()
    elif '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
