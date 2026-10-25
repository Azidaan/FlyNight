"""Bounded geometric prerequisite for a spatially structured FlyVis region readout."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import inspect
import hashlib
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_assignment_recovery2_20261025 as original
from scripts import z13_history_clock_gate_20261025 as template
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_spatial_bins_gate_20261025'
IDENT='z13_spatial_bins_gate_v1'


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)


def prepare():
    assert not RUN.exists();previous=ROOT/'runs/z13_history_clock_20261025'
    assert read(previous/'publication.json')['push_completed'] and not read(previous/'replay/report.json')['primary_pass']
    ps=WindowsProcesses()
    for phase in ('features','replay'):
        c=read(previous/'recovery1'/phase/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(original.RUN/'protocol.json');verify_sources(old);start=now();stop=start+timedelta(seconds=180)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_assignment.py','tracking/z13_readout.py','scripts/z13_history_clock_gate_20261025.py',
        'runs/z13_history_clock_20261025/replay/report.json','runs/z13_region_pooling_20261025/report.json']
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=180,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Candidate internal readout: replace one mean-pooled68-channel vector per detector region by four ordered2x2 spatial-bin vectors. Keep original unit-clock core, weights and target model; equal-parameter control repeats the original mean into all4bins. This gate measures only geometric feasibility and oracle exposure, not prediction or performance.',
        question='Do enough known queries with a reachable beneficial action also have a correct and an incorrect real detector candidate, each with nonempty retinal support in all4ordered quadrants?',
        acceptance='All130video/37296query/26600known bindings, exact previous region choices and source hashes. Admission requires lower paired-video95%CI of the fraction of known evaluate26 queries with a beneficial action and a noncorrect competing region, both four-bin resolved, strictly>0.02. Upper<0.02 stops this specific spatial-bin prototype for insufficient exposure; otherwise inconclusive. This is an oracle availability ceiling, not achieved2ppgain.',
        distinction='Mean->max fixed scalar pooling failed and is stopped. This candidate retains ordered within-region spatial information before the learned embedding, rather than changing a scalar maximum, loss, threshold or clock. It does not reproduce SPP-net or replace YOLO/ByteTrack.',
        literature='He et al.,Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,TPAMI2015,sectionII-B https://arxiv.org/html/1406.4729v4 : local spatial bins retain spatial information. Published classification/detection gains do not predict a FlyVis tracking gain.',
        counterargument='The lattice may resolve4bins yet contain no identity evidence. Neighboring/overlapping regions may share their content. Extra dimensions can overfit and repeat geometry; any full prototype needs equal-parameter repeated-mean, stockC, R1 and geometry controls,3seeds and actual assignment effects. Known outcomes are used only for this prerequisite, never as runtime inputs. Reusedtrain26 is exploratory.',
        next='Pass -> separately preregister and budget four-bin feature extraction, old-response parity, equal-capacity readout controls and assignment gate. No automatic HOTA or training admission from geometric coverage. Fail/inconclusive -> preserve this exact criterion; do not relax bin occupancy or choose a favorable subset.',
        limits='CPU180s+20,hidden independent controller, noGPU/training/dev/HOTA/75/download. Sunday individual deadline is smaller than authorMonday26Oct07:40.',sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Four-bin retinal coverage and reachable assignment inventory',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Пространственное считывание: доступность четырёх частей реальной рамки',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question=p['question'],hypothesis='Рабочая гипотеза: пространственный рисунок внутри области может содержать сведения, отброшенные одним средним. Геометрическая доступность этих сведений и их полезность проверяются отдельно.',expected=p['action'],acceptance=[dict(criterion=p['acceptance'],met=None)],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Мишень, равный контроль, критерий и бюджет до измерения')],obtained='',interpretation='',decision='',cannot_claim=p['counterargument']))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка доступности пространственного рисунка внутри рамки','*Действие.* '+p['action']+'\n\n*Мишень и критерий.* '+p['acceptance']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Источник.* '+p['literature']+'\n\n*Бюджет.* '+p['limits']+' Save '+p['save_at']+' / hard '+p['hard_stop']+'. Источник runs/z13_spatial_bins_gate_20261025/protocol.json.')


def worker():
    from tracking.z13_region_assignment import regions
    from tracking.z13_readout import to_eye
    p=read(RUN/'protocol.json');verify_sources(p);ns=original.namespace();queries,_=ns['queries_and_crops']();geom=ns['geometry']();records={}
    lr=ROOT/'runs/z13_region_assignment_20261025/labels';manifest=read(lr/'report.json')['video_sha256']
    for i,(video,items) in enumerate(queries.items()):
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original180s spatial inventory deadline')
        path=lr/'videos'/(video+'.json');assert sha(path)==manifest[video];labels={r['uid']:r for r in read(path)['rows']}
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz') as z:d={k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
        cases=[]
        for q in items:
            l=q['label'];r=labels[q['uid']];assert r['video']==video and r['frame']==q['frame'] and r['kind']==l['kind']
            ix=(d['frame_idx']==q['frame'])&(d['conf']>=.25);boxes=d['xyxy'][ix]
            choices,masks,_=regions(geom,l['crop'],l['pred'],boxes,d['conf'][ix],d['cls'][ix],r['regions']['track_class'])
            assert choices==r['regions']['choices'][1:]
            good=[];bad=[];occupancy=[]
            for j,mask,o in zip(choices,masks,r['arms']['regions'][1:]):
                if j<0:occupancy.append(None);continue
                center=np.asarray(to_eye((boxes[j,:2]+boxes[j,2:])/2,l['crop'],geom.eye_size))
                bins=(geom.points[:,0]>=center[0]).astype(int)+2*(geom.points[:,1]>=center[1]).astype(int)
                count=np.bincount(bins[mask],minlength=4);resolved=bool((count>0).all());occupancy.append(count.tolist())
                if resolved:
                    if o['beneficial']:good.append(j)
                    if not o['own_after'] or o['harmful']:bad.append(j)
            cases.append(dict(uid=q['uid'],kind=l['kind'],beneficial_any=any(o['beneficial'] for o in r['arms']['regions'][1:]),four_bin_beneficial_and_incorrect=bool(good and bad),quadrant_counts=occupancy))
        write(RUN/'videos'/(video+'.json'),dict(video=video,cases=cases,old_choices_exact=True))
        records[video]=dict(rows=len(cases),known=sum(x['kind']>=0 for x in cases),beneficial_any=sum(x['kind']>=0 and x['beneficial_any'] for x in cases),four_bin_reachable=sum(x['kind']>=0 and x['four_bin_beneficial_and_incorrect'] for x in cases))
        write(RUN/'status.json',dict(state='running',stage='Геометрия четырёх частей, без вычислений мухи',done=i+1,total=130,variant=video,updated=now().isoformat()))
    from scripts.z13_continue import paired
    values={v:records[v]['four_bin_reachable']/records[v]['known'] for v in p['evaluate26']};effect=paired(values,{v:0. for v in values})
    state='admit' if effect['ci95'][0]>.02 else ('insufficient_exposure' if effect['ci95'][1]<.02 else 'inconclusive')
    verify_sources(p);write(RUN/'report.json',dict(complete=True,admission_pass=state=='admit',gate_state=state,spatial_oracle_exposure=effect,per_video=records,total_queries=sum(r['rows'] for r in records.values()),known_queries=sum(r['known'] for r in records.values()),video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in records},hota_measured=False,claim='Only a necessary geometric/oracle exposure condition; no model predictions or benefit measured.'))
    write(RUN/'status.json',dict(state='completed',stage=state,updated=now().isoformat()))


def finish():
    ps=WindowsProcesses();c=read(RUN/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'report.json');assert r['complete'] and r['total_queries']==37296 and r['known_queries']==26600
    values={}
    for v,h in r['video_sha256'].items():
        path=RUN/'videos'/(v+'.json');assert sha(path)==h;rows=read(path)['cases'];values[v]=sum(x['kind']>=0 and x['four_bin_beneficial_and_incorrect'] for x in rows)/sum(x['kind']>=0 for x in rows)
    from scripts.z13_continue import paired
    effect=paired({v:values[v] for v in p['evaluate26']},{v:0. for v in p['evaluate26']});assert effect==r['spatial_oracle_exposure']
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    card.update(state={'admit':'completed_positive','insufficient_exposure':'closed','inconclusive':'inconclusive'}[r['gate_state']],done=now().strftime('%Y-%m-%d'),summary='Геометрическая мишень: '+r['gate_state'],obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → spatial_oracle_exposure: '+str(effect)+';37296запросов,26600известных, прежние реальные области совпадают.',interpretation='Это только доля случаев с разрешимыми четырьмя частями рамки и оракульно достижимым полезным действием среди неверных кандидатов. Не измерены различимость признаков, точность или HOTA.',decision=p['next']+' Gate: '+r['gate_state'])
    card['acceptance'][0].update(met=bool(r['admission_pass']),evidence=(RUN/'report.json').relative_to(ROOT).as_posix());write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,gate_state=r['gate_state']))
    journal('итог доступности пространственных частей','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    original.namespace()['handoff']('Завершён z13_spatial_bins_gate_v1. '+card['obtained']+' '+card['decision']+' Это не улучшение, а геометрическая предпосылка. Clock/semantic отрицательны; однолистовая архитектура105узлов95стрелок просмотрена, опубликована61e7e502. Новый готовый исполнитель следующей научной ступени отсутствует; выбрать обоснованный внутренний узел, без перебора потерь/порогов. Всё скрыто;57чужихграфправок/.gitignore сохранять.')
    publish()


def publish():
    code=inspect.getsource(template.publish).replace('Measure duplicate-time exposure before a FlyVis state intervention','Check spatial-bin exposure before a structured FlyVis readout')
    ns=dict(globals());exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']()


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
    elif '--publish' in sys.argv:publish()
    else:raise RuntimeError('Explicit mode required')
