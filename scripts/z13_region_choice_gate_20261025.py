"""Cheap gate before a learned target-conditional detection-region readout."""
from pathlib import Path
from datetime import datetime,timedelta
import os
import sys
import time
import inspect
import hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_region_choice_gate_20261025'
OLD=ROOT/'runs/z13_assignment_effect_20261024/recovery1'
IDENT='z13_region_choice_gate_v1'
T0=time.monotonic()


def journal(title,text):
    from scripts import z13_localization_labels_20261023 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def replay_source():
    from scripts.z13_region_readout_20261025 import replay_source as base
    s=base();old='new_cost_rows(q,costs[ri],detections)'
    assert s.count(old)==1
    return s.replace(old,'new_cost_rows(q,costs[ri],detections,t.cls)')


def prepare():
    assert not RUN.exists()
    prior=read(ROOT/'runs/z13_presence_scope_20261025/finish_recorded.json');assert prior['complete'] and not prior['primary_pass']
    base=read(OLD/'protocol.json')
    names=['tracking/z13_region_choice.py','tests/test_z13_region_choice.py','scripts/z13_region_choice_gate_20261025.py','scripts/z13_region_choice_gate_console.ps1',
        'scripts/z13_region_readout_20261025.py','tracking/z13_region_readout.py','runs/z13_local_peak_20261021/geometry.json',
        'runs/z13_presence_scope_20261025/report.json','runs/z13_presence_scope_20261025/independent_audit.json',
        (OLD/'report.json').relative_to(ROOT).as_posix()]
    names += [(OLD/'videos'/(v+'.json')).relative_to(ROOT).as_posix() for v in base['evaluate26']]
    stop=now()+timedelta(seconds=600)
    assert stop<datetime.fromisoformat(read(ROOT/'runs/z13_weekend_20261024/authorization.json')['save_at'])
    p=dict(id=IDENT,created=now().isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=600,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        videos=base['evaluate26'],evaluate26=base['evaluate26'],
        action='Replace coordinate-only output by a learned target-conditional categorical choice among actual high-score same-class detection regions in the eye and NULL. Before feature extraction/training, test whether this expressible output has additional useful assignment reserve over the failed three-peak center action.',
        mechanism='For each unchanged tracker row: original cost vector, NULL vector all1, and each visible same-class region with one-hot probability1. With the unchanged cost=1-det_conf*prob interface, chosen cost=1-det_conf, every other cost1. Solve the whole original assignment. Extreme distributions are only an optimistic diagnostic of that interface. No true identity, target box or query outcome enters action generation.',
        support='Same eye geometry as the tested region readout: detection center in square and within0.6 lattice spacing of a hex column; rectangle contains at least one column. Same detector class as the existing track. No access to detections outside the observed eye.',
        comparison='All8326 natural query rows on26 reused videos. Known5822 only for scored effects; unknown2504 retained. A beneficial action must correct its previously incorrect row and increase the total correct eligible assignments. Report unchanged, NULL, each real region, and the annotation-only any-beneficial oracle. Pair with original D three-peak oracle on the exact same UID/labels and known denominators. No learned-competitor comparison is claimed by an oracle.',
        technical='26 original tracker outputs float32-exact; predictions, causal identities, labels and UIDs identical to saved one-step replay. First alternative equals unchanged assignment exactly. Do not alter original state or costs. First2 videos:2*mean_elapsed*24+60 must fit remaining600s budget. Completed videos never repeated.',
        primary='Mean-video any-beneficial reserve>=0.02 and its lower95%paired-videoCI>0, plus categorical-reserve minus Dthreepeak-reserve lower95%paired-videoCI>0. Failing gate stops this specific output interface before learned feature extraction. Passing only warrants registration of a matched learned region head; it is not FlyVis benefit or HOTA.',
        counterargument='An oracle can exploit detections without a fly; the real readout may not distinguish them. One-row shadow choices omit feedback, simultaneous interventions and subsequent association stages. One-hot probabilities are optimistic extremes. A larger reserve is necessary for this proposed implementation, not sufficient for training success or scientific novelty.',
        difference='The old detection3 gate measured center accuracy. The failed region_readout used a fixed mean of an existing response map. The new proposed learner would compare raw target-conditioned activity at actual detection regions, with competition and NULL. This gate tests its action space only, before building that learner; it does not rescue failed fixed pooling or repeat a three-peak ranker.',
        literature='SiamMOT https://arxiv.org/html/2105.11595v1 sections3,5: region-based target-conditional motion for matching; our categorical interface and reserve gate are adaptations, not its published numbers.',
        next='Pass -> register raw-activity region learner with equal geometry-only/R1/C controls, exact baseline reproduction and fit-only hyperparameters before GPU extraction. Fail -> stop this interface; choose a different internal FlyVis node, no retrospective weakening of the2pp gate.',
        executed_source_sha256=hashlib.sha256(replay_source().encode()).hexdigest(),sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'executed_replay.py.txt',dict(source=replay_source()))
    write(RUN/'status.json',dict(state='ready',stage='Запас прямого выбора детекции считывателем',done=0,total=26,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: запас считывания выбора реальной области вместо центра',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Есть ли достаточный дополнительный запас полезных назначений у прямого выбора реальной области?',hypothesis='Даже идеального бинарного присутствия при прежнем ранжировании недостаточно; модель цели может нуждаться в другом выходе и конкуренции реальных областей.',expected=p['mechanism'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],summary='Оракульная проверка мишени до реализации обучаемого считывателя.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие и критерии записаны до расчёта')]))
    write(cp,cat)
    journal('постановка мишени обучаемого регионального выбора','*Действие.* '+p['action']+'\n\n*Механизм.* '+p['mechanism']+'\n\n*Критерий.* '+p['primary']+'\n\n*Отличие от испытанного.* '+p['difference']+'\n\n*Возражение.* '+p['counterargument']+'\n\nCPU600с+20с, один независимый контроллер и видимая консоль. '+p['literature'])


def worker():
    import numpy as np
    from scripts import z13_assignment_effect_20261024 as original
    from tracking.z13_readout import HexGeometry,to_eye
    from tracking.z13_region_readout import region_probabilities
    from tracking.z13_region_choice import categorical_rows
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p)
    geomdata=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(geomdata['points'],np.float32),geomdata['eye_size'])
    queries,centers=original.load_queries(read(OLD/'protocol.json'));metadata={}
    def new_cost_rows(q,old,detections,track_class):
        boxes=np.asarray([d.xyxy for d in detections],float).reshape(-1,4);projected=[];support=[]
        for d,box in zip(detections,boxes):
            a=to_eye(box[:2],q['label']['crop'],geomdata['eye_size']);b=to_eye(box[2:],q['label']['crop'],geomdata['eye_size']);c=(np.asarray(a)+b)/2
            projected.append([*a,*b]);support.append(bool(int(d.cls)==int(track_class) and np.all(c>=0) and np.all(c<=geomdata['eye_size']) and np.linalg.norm(geom.points[geom.nearest(c)]-c)<=.6*geom.spacing))
        _,valid=region_probabilities(np.zeros(len(geom.points)),geom.points,np.asarray(projected).reshape(-1,4),support)
        confidence=np.asarray([d.score for d in detections],float);rows,choices=categorical_rows(old,confidence,valid)
        metadata[q['uid']]=dict(choices=choices,supported=int(valid.sum()),high_detections=len(detections),track_class=int(track_class))
        return {'regions':rows}
    code=replay_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_source_sha256']
    ns={**vars(original),'RUN':RUN,'new_cost_rows':new_cost_rows,'T0':time.monotonic()};exec(compile(code,str(Path(__file__)),'exec'),ns)
    times=[];reserve={};prior={};tally=dict(rows=0,known=0,unknown=0,supported_regions=0)
    for i,v in enumerate(p['videos']):
        path=RUN/'videos'/(v+'.json');assert not path.exists()
        began=time.monotonic();rows,checks=ns['replay_video'](v,queries[v],centers,p)
        oldrows={r['uid']:r for r in read(OLD/'videos'/(v+'.json'))['rows']};known=[];oldknown=[]
        for r in rows:
            before=oldrows[r['uid']];assert all(before[k]==r[k] for k in ('video','label','kind','frame','tid'))
            r['regions']=metadata[r['uid']];assert len(r['arms']['regions'])==len(r['regions']['choices'])
            assert r['kind']<0 or not (r['arms']['regions'][0]['beneficial'] or r['arms']['regions'][0]['harmful'])
            tally['rows']+=1;tally['supported_regions']+=r['regions']['supported']
            if r['kind']>=0:known.append(r);oldknown.append(before);tally['known']+=1
            else:tally['unknown']+=1
        reserve[v]=float(np.mean([any(a['beneficial'] for a in r['arms']['regions']) for r in known]))
        prior[v]=float(np.mean([any(a['beneficial'] for a in r['arms']['D']) for r in oldknown]))
        write(path,dict(rows=rows,checks=checks,seconds=time.monotonic()-began));times.append(time.monotonic()-began)
        write(RUN/'status.json',dict(state='running',stage='Реальные области, NULL и полное сопоставление',variant=v,done=i+1,total=26,updated=now().isoformat()))
        if i==1:
            estimate=2*np.mean(times)*24+60;left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
            write(RUN/'price.json',dict(estimate=float(estimate),remaining=left,proceed=bool(estimate<left)));assert estimate<left,'Save first2; do not exceed original budget'
    assert tally['rows']==8326 and tally['known']==5822 and tally['unknown']==2504
    comp=dict(regions_baseline=paired(reserve,{v:0. for v in reserve}),regions_Dpeak3=paired(reserve,prior))
    passed=comp['regions_baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values())
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=passed,paired=comp,per_video=reserve,prior_per_video=prior,counts=tally,video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in p['videos']},hota_measured=False,limitations=p['counterargument']))
    write(RUN/'status.json',dict(state='completed',stage='Оракульный запас реальных областей проверен',done=26,total=26,updated=now().isoformat()))


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json')
    while not (RUN/'controller_result.json').exists():
        if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=60):raise TimeoutError('Missing controller')
        time.sleep(2)
    e=read(RUN/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    verify_sources(p);r=read(RUN/'report.json')
    import numpy as np
    from scripts.z13_continue import paired
    reserve={};prior={};count=0
    for v,h in r['video_sha256'].items():
        path=RUN/'videos'/(v+'.json');assert sha(path)==h;data=read(path);assert data['checks']['raw_replay_exact']
        rows=data['rows'];count+=len(rows);old={x['uid']:x for x in read(OLD/'videos'/(v+'.json'))['rows']}
        known=[x for x in rows if x['kind']>=0]
        reserve[v]=sum(any(a['beneficial'] for a in x['arms']['regions']) for x in known)/len(known)
        prior[v]=sum(any(a['beneficial'] for a in old[x['uid']]['arms']['D']) for x in known)/len(known)
    assert count==8326 and reserve==r['per_video'] and prior==r['prior_per_video']
    comp=dict(regions_baseline=paired(reserve,{v:0. for v in reserve}),regions_Dpeak3=paired(reserve,prior));assert comp==r['paired']
    passed=comp['regions_baseline']['mean_per_video']>=.02 and all(c['ci95'][0]>0 for c in comp.values());assert passed==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),raw_rows_sha_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    fields=dict(summary='Запас прямого выбора области '+('прошёл критерий подготовки обучаемого узла.' if passed else 'не прошёл критерий подготовки обучаемого узла.'),obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → '+str(comp)+'; '+str(r['counts']),interpretation='Это оракульная достижимость внутри одной строки назначения, одинаковая для любого представления. Польза FlyVis и HOTA не измерены.',decision=p['next']+' Измеренный итог критерия: '+str(passed),cannot_claim=p['counterargument'])
    head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Проверен полный оракульный запас реальных областей.'))
        card[k]=v
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    for i,met in enumerate((True,passed)):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'report.json')).relative_to(ROOT).as_posix())
    write(cp,cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed,report_sha256=sha(RUN/'report.json')))
    journal('итог мишени прямого выбора области','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA не измерялась, процент к стенду не определён. Доля потолка для обученного узла ещё не измерена. Оракул не входит в лучшие улучшения. Нового архитектурного узла нет: измерена мишень перед реализацией.')
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md';h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен. '+fields['obtained']+' '+fields['decision']+' Публикация runs/z13_region_choice_gate_20261025/publication.json. Старые результаты и57чужихправок сохранить; критерии не ослаблять. Авторизация26.10 07:40МСК остаётся.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf-8'),encoding='utf-8')
    # Same audited publication boundary; this module supplies its own exact files/run.
    from scripts import z13_presence_scope_20261025 as pub
    code=inspect.getsource(pub.publish).replace('tracking/z13_presence_scope.py','tracking/z13_region_choice.py').replace('tests/test_z13_presence_scope.py','tests/test_z13_region_choice.py').replace('scripts/z13_presence_scope_20261025.py','scripts/z13_region_choice_gate_20261025.py').replace('scripts/z13_presence_scope_console.ps1','scripts/z13_region_choice_gate_console.ps1').replace('Audit target-presence supervision before refusal training','Measure assignment reserve of categorical region readout')
    code=code.replace("names += [f.relative_to(ROOT).as_posix() for f in RUN.glob('*.json') if f.name!='publication.json']","names += [f.relative_to(ROOT).as_posix() for f in RUN.glob('*.json') if f.name!='publication.json']\n    names += [f.relative_to(ROOT).as_posix() for f in (RUN/'videos').glob('*.json')]")
    ns={**vars(pub),'RUN':RUN};exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    elif '--finish' in sys.argv:
        try:finish()
        except Exception as exc:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
    else:raise RuntimeError('Explicit mode required')
