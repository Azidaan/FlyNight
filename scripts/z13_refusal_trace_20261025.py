"""Instrument the frozen exact replay to explain selected and missed actions."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import inspect
import hashlib
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha
from scripts.z13_refusal_causes_20261025 import journal, IDENT

RUN=ROOT/'runs/z13_refusal_causes_20261025/trace'
FIRST=RUN.parent
OLD=ROOT/'runs/z13_assignment_effect_20261024/recovery1'
MSK=timezone(timedelta(hours=3))


def now():
    return datetime.now(MSK)


def instrumented_source():
    from scripts.z13_assignment_effect_20261024 import replay_video
    source=inspect.getsource(replay_video)
    old='original=t.mean.copy()'
    assert source.count(old)==1
    source=source.replace(old,old+"\n                    event['trace']=dict(baseline_assignment=base.tolist(),query_row=ri,track_ids=ids,causal_identities=[str(x[0]) for x in event['causal']],query_mean=t.mean[:4].tolist(),baseline_cost_row=costs[ri].tolist(),high_detections=[dict(xyxy=d.xyxy.tolist(),score=float(d.score),cls=int(d.cls)) for d in detections],candidate_centers={},candidate_cost_rows={})")
    old="alts=[]\n                            for center in centers[q['uid']][arm]:"
    assert source.count(old)==1
    source=source.replace(old,"alts=[]\n                            event['trace']['candidate_centers'][arm]=[];event['trace']['candidate_cost_rows'][arm]=[]\n                            for center in centers[q['uid']][arm]:")
    old='alts.append(assignment_vector(altered,self.args.match_thresh))'
    assert source.count(old)==1
    source=source.replace(old,old+"\n                                event['trace']['candidate_centers'][arm].append(center.tolist());event['trace']['candidate_cost_rows'][arm].append(altered[ri].tolist())")
    old="for arm,alts in event['alternatives'].items():"
    assert source.count(old)==1
    source=source.replace(old,"rr['trace']={**event['trace'],'match_threshold':float(tr.args.match_thresh),'high_identities':high,'raw_detections':[dict(xyxy=b.tolist(),score=float(s),cls=int(c),identity=a) for b,s,c,a in zip(boxes,scores,cls,labels)],'candidate_assignments':{a:[x.tolist() for x in xs] for a,xs in event['alternatives'].items()}}\n                "+old)
    return source


def prepare():
    assert not RUN.exists()
    e=read(FIRST/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended']
    first=read(FIRST/'report.json');assert first['complete'] and first['gate_unchanged']
    assert sha(FIRST/'rows.json')==first['rows_sha256']
    rows=[r for r in read(FIRST/'rows.json') if r['arm']=='D' and r['kind']>=0]
    wanted={r['uid'] for r in rows if r['selected'] or (r['outcome']=='beneficial' and not r['selected'])}
    oldp=read(OLD/'protocol.json')
    for v in oldp['evaluate26']:
        for r in read(OLD/'videos'/(v+'.json'))['rows']:
            if any(x['old_center_useful'] and x['harmful'] for x in r['arms']['D']):
                wanted.add(r['uid'])
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_refusal_trace_console.ps1',
        'scripts/z13_bounded_stage_20261022.py','scripts/z13_assignment_effect_20261024.py',
        (FIRST/'protocol.json').relative_to(ROOT).as_posix(),(FIRST/'report.json').relative_to(ROOT).as_posix(),
        (FIRST/'rows.json').relative_to(ROOT).as_posix(),(FIRST/'controller_result.json').relative_to(ROOT).as_posix()]
    p=dict(id=IDENT,device='cpu',created=now().isoformat(),budget_seconds=600,
        save_at=(now()+timedelta(seconds=600)).isoformat(),hard_stop=(now()+timedelta(seconds=620)).isoformat(),
        authorization='Second diagnostic in author-requested investigation of refusal causes; no new model or policy.',
        wanted_uids=sorted(wanted),videos=oldp['evaluate26'],
        scope='All D fixed-selected known queries, all D useful top1 rejected, and all old-center-useful harmful candidates. Capture full Hungarian assignment, relevant cost row, detection identities and scores. GT labels outcomes only. Replay all26baseline videos without state intervention, verify bit-exact outputs and old semantic outcome rows.',
        technical='Instrumented code adds trace fields only; old replay function source and executed source hash retained. Every selected UID accounted for exactly; all semantic outcome fields exactly equal saved original; every baseline video raw output exact. No new numeric tolerance.',
        decision='Distinguish absent high-score correct detection, competing assignment, harmful unnecessary change, or residual cost/geometry mismatch. These are exact one-step mechanisms in baseline states, not a HOTA improvement or proof of representation inadequacy. Select one readout repair after this trace; no unbounded diagnostic series.',
        sha256={**oldp['sha256'],**{n:sha(ROOT/n) for n in names}},
        executed_source_sha256=hashlib.sha256(instrumented_source().encode()).hexdigest())
    write(RUN/'protocol.json',p);write(RUN/'executed_replay_source.txt',instrumented_source())
    write(RUN/'status.json',dict(state='ready',stage='Точное объяснение выбранных вмешательств и пропусков',done=0,total=26,updated=now().isoformat()))
    journal('вторая проверка причин: назначения и конкурирующие рамки',
        '*Постановка.* '+p['scope']+'\n\n*Проверка.* '+p['technical']+'\n\n*Решение.* '+p['decision']+
        '\n\nИсточник: runs/z13_refusal_causes_20261025/trace/protocol.json;600сCPUснезависимымконтроллером. '
        'Исходные завершённые вычисления мухи не повторяются: новый проход трекера собирает ранее отсутствовавшие детекции/стоимости/назначения на кэшах. Без GPU,обучения,видеодекодирования или новойHOTA.')


def mechanism(r, arm, rank):
    trace=r['trace'];row=trace['query_row'];identity=trace['causal_identities'][row]
    before=trace['baseline_assignment'][row]
    alt=trace['candidate_assignments'][arm][rank];after=alt[row]
    correct=[j for j,x in enumerate(trace['high_identities']) if x==identity]
    raw=[d for d in trace['raw_detections'] if d['identity']==identity]
    taken=[trace['track_ids'][i] for i,j in enumerate(alt) if i!=row and j in correct]
    if not correct:
        reason='correct_detection_only_below_high_threshold' if raw else 'no_correct_labelled_detection'
    elif after in correct:
        reason='correct_detection_assigned'
    elif taken:
        reason='correct_detection_assigned_to_competing_track'
    elif min(trace['candidate_cost_rows'][arm][rank][j] for j in correct)>trace['match_threshold']:
        reason='correct_detection_cost_above_match_threshold'
    else:
        reason='global_assignment_prefers_other_or_unmatched'
    return dict(reason=reason,baseline_detection=before,alternative_detection=after,
        correct_high_detection_indices=correct,competing_track_ids=taken,
        old_semantic_outcome=r['arms'][arm][rank])


def worker():
    from scripts import z13_assignment_effect_20261024 as O
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p)
    source=instrumented_source();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    O.RUN=RUN
    namespace=dict(vars(O));exec(compile(source,str(Path(__file__)), 'exec'),namespace)
    replay=namespace['replay_video'];oldp=read(OLD/'protocol.json')
    queries,centers=O.load_queries(oldp);wanted=set(p['wanted_uids']);captured=[]
    for vi,video in enumerate(p['videos']):
        selected=[q for q in queries[video] if q['uid'] in wanted]
        rows,checks=replay(video,selected,centers,p)
        reference={r['uid']:r for r in read(OLD/'videos'/(video+'.json'))['rows']}
        for r in rows:
            assert {k:v for k,v in r.items() if k!='trace'}==reference[r['uid']]
        write(RUN/'videos'/(video+'.json'),dict(rows=rows,checks=checks))
        captured.extend(rows)
        write(RUN/'status.json',dict(state='running',stage='Назначения, рамки и точное повторение исходного выхода',variant=video,done=vi+1,total=26,updated=now().isoformat()))
    assert {r['uid'] for r in captured}==wanted and len(captured)==len(wanted)
    first={r['uid']:r for r in read(FIRST/'rows.json') if r['arm']=='D'}
    selected=[];missed=[];oldharm=[]
    for r in captured:
        gate=first[r['uid']]
        if gate['selected']:
            selected.append(dict(uid=r['uid'],video=r['video'],frame=r['frame'],tid=r['tid'],gate=gate,**mechanism(r,'D',0)))
        elif gate['outcome']=='beneficial':
            missed.append(dict(uid=r['uid'],video=r['video'],frame=r['frame'],tid=r['tid'],gate=gate,**mechanism(r,'D',0)))
        for rank,c in enumerate(r['arms']['D']):
            if c['old_center_useful'] and c['harmful']:
                oldharm.append(dict(uid=r['uid'],video=r['video'],frame=r['frame'],tid=r['tid'],rank=rank,**mechanism(r,'D',rank)))
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,diagnostic_only=True,not_hota=True,
        captured_queries=len(captured),baseline_videos=26,semantic_rows_exact=True,
        selected=selected,missed_beneficial_top1=missed,old_center_useful_harmful=oldharm,
        video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in p['videos']},limits=p['decision']))
    write(RUN/'status.json',dict(state='completed',stage='Причины назначений сохранены;не HOTA',done=26,total=26,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--worker' in sys.argv:
        try:
            worker()
        except Exception as error:
            write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()))
            raise
    else:
        raise ValueError('Choose --prepare or --worker')
