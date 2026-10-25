"""Bounded cause-to-action gate for the 895 unsuccessful FlyVis proposals."""
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import os
import sys
import inspect
import hashlib
import subprocess
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_continue import paired
RUN = ROOT/'runs/z13_proposal_geometry_20261025'
OLD = ROOT/'runs/z13_assignment_effect_20261024/recovery1'
IDENT = 'z13_proposal_geometry_v1'
NEW_ARMS = ('null', 'centers_eye', 'boxes_eye', 'centers_frame', 'boxes_frame',
            'oracle_gt_center_eye', 'oracle_gt_box_eye')


def journal(title, text):
    from scripts import z13_candidate_reserve_20261021 as m
    saved = m.IDENT
    try:
        m.IDENT = IDENT
        m.journal(title, text)
    finally:
        m.IDENT = saved


def source():
    from scripts.z13_assignment_effect_20261024 import replay_video
    s = inspect.getsource(replay_video)
    anchor = "event['alternatives'][arm]=alts"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + '''
                        generated, meta = new_geometry(q,t,detections,costs[ri],self.get_dists,R[f])
                        event['geometry_meta']=meta
                        for arm,candidates in generated.items():
                            alts=[]
                            for candidate in candidates:
                                altered=costs.copy();altered[ri]=candidate
                                alts.append(assignment_vector(altered,self.args.match_thresh))
                            assert np.array_equal(alts[0],base)
                            event['alternatives'][arm]=alts
                        event['context']=dict(cost_matrix=costs.tolist(),row=ri,baseline_assignment=base.tolist(),track_ids=ids,match_threshold=float(self.args.match_thresh))''')
    anchor = "for arm,alts in event['alternatives'].items():"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "rr['geometry_meta']=event['geometry_meta'];rr['context']=dict(event['context'],high_identities=high,raw_identities=labels,causal_identities=[str(x[0]) for x in event['causal']])\n                " + anchor)
    anchor = "for j,outcome in enumerate(rr['arms'][arm]):"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "if arm not in centers[q['uid']]:continue\n                    " + anchor)
    return s


def summarize(per_video):
    p = read(RUN/'protocol.json')
    labels = read(ROOT/'runs/z13_refusal_causes_20261025/rows.json')
    known_n = Counter(r['video'] for r in labels if r['arm']=='D' and r['kind']>=0)
    values = {a: {} for a in (*NEW_ARMS, 'Dpeak3')}
    partition = Counter()
    detection_partition = Counter()
    cross = Counter()
    total = no_proposal = 0
    for video, data in per_video.items():
        tallies = Counter()
        for r in data['rows']:
            total += 1
            useful = {a:any(x['beneficial'] for x in r['arms'][a]) for a in NEW_ARMS}
            useful['Dpeak3'] = any(x['beneficial'] for x in r['arms']['D'])
            for a, value in useful.items():
                tallies[a] += value
            if useful['Dpeak3']:
                continue
            no_proposal += 1
            meta, ctx = r['geometry_meta'], r['context']
            identity = ctx['causal_identities'][ctx['row']]
            correct = [j for j,x in enumerate(ctx['high_identities']) if x==identity]
            inside = {x['index'] for x in meta['proposals'] if x['inside_eye']}
            if correct:
                reason = 'correct_high_in_eye' if any(j in inside for j in correct) else 'correct_high_outside_eye'
            elif identity in ctx['raw_identities']:
                reason = 'correct_detection_only_below_high_threshold'
            elif meta['oracle_box'] is not None:
                reason = 'annotated_object_without_labelled_raw_detection'
            else:
                reason = 'identity_not_annotated_here_not_proven_absent'
            detection_partition[reason] += 1
            # Fixed descriptive hierarchy, not an attribution to neural weights.
            order = ('null','centers_eye','boxes_eye','oracle_gt_center_eye',
                     'oracle_gt_box_eye','centers_frame','boxes_frame')
            action = next((a for a in order if useful[a]), 'none_of_tested_actions')
            partition[action] += 1
            cross[reason+';'+action] += 1
        for a in values:
            values[a][video] = tallies[a] / known_n[video]
    assert total==1001 and no_proposal==895
    zero = {v:0. for v in p['videos']}
    contrasts = {a+'-baseline':paired(values[a],zero) for a in NEW_ARMS}
    contrasts['centers_eye-Dpeak3'] = paired(values['centers_eye'],values['Dpeak3'])
    contrasts['boxes_eye-centers_eye'] = paired(values['boxes_eye'],values['centers_eye'])
    contrasts['boxes_frame-boxes_eye'] = paired(values['boxes_frame'],values['boxes_eye'])
    admit_center = contrasts['centers_eye-baseline']['mean_per_video']>=.02 and contrasts['centers_eye-baseline']['ci95'][0]>0 and contrasts['centers_eye-Dpeak3']['ci95'][0]>0
    admit_box = contrasts['boxes_eye-baseline']['mean_per_video']>=.02 and contrasts['boxes_eye-baseline']['ci95'][0]>0 and contrasts['boxes_eye-centers_eye']['ci95'][0]>0
    return dict(complete=True, diagnostic_only=True,not_hota=True, incorrect_baseline_rows=total,
        no_current_D_candidate_rows=no_proposal, remaining895_detection_partition=dict(detection_partition),
        remaining895_first_helpful_action=dict(partition), detection_action_cross=dict(cross),
        per_video=values,paired=contrasts, admit_center_prototype=admit_center,admit_full_box_prototype=admit_box,
        limits=p['limits'])


def prepare():
    assert not RUN.exists(), 'Preserve existing run'
    old = read(OLD/'protocol.json')
    diagnosis = read(ROOT/'runs/z13_refusal_causes_20261025/diagnosis.json')
    assert diagnosis['D_no_current_candidate_can_help']==895
    rows = read(ROOT/'runs/z13_refusal_causes_20261025/rows.json')
    wanted = [r['uid'] for r in rows if r['arm']=='D' and r['kind']>=0 and not r['own_before']]
    assert len(wanted)==len(set(wanted))==1001
    start=now();authorization=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    stop=min(start+timedelta(seconds=600),datetime.fromisoformat(authorization['save_at']))
    names=['scripts/z13_proposal_geometry_20261025.py','tracking/z13_proposal_geometry.py',
        'tests/test_z13_proposal_geometry.py','scripts/z13_assignment_effect_20261024.py',
        'scripts/z13_bounded_recovery_controller_20261022.py','scripts/z13_continue.py',
        'scripts/association_log.py','scripts/z13_data.py','scripts/freeze_bench.py',
        'tracking/xywh_bytetrack.py','tracking/offline_association.py',
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
        '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py',
        '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
        'runs/z13_refusal_causes_20261025/diagnosis.json','runs/z13_refusal_causes_20261025/rows.json',
        'research_flyvis_missing_proposals_2026-10-25/СВОДКА.md']
    for v in old['evaluate26']:
        names += [f'runs/train_cache/{v}/detections/raw_video.npz',f'runs/fly_signatures/{v}/signatures.npz',
            f'runs/train_cache/{v}/alignment.json',f'data/bdd100k/labels/box_track_20/train/{v}.json',
            (OLD/'videos'/(v+'.json')).relative_to(ROOT).as_posix()]
    p=dict(id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',device='cpu',budget_seconds=600,
        save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),
        stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),videos=old['evaluate26'],wanted_uids=wanted,
        action='Prepare one target-conditioned proposal readout: choose whether its output should generate a new center or full XYWH box. This single gate determines that action space before any learner. User25Oct explicitly authorized trying the literature-derived variants and a final outcome report.',
        interventions='Keep original D/C/R1 candidate actions. Add unchanged/null, centers of every actual same-class high detection inside the original crop, full XYWH of those same detections, and whole-frame counterparts. Every alternative recomputes the unchanged tracker cost formula and complete assignment after replacing only this row. Explicit annotation-only GT-center/GT-box arms are representative feasibility probes, not continuous maxima, causal proof or runtime actions. No new detection/threshold/score/ID.',
        technical='All26raw baseline outputs bit-exact. All1001original incorrect known rows and original C/D/R1 semantic outcomes exact. The remaining895 match old no-useful-D grouping exactly. Original first alternative equals baseline in every arm, full mean restored including on exceptions, hashes immutable. First2videos conservative2*mean*24+45 must fit original600s. Completed videos not repeated.',
        criteria='Center prototype only if observable-eye center oracle mean>=2pp of all5822known rows, lower95%videoCI>0 and increment over original D-threepeak oracle lower>0. Full-box prototype only if observable-eye full-box oracle mean>=2pp, lower>0 and increment over observable-eye center oracle lower>0. All106original recoverable and895remaining rows retained. NULL-only or GT-only reserve does not admit a box/center learner. Outside-eye reserve requires its own observed-input protocol, never immediate full-frame training. These are development gates, not a scientific effect threshold.',
        first_helpful_order=['null','centers_eye','boxes_eye','oracle_gt_center_eye','oracle_gt_box_eye','centers_frame','boxes_frame','none_of_tested_actions'],
        counterargument='Oracle selection uses outcome labels, so it is an optimistic geometry-only diagnostic independent of FlyVis. Actual detection proposals and annotation boxes do not exhaust continuous actions. One-step baseline states omit simultaneous corrections/feedback/low-score association. Missing annotation is not physical absence. Correct high detection may still be unreachable because of competition; this is not all assigned to representation.',
        next='If a geometry gate passes, register one learned FlyVis internal proposal/box readout with equal C/R1/geometry controls, fit-only settings and original D reproduction; do not repeat failed size-regression or peak rankers. If both fail, preserve counts and stop these proposed center/full-box implementations; choose a different action with evidence rather than tune gates.',
        literature='KeepTrack https://arxiv.org/html/2103.16556v2 AppendixA.1 Table9 andB.2; GlobalTrack https://arxiv.org/html/1912.08531 Table1; ATOM https://arxiv.org/html/1811.07628 section4.2; SiamFC++ https://arxiv.org/html/1911.06188 section3. Reduced action-space diagnostic, not a reproduction of any paper.',
        limits='CPU600s+20 independent stop; no GPU/core training/downloads/dev/HOTA/heldout75. Reused train26 and sparse labels; original detector trained on these train videos. Never a detector generalization result. Final report distinguishes admitted action, learned result, and stand HOTA. All other branches/results preserved.',
        executed_source_sha256=hashlib.sha256(source().encode()).hexdigest(),sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Причины895: центр, рамка и доступные детекции',done=0,total=26,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Предложения FlyVis: различение ограничения центра, рамки и доступных назначений',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],question='Какое новое доступное действие даёт дополнительный запас относительно нынешних трёх центров?',hypothesis='Рабочие версии: пропущен полезный центр, недостаточна геометрия рамки, отсутствует допустимая правильная детекция либо мешает совместное назначение. Один диагностический опыт выбирает выход будущего внутреннего считывания.',expected=p['interventions'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['criteria'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Действие, причины и критерии до расчёта')]))
    write(cp,cat)
    journal('постановка проверки доступного действия после поиска литературы','*Действие.* '+p['action']+'\n\n*Критерии до расчёта.* '+p['criteria']+'\n\n*Проверка.* '+p['technical']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Источники.* '+p['literature']+'\n\nCPU600с+20с, скрытый независимый контроллер, авторский срок26.10 07:40МСК. Прежняя серия отказа не продолжается: это одна проверка выбора геометрического выхода нового действия; дальнейшие диагностические серии для него не допускаются без причины.')


def worker():
    from scripts import z13_assignment_effect_20261024 as original
    from tracking.z13_proposal_geometry import geometry_actions
    p=read(RUN/'protocol.json');verify_sources(p)
    queries,centers=original.load_queries(read(OLD/'protocol.json'));wanted=set(p['wanted_uids'])
    def new_geometry(q,track,detections,cost,get_dists,objects):
        hit=[np.asarray(b) for identity,_,b in objects if str(identity)==q['identity']]
        return geometry_actions(track,detections,cost,get_dists,q['label']['crop'],hit[0] if hit else None)
    s=source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_source_sha256']
    ns={**vars(original),'RUN':RUN,'new_geometry':new_geometry,'T0':time.monotonic()}
    exec(compile(s,__file__,'exec'),ns)
    data={};times=[]
    for i,v in enumerate(p['videos']):
        path=RUN/'videos'/(v+'.json');assert not path.exists()
        start=time.monotonic();qs=[q for q in queries[v] if q['uid'] in wanted]
        rows,checks=ns['replay_video'](v,qs,centers,p)
        old={r['uid']:r for r in read(OLD/'videos'/(v+'.json'))['rows']}
        for r in rows:
            expected=old[r['uid']]
            assert {k:r[k] for k in expected if k!='arms'}=={k:expected[k] for k in expected if k!='arms'}
            assert all(r['arms'][a]==expected['arms'][a] for a in ('D','C','R1'))
        data[v]=dict(rows=rows,checks=checks);write(path,data[v]);times.append(time.monotonic()-start)
        write(RUN/'status.json',dict(state='running',stage='Проверка геометрии и назначения на прежних данных',variant=v,done=i+1,total=26,updated=now().isoformat(),last_video_seconds=times[-1]))
        if i==1:
            remaining=(datetime.fromisoformat(p['save_at'])-now()).total_seconds();estimate=2*np.mean(times)*24+45
            write(RUN/'price.json',dict(remaining_seconds=remaining,conservative_required_seconds=estimate,passed=bool(estimate<remaining)))
            assert estimate<remaining,'Original600s price gate failed'
    verify_sources(p);r=summarize(data);r['video_sha256']={v:sha(RUN/'videos'/(v+'.json')) for v in p['videos']};write(RUN/'report.json',r)
    write(RUN/'status.json',dict(state='completed',stage='Геометрия и причины895 сохранены; не HOTA',done=26,total=26,updated=now().isoformat()))


def finish():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p=read(RUN/'protocol.json');e=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items());verify_sources(p)
    r=read(RUN/'report.json');data={}
    for v,h in r['video_sha256'].items():
        path=RUN/'videos'/(v+'.json');assert sha(path)==h;data[v]=read(path);assert data[v]['checks']['raw_replay_exact']
    check=summarize(data);assert all(r[k]==value for k,value in check.items())
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all26_original_replays_exact=True,all1001_semantic_rows_exact=True,partition895_and_paired_recomputed=True,report_sha256=sha(RUN/'report.json')))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);c=next(x for x in cat['experiments'] if x['id']==IDENT)
    passed=r['admit_center_prototype'] or r['admit_full_box_prototype']
    c.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'),
        obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → remaining895_detection_partition/remaining895_first_helpful_action/paired: '+str({k:r[k] for k in ('remaining895_detection_partition','remaining895_first_helpful_action','paired')}),
        interpretation='Диагностическая достижимость действий с идеальным выбором, без преимущества какого-либо представления. Правильная детекция/центр/рамка и полезное назначение различаются. Не причинное доказательство неисправности ядра и не HOTA.',
        decision=p['next']+' Допуск центра: '+str(r['admit_center_prototype'])+'; полной рамки: '+str(r['admit_full_box_prototype'])+'.')
    c['acceptance'][0].update(met=True,evidence=(RUN/'independent_audit.json').relative_to(ROOT).as_posix())
    c['acceptance'][1].update(met=passed,evidence=(RUN/'report.json').relative_to(ROOT).as_posix())
    write(cp,cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,action_gate_passed=passed,report_sha256=sha(RUN/'report.json')))
    journal('итог выбора геометрического действия','*Получено.* '+c['obtained']+'\n\n*Вывод.* '+c['interpretation']+'\n\n*Решение.* '+c['decision']+'\n\nHOTA не измерялась, процент к стенду и доля потолка обученного узла не определены. Геометрический оракул не добавляется в лучшие результаты FlyVis.')
    handoff=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    handoff.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен. '+c['decision']+' Первичные причины/числа: runs/z13_proposal_geometry_20261025/report.json. Только проверка допустимости действия, не рабочая политика. Проверить/записать итог и выбрать одну реализацию по этому правилу, без повторения проваленных региональных потерь. Автор просит после окончания всех тестов сообщить что сработало/нет. До26.10 07:40МСК, скрытые процессы,57чужихправок/.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'+handoff.read_text(encoding='utf-8'),encoding='utf-8')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:
            write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
    else:raise RuntimeError('Explicit mode required')
