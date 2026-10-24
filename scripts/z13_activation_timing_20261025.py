"""CPU ceiling for activating the FlyVis state/readout before a track is Lost.

Ground truth is an explicitly labelled diagnostic oracle, never a runtime input.
The real tracker is never changed: every shadow center is restored before update.
"""
from pathlib import Path
from datetime import timedelta, datetime
from collections import Counter
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now

RUN = ROOT / 'runs/z13_activation_timing_20261025'
IDENT = 'z13_activation_timing_v1'
OLD = ROOT / 'runs/z13_assignment_effect_20261024/recovery1'
W = ROOT / 'runs/z13_weekend_20261024'


def journal(title, text):
    from scripts import z13_val20_common_20261024 as source
    old = source.IDENT
    try:
        source.IDENT = IDENT
        source.journal(title, text)
    finally:
        source.IDENT = old


def correctness(assignment, identities, eligible, detection_identities):
    """The established first-stage semantic definition, including no detection."""
    return np.asarray([
        bool(ok) and ((j >= 0 and detection_identities[j] == identity)
                      if identity in detection_identities else j < 0)
        for j, identity, ok in zip(assignment, identities, eligible)
    ], bool)


def shadow_center(tracker, pool, detections, costs, row, center):
    from scripts.z13_assignment_effect_20261024 import assignment_vector
    original = pool[row].mean.copy()
    try:
        pool[row].mean[:2] = center
        altered = costs.copy()
        altered[row] = tracker.get_dists([pool[row]], detections)[0]
        return assignment_vector(altered, tracker.args.match_thresh)
    finally:
        pool[row].mean[:] = original


def prepare():
    assert not RUN.exists(), 'Never overwrite or launch a duplicate'
    assert read(OLD / 'controller_result.json')['all_owned_ended']
    videos = read(OLD / 'protocol.json')['evaluate26']
    assert len(set(videos)) == 26
    names = [Path(__file__).relative_to(ROOT).as_posix(),
             'scripts/z13_activation_timing_console.ps1',
             'tests/test_z13_activation_timing.py',
             'scripts/z13_bounded_stage_20261022.py',
             'scripts/z13_assignment_effect_20261024.py',
             'scripts/z13_data.py', 'scripts/association_log.py',
             'scripts/freeze_bench.py', 'tracking/xywh_bytetrack.py',
             'tracking/offline_association.py', 'tracking/z13_ridge_localizer.py',
             'tracking/z13_spatial_localizer.py',
             'tracking/foveated_eyes.py', 'tracking/z13_readout.py',
             'runs/z13_local_peak_20261021/geometry.json',
             '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
             '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py',
             '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py']
    for v in videos:
        names += [f'runs/train_cache/{v}/detections/raw_video.npz',
                  f'runs/train_cache/{v}/alignment.json',
                  f'runs/fly_signatures/{v}/signatures.npz',
                  f'data/bdd100k/labels/box_track_20/train/{v}.json']
    began = now()
    authorization = read(W / 'authorization.json')
    p = dict(id=IDENT, created=began.isoformat(), device='cpu', budget_seconds=900,
        save_at=min(began + timedelta(seconds=900), datetime.fromisoformat(authorization['save_at'])).isoformat(),
        hard_stop=min(began + timedelta(seconds=920), datetime.fromisoformat(authorization['hard_stop'])).isoformat(),
        videos=videos,
        action='Consider keeping the internal FlyVis state and readout active on currently Tracked targets, before the first incorrect association, instead of starting only after Lost. This stage tests the available target for that change, not a new runtime policy.',
        question='Does the Lost-only activation boundary exclude correctable first-stage association errors on still-Tracked objects?',
        hypothesis='A wrong match can keep a track in Tracked, so the current Lost-only readout is never called at that event. This is a structural hypothesis, not evidence that earlier FlyVis predictions will be accurate.',
        method='Replay all26 baseline videos exactly. At real annotated frames only, bind pool rows to the most recent observed annotated identity, freshness <=12 frames, with the same sparse-annotation gap exclusions as assignment_effect. Report all known rows by incoming Tracked/Lost state. For an initially incorrect row whose target has a high-score detection, translate only the predicted center to the annotated target-box center, preserving width/height and all other state. Recompute the whole unchanged Hungarian assignment with that single row replaced, then restore the original state before the real tracker update. For the primary gate, additionally require the target center inside the existing predicted eye crop and within0.6 lattice spacing of a retinal column, exactly the original field criterion. Out-of-field oracles are reported separately. Detection identities are assigned to all raw detections before the unchanged high-score filter, as in assignment_effect. No video, feature extraction, FlyVis inference, training or HOTA.',
        primary='For each video, count individually beneficial IN-FIELD center-oracle actions on incoming Tracked rows divided by ALL known pool rows (Tracked+Lost), using the same denominator for early and Lost-only ceilings. Mean added opportunity >=0.02 and paired-video95%lower>0 are both required before preparing an all-active FlyVis extraction. This is a conservative practical gate for this center-only action, not a universal criterion for timing mechanisms. Interacting oracle actions are not combined or claimed jointly attainable.',
        benefit='Same established one-step definition: own incorrect assignment becomes correct AND total correct eligible assignments strictly increases. Harm: own correct becomes incorrect OR total correct count decreases. If no high-score detection of the known identity exists, unmatched is correct for this first stage only. Such rows are reported but are not center-oracle candidates.',
        technical='All26 raw baseline outputs must match saved float32 rows exactly. Full pool costs unchanged after every shadow action. Identity binding uses past observations only; evaluation truth is never passed to tracker.update. End state after the real full update is reported separately, so first-stage miss is not presented as final loss. Sources and original thresholds unchanged. First two fixed videos measure cost; 2*mean_seconds*remaining24+60 must fit remaining900s, otherwise preserve partial results without a verdict.',
        counterargument='A perfect annotated center is unavailable at runtime, is not an upper bound for arbitrary box/identity modules, and can still fail IoU matching with unchanged dimensions. Sparse annotation and local identity freshness do not reveal every ID switch. Early activation may increase cost and erroneous interventions. A positive ceiling is necessary evidence for a prototype, not sufficient evidence of learnability or improved HOTA.',
        difference_from_tried='Earlier memory/temporal-candidate variants change observations inside already-Lost episodes; scalar-confidence and observed-contrast variants reorder the same already-Lost top1 proposals. This diagnostic asks about events entirely excluded from those episodes. It does not repeat their fitting or relax their failed criteria.',
        next='Only a passing ceiling permits a separately registered, cost-measured internal early-state/readout prototype with C/D/R1 and a matched no-FlyVis control. Otherwise stop this center-only early-activation action; do not launch full video extraction merely to remain busy.',
        literature=dict(url='https://openaccess.thecvf.com/content/CVPR2021/html/Shuai_SiamMOT_Siamese_Multi-Object_Tracking_CVPR_2021_paper.html',
            title='SiamMOT: Siamese Multi-Object Tracking',
            support='Primary abstract verifies instance-level motion modelling across consecutive frames for association. It motivates evaluating an earlier motion/readout interface, but supplies no evidence for this FlyVis-specific hypothesis or its numerical gains.'),
        limits='Reused26train and sparse annotations. First-stage baseline-state counterfactual diagnostic, not HOTA, deployed policy, causal proof about FlyVis representations, or tracking benchmark. No dev/20val tuning/75/downloads/GPU.',
        sha256={n:sha(ROOT / n) for n in sorted(set(names))})
    write(RUN / 'protocol.json', p)
    write(RUN / 'status.json', dict(state='ready', stage='Проверка момента включения: до потери или после', done=0, total=26, updated=now().isoformat()))
    cp = ROOT / 'experiments/project_history.json'
    cat = read(cp)
    assert not any(c['id'] == IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT, title='З13: потолок раннего включения считывания до состояния потери',
        area='flyvis', state='ready', started=now().strftime('%Y-%m-%d'), tags=['fly-problem-P8', 'SiamMOT'],
        question=p['question'], hypothesis=p['hypothesis'], expected=p['action'] + ' ' + p['method'],
        acceptance=[dict(criterion=p['technical'], met=None), dict(criterion=p['primary'], met=None)],
        summary='Короткая проверка мишени для иной границы работы внутреннего состояния; не новый обученный вариант.',
        obtained='', interpretation='', decision='', cannot_claim=p['counterargument'] + ' ' + p['limits'],
        metrics=[], commits=[], files=names[:3], evidence=[dict(kind='file', ref=(RUN / 'protocol.json').relative_to(ROOT).as_posix(), claim='Действие, критерии и цена записаны до расчёта')]))
    write(cp, cat)
    journal('постановка проверки момента включения до потери',
        '*Действие.* ' + p['action'] + '\n\n*Вопрос.* ' + p['question'] +
        '\n\n*Метод и критерии до исходов.* ' + p['method'] + ' ' + p['technical'] + ' ' + p['primary'] +
        '\n\n*Отличие от испытанного.* ' + p['difference_from_tried'] +
        '\n\n*Контраргумент.* ' + p['counterargument'] +
        '\n\nSiamMOT: ' + p['literature']['url'] + '. Переносится мотивация проверки, не численный результат. CPU до900с, независимый контроллер и отдельная консоль. Старые критерии не меняются.')


def replay(video, protocol):
    from scripts.z13_assignment_effect_20261024 import assignment_vector, action_outcome
    from scripts.z13_data import raw_index
    from scripts.association_log import label
    from scripts.output_filter_oracle import _iou
    from scripts.freeze_bench import XYWH_CONFIG
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from datasets.bdd100k_loader import CLASS_NAMES
    from tracking.foveated_eyes import crop_box
    from tracking.z13_readout import HexGeometry, to_eye
    began = time.monotonic()
    with np.load(ROOT / f'runs/train_cache/{video}/detections/raw_video.npz') as z:
        det = {k:z[k] for k in ('xyxy', 'conf', 'cls', 'frame_idx')}
    with np.load(ROOT / f'runs/fly_signatures/{video}/signatures.npz') as z:
        count, expected = int(z['frame_count']), z['outputs']
    truth = read(ROOT / f'data/bdd100k/labels/box_track_20/train/{video}.json')
    frames = [x['video_frame_index'] for x in read(ROOT / f'runs/train_cache/{video}/alignment.json')['matches']]
    R = raw_index(truth, frames, {'pedestrian','rider','car','bus','truck','train','motorcycle','bicycle'})
    ann = sorted(R)
    neighbors = {f:(ann[i-1] if i else None, ann[i+1] if i+1 < len(ann) else None) for i,f in enumerate(ann)}
    geometry = read(ROOT / 'runs/z13_local_peak_20261021/geometry.json')
    geom = HexGeometry(np.asarray(geometry['points'], np.float32), geometry['eye_size'])
    own, rows, counts, outputs = {}, [], Counter(), []
    ctx = dict(frame=-1)

    class Observe(XYWHByteTrack):
        def _first_association(self, pool, detections, activated, refind):
            f = ctx['frame']
            ctx['events'] = []
            if f not in R or not pool:
                return super()._first_association(pool, detections, activated, refind)
            objects = R[f]
            byid = {str(x[0]):x for x in objects}
            prev, nxt = neighbors[f]
            gaps = {str(x[0]) for x in R.get(prev, [])} & {str(x[0]) for x in R.get(nxt, [])} - set(byid)
            identities, eligible = [], []
            for t in pool:
                identity, observed = own.get(int(t.track_id), ('', -10000))
                fresh = (int(t.frame_id)-1) - observed <= 12
                identities.append(identity)
                eligible.append(bool(identity) and fresh and identity not in gaps)
            high_ids = ctx['high_ids']
            assert len(high_ids) == len(detections)
            costs = self.get_dists(pool, detections)
            baseline = assignment_vector(costs, self.args.match_thresh)
            correct = correctness(baseline, identities, eligible, high_ids)
            for ri, t in enumerate(pool):
                state = 'tracked' if t.state == TrackState.Tracked else 'lost'
                counts[state + '_pool'] += 1
                if not eligible[ri]:
                    counts[state + '_unknown'] += 1
                    continue
                counts[state + '_known'] += 1
                counts[state + ('_correct' if correct[ri] else '_incorrect')] += 1
                identity = identities[ri]
                present_high = identity in high_ids
                counts[state + ('_high_present' if present_high else '_high_absent')] += 1
                event = dict(video=video, frame=f, tid=int(t.track_id), incoming_state=state,
                             identity=identity, before_correct=bool(correct[ri]), high_present=present_high,
                             baseline_detection=int(baseline[ri]), beneficial=False, harmful=False)
                if not correct[ri] and present_high:
                    box = np.asarray(byid[identity][2], float)
                    center = (box[:2] + box[2:]) / 2
                    crop = crop_box(tuple(float(v) for v in t.mean[:4]), (1280,720))
                    eye = np.asarray(to_eye(center, crop, geometry['eye_size']))
                    in_field = bool(crop[0] <= center[0] <= crop[2] and crop[1] <= center[1] <= crop[3]
                        and np.linalg.norm(geom.points[geom.nearest(eye)] - eye) <= .6 * geom.spacing)
                    alternate = shadow_center(self, pool, detections, costs, ri, center)
                    flags = correctness(alternate, identities, eligible, high_ids)
                    outcome = action_outcome(correct[np.asarray(eligible)], flags[np.asarray(eligible)], int(np.asarray(eligible)[:ri].sum()))
                    event.update(outcome, oracle_center=center.tolist(), oracle_detection=int(alternate[ri]), in_field=in_field)
                    counts[state + '_oracle_evaluated'] += 1
                    counts[state + '_oracle_beneficial'] += outcome['beneficial']
                    counts[state + '_oracle_harmful'] += outcome['harmful']
                    counts[state + '_infield_beneficial'] += outcome['beneficial'] and in_field
                ctx['events'].append(event)
            assert np.array_equal(self.get_dists(pool, detections), costs), 'Shadow action changed real costs'
            return super()._first_association(pool, detections, activated, refind)

    BaseTrack.reset_id()
    tracker = Observe(make_tracker_args(**XYWH_CONFIG))
    for f in range(count):
        if (RUN / 'stop.request').exists() or now().isoformat() >= protocol['save_at']:
            raise TimeoutError('Original timing diagnostic budget')
        ix = np.flatnonzero(det['frame_idx'] == f)
        boxes, scores, classes = det['xyxy'][ix], det['conf'][ix], det['cls'][ix]
        ctx.update(frame=f, events=[])
        if f in R:
            all_ids = label(list(boxes), [CLASS_NAMES[int(c)] for c in classes], R[f])
            ctx['high_ids'] = [identity for identity, score in zip(all_ids, scores) if score >= tracker.args.track_high_thresh]
        out = tracker.update(_DetSlice(boxes, scores, classes))
        if len(out):
            outputs.append(np.column_stack([np.full(len(out), f), np.asarray(out)[:, :7]]).astype(np.float32))
        final_states = {int(t.track_id):'tracked' for t in tracker.tracked_stracks}
        final_states.update({int(t.track_id):'lost' for t in tracker.lost_stracks})
        for event in ctx['events']:
            event['actual_final_state'] = final_states.get(event['tid'], 'removed')
            rows.append(event)
            if event['incoming_state'] == 'tracked' and not event['before_correct']:
                counts['tracked_first_incorrect_final_' + event['actual_final_state']] += 1
        if f in R and len(boxes):
            detection_ids = label(list(boxes), [CLASS_NAMES[int(c)] for c in classes], R[f])
            for t in tracker.tracked_stracks:
                if t.frame_id == tracker.frame_id:
                    own[int(t.track_id)] = (detection_ids[int(np.argmax(_iou([t.xyxy], boxes)[0]))], f)
        if f % 100 == 0:
            write(RUN / 'status.json', dict(state='running', stage='Точный повтор и граница включения',
                variant=video, frame=f, frames=count, done=protocol['videos'].index(video), total=26, updated=now().isoformat()))
    actual = np.concatenate(outputs) if outputs else np.empty((0,8), np.float32)
    assert np.array_equal(actual, expected), (video, 'Baseline output changed')
    assert counts['tracked_known'] + counts['tracked_unknown'] == counts['tracked_pool']
    assert counts['lost_known'] + counts['lost_unknown'] == counts['lost_pool']
    return dict(video=video, seconds=time.monotonic()-began, frames=count, counts=dict(counts), rows=rows,
                exact_baseline=True, baseline_sha256=sha(ROOT / f'runs/fly_signatures/{video}/signatures.npz'))


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    p = read(RUN / 'protocol.json')
    verify_sources(p)
    jobs = []
    for video in p['videos']:
        path = RUN / 'videos' / (video + '.json')
        assert not path.exists(), 'Completed job must not be repeated'
        job = replay(video, p)
        write(path, job)
        jobs.append(job)
        if len(jobs) == 2:
            left = (datetime.fromisoformat(p['save_at']) - now()).total_seconds()
            estimate = 2 * np.mean([j['seconds'] for j in jobs]) * (26-2) + 60
            write(RUN / 'price.json', dict(remaining=left, estimated=estimate, passed=bool(estimate < left)))
            if estimate >= left:
                raise RuntimeError('Timing gate: conservative remaining cost exceeds original budget')
    totals = Counter()
    for job in jobs:
        totals.update(job['counts'])
    rates = {}
    for job in jobs:
        c = Counter(job['counts'])
        n = c['tracked_known'] + c['lost_known']
        assert n > 0
        rates[job['video']] = dict(known=n, tracked_only=c['tracked_infield_beneficial']/n,
            lost_only=c['lost_infield_beneficial']/n,
            all_active=(c['tracked_infield_beneficial']+c['lost_infield_beneficial'])/n)
    result = paired({v:r['all_active'] for v,r in rates.items()}, {v:r['lost_only'] for v,r in rates.items()})
    verify_sources(p)
    passed = result['mean_per_video'] >= .02 and result['ci95'][0] > 0
    write(RUN / 'report.json', dict(complete=True, primary_pass=bool(passed), counts=dict(totals), per_video=rates,
        paired_added_tracked_center_oracle=result, exact_baseline_videos=26, sources_unchanged=True,
        not_hota=True, not_flyvis_performance=True, oracle_actions_not_jointly_applied=True, limits=p['limits']))
    write(RUN / 'status.json', dict(state='completed', stage='Проверка мишени раннего включения окончена', done=26, total=26, updated=now().isoformat()))


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--worker' in sys.argv:
        try:
            worker()
        except Exception as error:
            write(RUN / 'status.json', dict(state='failed', stage=repr(error), updated=now().isoformat()))
            raise
    else:
        raise ValueError('Choose --prepare or --worker')
