"""Bounded CPU preparation of causal regional pairs; no model or activity execution."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
RUN = ROOT / 'runs/z13_temporal_pair_readiness_20261025'
BASE = ROOT / 'runs/z13_box_supervision_followup_20261025'
IDENT = 'z13_temporal_pair_readiness_v1'
MSK = timezone(timedelta(hours=3))
CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def now():
    return datetime.now(MSK)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
    temporary.replace(path)


def journal(title, text):
    path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    before = path.read_bytes()
    newline = b'\r\n' if b'\r\n' in before else b'\n'
    marker = '## Журнал изменений'.encode() + newline + newline
    assert before.count(marker) == 1
    entry = f'**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n{text}\n\n'
    path.write_bytes(before.replace(marker, marker + entry.replace('\n', newline.decode()).encode(), 1))


def choose_previous(indices, frames, current_index, supported, annotated):
    """Selection sees observation availability, never identity labels or outcomes."""
    available = [int(i) for i in indices if int(i) < current_index
                 and int(frames[i]) < int(frames[current_index])
                 and int(frames[i]) in annotated and supported(int(i))]
    return max(available, key=lambda i: (int(frames[i]), i)) if available else None


def pair_labels(current, previous):
    # Unmatched detections remain unknown, never fabricated negative identities.
    return [[i, j, int(a == b)] for i, a in enumerate(current) if a is not None
            for j, b in enumerate(previous) if b is not None]


def paired(values):
    d = np.asarray(list(values.values()), float)
    samples = d[np.random.default_rng(1).integers(0, len(d), (5000, len(d)))].mean(1)
    return dict(mean_per_video=float(d.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                better=int((d > 0).sum()), n=len(d))


def prepare():
    assert not RUN.exists(), 'Do not overwrite an existing preparation'
    RUN.mkdir()
    original = read(BASE / 'protocol.json')
    videos = original['fit20'] + original['fit84'] + original['evaluate26']
    assert len(videos) == len(set(videos)) == 130
    sources = ['runs/z13_box_supervision_followup_20261025/protocol.json',
               'runs/z13_box_supervision_followup_20261025/temporal_readiness.json',
               'runs/z13_box_supervision_followup_20261025/strict_action_gate.json',
               'runs/z13_local_peak_20261021/geometry.json',
               'tracking/z13_region_assignment.py', 'tracking/z13_readout.py',
               'scripts/association_log.py', 'scripts/output_filter_oracle.py',
               Path(__file__).relative_to(ROOT).as_posix(),
               'tests/test_z13_temporal_pair_readiness.py']
    sources += ['runs/z13_candidate_reserve_20261021/fit/labels.json',
                'runs/z13_candidate_data104_20261021/evaluate/labels.json',
                'runs/z13_candidate_reserve_20261021/evaluate/labels.json']
    for video in videos:
        sources += [f'runs/z13_data/{video}.npz', f'runs/train_cache/{video}/detections/raw_video.npz',
                    f'runs/train_cache/{video}/alignment.json',
                    f'data/bdd100k/labels/box_track_20/train/{video}.json',
                    f'runs/z13_box_supervision_followup_20261025/cache/videos/{video}.json']
    start = now()
    hard = min(start + timedelta(seconds=1200), datetime(2026, 10, 26, 7, 40, tzinfo=MSK))
    p = dict(id=IDENT, created=start.isoformat(), timezone='Europe/Moscow', device='cpu',
             budget_seconds=1200, save_at=(hard - timedelta(seconds=20)).isoformat(),
             hard_stop=hard.isoformat(), videos=videos,
             splits={k: original[k] for k in ('fit20', 'fit84', 'evaluate26')},
             authorization='Author approved this specific <=20 minute CPU metadata/region/identity-label preparation; no training or new FlyVis activity.',
             action='Prepare a causal same-episode regional correspondence readout that could change the existing full-box choice; this stage only prepares its data.',
             selection='Latest strictly earlier stored real annotated frame with >=1 supported high-confidence region in the SAME original episode prefix. Repeated padded frame slots use the last occurrence. Class is the current observable track class. No GT identity, good/harm label, model score or future frame selects the past. Annotations specify supervision availability only; this fixed annotated schedule is a training-data policy, not a runtime requirement.',
             regions='Unchanged regions() with conf>=.25; current choices must exactly equal old choices. Earlier crop is the original stored crop; its midpoint supplies only unused geometry prediction coordinates. No fabricated tracker mean. Save mask column indices and original retinal-array pointers.',
             truth='Offline same-class Hungarian IoU>=.5 on real raw annotations. Only pairs with BOTH matched GT identities get positive/negative correspondence labels; unmatched detections remain unknown. A missing positive does not establish object absence. Tracker IDs group observations but never label identity.',
             caveat='Same-episode histories differ from old previous-query links, which can cross episode resets; report both gained and lost support. No claim all D/C/R1 feature values already exist at restored frames. Retinal source pointers are common, but neural features still require original state replay.',
             gates=['All130 videos and37296 unchanged query/current-action bindings audited; all sources unchanged.',
                    'Past selection strictly causal and same-episode; labels physically separate from inputs; unknown identities retained.',
                    'Report original and restored structural reserve, labeled-pair and target-linked subsets with full26 denominators and video bootstrap. No threshold tuning or scientific success inference.',
                    'Preserve partial work on budget/price failure; first2video projected remaining cost must fit original1200 seconds. No GPU or Torch import.'],
             sha256={name: sha(ROOT / name) for name in sources})
    write(RUN / 'protocol.json', p)
    # Protect all pre-existing modifications, untracked files and local deletions.
    entries = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z'], cwd=ROOT).decode('utf8').split('\0')
    foreign = {}
    for entry in entries:
        if not entry:
            continue
        name = entry[3:]
        if name in (Path(__file__).relative_to(ROOT).as_posix(), 'tests/test_z13_temporal_pair_readiness.py'):
            continue
        path = (ROOT / name).resolve()
        path.relative_to(ROOT.resolve())
        try:
            path.relative_to(RUN.resolve())
            continue
        except ValueError:
            pass
        if path.is_dir():
            for f in path.rglob('*'):
                if f.is_file():
                    foreign[f.relative_to(ROOT).as_posix()] = sha(f)
        else:
            foreign[name] = sha(path) if path.is_file() else None
    write(RUN / 'foreign_snapshot.json', foreign)
    catalog = read(ROOT / 'experiments/project_history.json')
    assert not any(c['id'] == IDENT for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT, title='З13: готовность реальных межкадровых областей и меток соответствий',
        area='flyvis', state='ready', started=start.strftime('%Y-%m-%d'), tags=['fly-problem-P8'],
        question='Какую часть потерянного запаса можно обеспечить причинно доступной историей и проверяемыми метками соответствий?',
        hypothesis='Часть отсутствующих пар восстанавливается из сохранённых наблюдений до отдельного запроса; это гипотеза готовности данных, не причина ошибок FlyVis.',
        expected=p['selection'] + ' CPU <=1200с, без обучения и новой активности.',
        acceptance=[dict(criterion=x, met=None) for x in p['gates']], obtained='', interpretation='', decision='',
        cannot_claim='Нет проверки обучаемости, улучшения FlyVis или HOTA. Идеальный выбор по разметке не является прогнозом. Неизвестные соответствия не заменяются номерами трекера.',
        files=[Path(__file__).relative_to(ROOT).as_posix(), 'tests/test_z13_temporal_pair_readiness.py'],
        evidence=[dict(kind='file', ref=(RUN / 'protocol.json').relative_to(ROOT).as_posix(), claim='Разрешение, неизменные источники и условия до проверки')]))
    write(ROOT / 'experiments/project_history.json', catalog)
    journal('постановка согласованной подготовки временных пар', '*Постановка.* Автор согласовал до20мин CPU на130видео, без обучения/новой активности. '+p['selection']+'\n\n*Критерии до прогона.* '+' '.join(p['gates'])+'\n\n*Срок.* '+p['hard_stop']+'. Источник: `runs/z13_temporal_pair_readiness_20261025/protocol.json`.')
    write(RUN / 'status.json', dict(state='ready', done=0, total=130, updated=now().isoformat()))


def worker():
    from tracking.z13_readout import HexGeometry
    from tracking.z13_region_assignment import regions
    from scripts.association_log import label as detection_labels
    p = read(RUN / 'protocol.json')
    started = time.monotonic()
    oldpairs = read(BASE / 'temporal_readiness.json')['previous_query']
    g = read(ROOT / 'runs/z13_local_peak_20261021/geometry.json')
    geom = HexGeometry(np.asarray(g['points'], np.float32), g['eye_size'])
    groups = [('fit20', p['splits']['fit20'], ROOT / 'runs/z13_candidate_reserve_20261021/fit/labels.json'),
              ('fit84', p['splits']['fit84'], ROOT / 'runs/z13_candidate_data104_20261021/evaluate/labels.json'),
              ('evaluate26', p['splits']['evaluate26'], ROOT / 'runs/z13_candidate_reserve_20261021/evaluate/labels.json')]
    uid = 0
    per = {}
    totals = Counter()
    manifest = {}
    times = []
    for split, videos, labelpath in groups:
        all_labels = read(labelpath)
        offset = 0
        for video in videos:
            before = time.monotonic()
            if now() >= datetime.fromisoformat(p['save_at']) or (RUN / 'stop.request').exists():
                raise TimeoutError('Original CPU budget reached; completed videos retained')
            data_path = ROOT / f'runs/z13_data/{video}.npz'
            with np.load(data_path) as z:
                d = {k: z[k] for k in ('ep_tid', 'ep_identity', 'ep_pert', 'ep_offset', 'fr_frame', 'fr_crop')}
            with np.load(ROOT / f'runs/train_cache/{video}/detections/raw_video.npz') as z:
                det = {k: z[k] for k in ('xyxy', 'conf', 'cls', 'frame_idx')}
            annotations = read(ROOT / f'data/bdd100k/labels/box_track_20/train/{video}.json')
            aligned = read(ROOT / f'runs/train_cache/{video}/alignment.json')['matches']
            assert len(annotations) == len(aligned)
            truth = {}
            for frame, alignment in zip(annotations, aligned):
                objects = []
                for obj in frame.get('labels') or []:
                    if obj['category'] in CLASSES and not (obj.get('attributes') or {}).get('crowd', False):
                        b = obj['box2d']
                        objects.append((str(obj['id']), obj['category'], [b[k] for k in ('x1', 'y1', 'x2', 'y2')]))
                truth[int(alignment['video_frame_index'])] = objects
            cached = read(BASE / 'cache/videos' / (video + '.json'))['entries']
            frame_cache = {}
            region_cache = {}

            def detections(f):
                if f not in frame_cache:
                    ix = np.flatnonzero((det['frame_idx'] == f) & (det['conf'] >= .25))
                    boxes, conf, classes = det['xyxy'][ix], det['conf'][ix], det['cls'][ix]
                    ids = detection_labels(list(boxes), [CLASSES[int(c)] for c in classes], truth[f]) if f in truth else [None] * len(ix)
                    frame_cache[f] = (boxes, conf, classes, ids)
                return frame_cache[f]

            def region(index, category):
                key = (index, category)
                if key not in region_cache:
                    f = int(d['fr_frame'][index]); crop = d['fr_crop'][index]
                    boxes, conf, classes, _ = detections(f)
                    prediction = (crop[:2].astype(float) + crop[2:]) / 2
                    choices, masks, _ = regions(geom, crop, prediction, boxes, conf, classes, category)
                    region_cache[key] = (choices, masks)
                return region_cache[key]

            inputs, labels = [], []
            counts = Counter()
            gaps = Counter()
            for l in all_labels:
                if l['video'] != video:
                    continue
                if len(inputs) % 50 == 0 and (now() >= datetime.fromisoformat(p['save_at']) or (RUN / 'stop.request').exists()):
                    raise TimeoutError('Save boundary reached inside video')
                e = l['episode'] - offset
                assert not d['ep_pert'][e]
                start = int(d['ep_offset'][e]); current = start + 4 + l['pos']
                assert current < d['ep_offset'][e + 1]
                f = int(d['fr_frame'][current]); entry = cached[str(uid)]
                category = int(entry['track_class'])
                choices, masks = region(current, category)
                assert choices == entry['choices'] and l['kind'] == entry['kind']
                current_regions = choices[1:]
                previous = choose_previous(range(start, current), d['fr_frame'], current,
                    lambda i: len(region(i, category)[0]) > 1, set(truth)) if current_regions else None
                old = bool(oldpairs.get(str(uid), {}).get('eligible'))
                supported = previous is not None and bool(current_regions)
                cur_ids = [detections(f)[3][j] for j in current_regions]
                past_ids = []
                past_input = None
                if supported:
                    pf = int(d['fr_frame'][previous]); pc, pm = region(previous, category)
                    assert pf < f and start <= previous < current
                    past_ids = [detections(pf)[3][j] for j in pc[1:]]
                    gaps[f-pf] += 1
                    past_input = dict(array_index=previous, episode_relative_index=previous-start, frame=pf,
                        crop=d['fr_crop'][previous].tolist(), choices=pc[1:],
                        masks=[np.flatnonzero(m).tolist() for m in pm[1:]],
                        boxes=[detections(pf)[0][j].tolist() for j in pc[1:]])
                correspondence = pair_labels(cur_ids, past_ids)
                known_pair = bool(correspondence)
                positive = any(x[2] for x in correspondence)
                good = [j for j, value in zip(entry['choices'], entry['good']) if j >= 0 and value]
                helpful = bool(good) and l['kind'] >= 0
                target = str(d['ep_identity'][e]) or None
                target_linked = bool(target and target in past_ids and any(detections(f)[3][j] == target for j in good))
                flags = dict(rows=True, known=l['kind'] >= 0, current_supported=bool(current_regions),
                    old_supported=old, restored_supported=supported, gained_support=supported and not old,
                    lost_support=old and not supported, labeled_pairs=known_pair, positive_pair=positive,
                    oracle_all=helpful, oracle_old=helpful and old, oracle_restored=helpful and supported,
                    oracle_labeled=helpful and known_pair, oracle_target_linked=helpful and target_linked,
                    oracle_gained=helpful and supported and not old, oracle_lost=helpful and old and not supported,
                    oracle_restored_gap12=helpful and supported and f-int(d['fr_frame'][previous]) <= 12)
                counts.update({k: int(value) for k, value in flags.items()})
                counts['positive_correspondences'] += sum(x[2] for x in correspondence)
                counts['negative_correspondences'] += sum(not x[2] for x in correspondence)
                row = dict(uid=uid, video=video, split=split, episode_local=e, tid=int(d['ep_tid'][e]),
                    retinal_source=data_path.relative_to(ROOT).as_posix(), current_array_index=current,
                    current_frame=f, current_crop=d['fr_crop'][current].tolist(), current_choices=current_regions,
                    current_masks=[np.flatnonzero(m).tolist() for m in masks[1:]], previous=past_input,
                    state_prefix_start=start, state_prefix_end=current, current_binding_preserved=True)
                inputs.append(row)
                labels.append(dict(uid=uid, current_identity=cur_ids, previous_identity=past_ids,
                    correspondence=correspondence, target_identity=target, original_good=entry['good'],
                    original_harm=entry['harm'], flags=flags))
                uid += 1
            assert len(inputs) == len(cached)
            offset += len(d['ep_tid'])
            per[video] = dict(counts)
            totals.update(counts)
            ip = RUN / 'inputs' / (video + '.json'); lp = RUN / 'supervision' / (video + '.json')
            write(ip, dict(rows=inputs, source_sha256=p['sha256'][data_path.relative_to(ROOT).as_posix()]))
            write(lp, dict(rows=labels, identity_labels_are_offline_only=True))
            elapsed = time.monotonic() - before
            write(RUN / 'videos' / (video + '.json'), dict(complete=True, counts=dict(counts), gaps=dict(gaps),
                seconds=elapsed, input_sha256=sha(ip), supervision_sha256=sha(lp), current_choices_exact=True))
            manifest[video] = sha(RUN / 'videos' / (video + '.json'))
            times.append(elapsed)
            write(RUN / 'status.json', dict(state='running', stage='Причинные области и отдельные метки личности',
                done=len(times), total=130, variant=video, elapsed_seconds=time.monotonic()-started, updated=now().isoformat()))
            if len(times) == 2:
                estimate = 2 * np.mean(times) * 128 + 90
                remaining = (datetime.fromisoformat(p['save_at']) - now()).total_seconds()
                write(RUN / 'price.json', dict(first_two_seconds=times, conservative_remaining_seconds=float(estimate),
                      available_seconds=remaining, proceed=bool(estimate < remaining)))
                assert estimate < remaining, 'Original20minute budget insufficient; retain completed videos'
    assert uid == 37296 and totals['known'] == 26600
    metrics = {}
    for key in [k for k in totals if k.startswith('oracle_')]:
        metrics[key] = paired({v: per[v].get(key, 0)/per[v]['known'] for v in p['splits']['evaluate26']})
    assert abs(metrics['oracle_all']['mean_per_video'] - read(BASE/'strict_action_gate.json')['paired']['mean_per_video']) < 1e-15
    assert abs(metrics['oracle_old']['mean_per_video'] - read(BASE/'temporal_readiness.json')['eligible_oracle']['mean_per_video']) < 1e-15
    for name, digest in p['sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert 'torch' not in sys.modules
    split_counts = {s: dict(sum((Counter(per[v]) for v in vs), Counter())) for s, vs in p['splits'].items()}
    write(RUN / 'report.json', dict(complete=True, counts=dict(totals), per_video=per, per_split=split_counts,
        oracle26=metrics, manifest=manifest, sources_unchanged=True, original_oracles_reproduced=True,
        no_torch_import=True, trained_models=0, new_activity_arrays=0, seconds=time.monotonic()-started,
        caveat='Structural and labeled-data availability only. Target-linked subset is descriptive, not a ceiling for arbitrary readouts. Neural feature parity remains untested.'))
    write(RUN / 'status.json', dict(state='completed', done=130, total=130, updated=now().isoformat()))


def controller():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p = read(RUN/'protocol.json'); ps = WindowsProcesses()
    assert not (RUN/'controller_result.json').exists()
    with (RUN/'worker.stdout.log').open('w', encoding='utf8') as out, (RUN/'worker.stderr.log').open('w', encoding='utf8') as err:
        child = subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'), '-u', __file__, '--worker'], cwd=ROOT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=out, stderr=err,
            creationflags=subprocess.IDLE_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
        birth = ps.creation(child.pid); assert birth is not None
        write(RUN/'controller.json', dict(pid=os.getpid(), worker_pid=child.pid, worker_birth=birth,
            started=now().isoformat(), hard_stop=p['hard_stop']))
        stopped = False
        while child.poll() is None:
            if now() >= datetime.fromisoformat(p['save_at']):
                (RUN/'stop.request').touch(exist_ok=True)
            if now() >= datetime.fromisoformat(p['hard_stop']):
                if ps.creation(child.pid) == birth:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                stopped = True
                break
            time.sleep(.5)
        code = child.wait(timeout=10)
        ended = ps.creation(child.pid) != birth
        write(RUN/'controller_result.json', dict(exit_code=code, hard_timeout=stopped, all_owned_ended=ended,
            owned={str(child.pid): birth}, remaining={} if ended else {str(child.pid): birth}, ended=now().isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--controller', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.controller:
        controller()
    elif args.worker:
        try:
            worker()
        except Exception as error:
            write(RUN/'failure.json', dict(at=now().isoformat(), error=repr(error)))
            raise
    else:
        parser.error('Select one explicit operation')
