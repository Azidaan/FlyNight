"""Missing CPU metadata only. Reuse the frozen exporter; no image/model calls."""
from collections import Counter
import time
from research_singleton_working_domain_20261108.common import *


def key(frame, e):
    return f"{frame}|{e['old_tid']}|{e['new_tid']}"


def main(observation):
    import numpy as np
    import torch
    import research_bbox_confirmation_inventory_20261108.export as exporter
    import research_bbox_singleton_pilot_20261108.compare as evaluator
    from research_bbox_cost_pilot_20261108.compare import simulate
    from research_singleton_working_domain_20261108.independent import audit_video

    p = phase()
    assert read(RUN / 'authorization.json')['approved']
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    exporter.RUN = RUN
    exporter.status = status
    evaluator.RUN = RUN
    # Record the actual project modules loaded before the first scientific call.
    dependencies = {}
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix == '.py' and relative.parts[0] in ['tracking', 'scripts', 'datasets'] or path.suffix == '.py' and relative.parts[0].startswith('research_'):
            dependencies[relative.as_posix()] = sha(path)
    write(RUN / 'recovery1/execution_dependencies.json', dict(sources=dependencies,
        torch=torch.__version__, numpy=np.__version__), new=True)

    receipts = read(RUN / 'all_zero_parity.json')['videos']
    assert len(receipts) == 20
    for receipt in receipts.values():
        for name, digest in receipt['artifacts'].items():
            assert sha(ROOT / name) == digest, name

    results = {}
    artifacts = {}
    for video in p['videos']:
        status('labelling_and_auditing', video=video, completed_labels=len(results))
        data = read(RUN / 'native' / (video + '.json'))
        with np.load(RUN / 'native' / (video + '.npz'), allow_pickle=False) as z:
            native = z['outputs']
        raw_events = evaluator.events_for(data, dict(cap=2, threshold=.2, bidirectional=False))
        label_path = RUN / 'labels' / (video + '.json')
        labels = read(label_path)['rows'] if label_path.exists() else evaluator.labels_for(video, native, raw_events)
        cost_events = [dict(d, edges=[e for e in d['edges']
            if not np.float32(e['cost'] - .025) > .8]) for d in raw_events]
        output, decisions = simulate(native, data, cost_events, .025)
        per_stage = {}
        rows = {}
        for name, events in [('singleton', raw_events), ('cost_qualified', cost_events)]:
            counts = Counter()
            chosen = []
            for event in events:
                for e in event['edges']:
                    label = labels[key(event['frame'], e)]
                    kind = 'correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown'
                    counts[kind] += 1
                    row = dict(frame=event['frame'], old_tid=e['old_tid'], new_tid=e['new_tid'],
                               cost=e['cost'], age=e['age'], kind=kind, label=label,
                               reference_frame=e['row_data']['history'][-1]['frame'],
                               direct_reference_known=any(f == e['row_data']['history'][-1]['frame'] for f, _ in label['target_anchors']))
                    chosen.append(row)
                    rows[key(event['frame'], e)] = row
            per_stage[name] = dict(counts=dict(counts), rows=chosen)
        for name, field in [('canonical_allowed', 'allowed'), ('solver_selected', 'links')]:
            chosen = [dict(rows[key(d['frame'], e)], canonical_id=e['canonical_id'])
                      for d in decisions for e in d[field]]
            per_stage[name] = dict(counts=dict(Counter(e['kind'] for e in chosen)), rows=chosen)
        value = dict(passed=True, video=video, stages=per_stage, decisions=decisions,
                     original_export_counts=data['counts'],
                     working_iou_candidate_rows=[dict(frame=d['frame'], old_tid=row['old_tid'],
                        eligible_columns=[c['column'] for c in row['candidates'] if c['forward_iou'] >= .2])
                        for d in data['events'] for row in d['rows']],
                     public_output_signature=__import__('hashlib').sha256(output.tobytes()).hexdigest(),
                     physical_outputs_exact=True)
        path = RUN / 'outcomes' / (video + '.json')
        if path.exists():
            assert read(path) == __import__('json').loads(__import__('json').dumps(value))
        else:
            write(path, value, new=True)
        value = read(path)  # Audit the actual serialized artifact, as the auditor expects.
        audit = audit_video(video, native, output, data, value, p)
        write(RUN / 'audits' / (video + '.json'), audit, new=True)
        results[video] = {s: row['counts'] for s, row in per_stage.items()}
        for item in [path, RUN / 'labels' / (video + '.json'), RUN / 'audits' / (video + '.json')]:
            artifacts[item.relative_to(ROOT).as_posix()] = sha(item)
    stages = ['singleton', 'cost_qualified', 'canonical_allowed', 'solver_selected']
    totals = {s: dict(sum((Counter(v[s]) for v in results.values()), Counter())) for s in stages}
    distinct = {kind: [video for video, value in results.items()
                      if value['cost_qualified'].get(kind, 0)] for kind in ['correct', 'foreign']}
    gate = all(totals['cost_qualified'].get(k, 0) >= 3 and len(distinct[k]) >= 2 for k in distinct)
    for name, digest in dependencies.items():
        assert sha(ROOT / name) == digest, name
    phase()
    assert not torch.cuda.is_initialized()
    write(RUN / 'recovery1/report.json', dict(passed=True, complete=True, videos=len(results),
        totals=totals, per_video=results, distinct_class_videos=distinct,
        data_gate=dict(passed=gate, criterion=p['data_gate']), artifacts=artifacts,
        no_new_detector_calls=True, no_new_flyvis_calls=True, no_image_decode=True,
        no_training=True, no_hota_evaluation=True, no26_or20val_evaluation=True,
        first_problem_solved=False, no_automatic_followup=True,
        scope='Metadata readiness only in the fixed age2 alpha0.025 policy; sparse fragment labels, not direct identity at every frame.'), new=True)
    write(RUN / 'recovery1/independent_audit.json', dict(passed=True, videos=20,
        per_video_audits={v: sha(RUN / 'audits' / (v + '.json')) for v in results},
        report_sha256=sha(RUN / 'recovery1/report.json'), actual_dependencies_unchanged=True), new=True)
    status('data_readiness_complete', data_gate=gate, totals=totals)


if __name__ == '__main__':
    main(guard())
