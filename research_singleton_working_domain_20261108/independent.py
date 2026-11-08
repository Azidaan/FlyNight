"""Independent costs, eligibility, sparse labels and optional-assignment objective."""
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from research_singleton_working_domain_20261108.common import *


def identity_labels(video, native, saved):
    from datasets.bdd100k_loader import CLASS_NAMES
    annotations = read(ROOT / f'data/bdd100k/labels/box_track_20/train/{video}.json')
    aligned = read(ROOT / f'runs/train_cache/{video}/alignment.json')['matches']
    wanted = {a['frame']: a for a in saved['new_needed_frame_assignments']}
    labels = defaultdict(list)
    for frame, match in zip(annotations, aligned):
        f = int(match['video_frame_index'])
        if f not in wanted:
            continue
        out = native[native[:, 0] == f]
        found = [None] * len(out)
        objects = [x for x in frame.get('labels') or [] if x['category'] in CLASS_NAMES.values()
                   and not (x.get('attributes') or {}).get('crowd', False)]
        for cls in set(map(int, out[:, 7])):
            ii = np.flatnonzero(out[:, 7] == cls)
            jj = [j for j, x in enumerate(objects) if x['category'] == CLASS_NAMES[cls]]
            if not jj:
                continue
            a = out[ii, 1:5].astype(float)
            b = np.array([[objects[j]['box2d'][k] for k in ['x1','y1','x2','y2']] for j in jj], float)
            size = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
            inter = size.prod(2)
            area = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)[:, None] + np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)[None, :] - inter
            overlap = inter / np.maximum(area, 1e-12)
            overlap[overlap < .5] = 0
            ri, ci = linear_sum_assignment(-overlap)
            for i, j in zip(ri, ci):
                if overlap[i, j] > 0:
                    found[ii[i]] = str(objects[jj[j]]['id'])
        assert [[int(row[5]), name] for row, name in zip(out, found)] == wanted[f]['matches']
        for row, name in zip(out, found):
            labels[int(row[5])].append((f, name))
    return labels


def audit_video(video, native, output, data, value, protocol):
    from research_bbox_singleton_audit_recovery1_20261108.native_cost import native_cost
    with np.load(ROOT / protocol['videos'][video]['baseline'], allow_pickle=False) as z:
        assert np.array_equal(native[:, :8], z['outputs'])
    assert np.array_equal(native[:, [0,1,2,3,4,6,7,8]], output[:, [0,1,2,3,4,6,7,8]])
    labels = read(RUN / 'labels' / (video + '.json'))
    identities = identity_labels(video, native, labels)
    raw = {}
    qualified = {}
    checks = Counter()
    for event in data['events']:
        for row in event['rows']:
            for c in row['candidates']:
                assert native_cost(row['row_data']['mean'], c['box'], c['confidence']) == c['cost']
                checks['native_costs'] += 1
            candidates = [c for c in row['candidates'] if c['forward_iou'] >= .2]
            if row['age'] > 2 or len(candidates) != 1 or candidates[0]['new_tid'] is None:
                continue
            c = candidates[0]
            key = f"{event['frame']}|{row['old_tid']}|{c['new_tid']}"
            raw[key] = (event, row, c)
            if not np.float32(c['cost'] - .025) > .8:
                qualified[key] = raw[key]
    for name, collection in [('singleton', raw), ('cost_qualified', qualified)]:
        expected = Counter()
        saved = {f"{e['frame']}|{e['old_tid']}|{e['new_tid']}": e for e in value['stages'][name]['rows']}
        assert set(saved) == set(collection)
        for key, (event, row, c) in collection.items():
            f = event['frame']; last = row['row_data']['history'][-1]['frame']
            old = [(t, g) for t, g in identities[row['old_tid']] if last-12 <= t <= last and g is not None]
            new = [(t, g) for t, g in identities[c['new_tid']] if f <= t <= f+12 and g is not None]
            old_ids = {g for _, g in old}; new_ids = {g for _, g in new}
            kind = 'unknown' if len(old_ids) != 1 or len(new_ids) != 1 else 'correct' if old_ids == new_ids else 'foreign'
            assert saved[key]['kind'] == kind
            assert saved[key]['label']['target_anchors'] == [list(x) for x in old]
            assert saved[key]['label']['fragment_anchors'] == [list(x) for x in new]
            expected[kind] += 1
            checks['edge_labels'] += 1
        assert dict(expected) == value['stages'][name]['counts']
    for d in value['decisions']:
        allowed = d['allowed']; links = d['links']
        for e in allowed:
            assert f"{d['frame']}|{e['old_tid']}|{e['new_tid']}" in qualified
        rr = sorted({e['old_tid'] for e in allowed}); cc = sorted({e['new_tid'] for e in allowed})
        R, C = len(rr), len(cc)
        if R and C:
            matrix = np.full((R+C, R+C), np.inf); matrix[R:, C:] = 0
            for i in range(R): matrix[i, C+i] = .4
            for j in range(C): matrix[R+j, j] = .4
            for e in allowed: matrix[rr.index(e['old_tid']), cc.index(e['new_tid'])] = np.float32(e['cost']-.025)
            ii, jj = linear_sum_assignment(matrix)
            actual = sum(float(np.float32(e['corrected_cost'])) for e in links) + (R+C-2*len(links))*.4
            assert abs(float(matrix[ii, jj].sum())-actual) < 1e-5
            checks['optional_assignments'] += 1
    for stage in value['stages'].values():
        assert dict(Counter(row['kind'] for row in stage['rows'])) == stage['counts']
    return dict(passed=True, checks=dict(checks), full_native_columns_exact=8,
                physical_outputs_exact=True, sparse_actual_anchors_only=True)
