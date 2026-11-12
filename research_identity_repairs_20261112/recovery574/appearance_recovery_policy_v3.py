"""Candidate-only low-confidence recovery; no tracker mutation or truth inputs.

Borrowed fixed BoT-SORT appearance/proximity gates are hypotheses for MASA,
not a MASA probability calibration. All native assignments stay authoritative.
"""
from __future__ import annotations
import numpy as np

APPEARANCE_THRESH = 0.8
PROXIMITY_THRESH = 0.5
NMS_IOU = 0.7
NATIVE_LOW = 0.1

def overlap(left, right):
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    size = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(size.prod())
    union = float(np.maximum(0.0, a[2:] - a[:2]).prod() + np.maximum(0.0, b[2:] - b[:2]).prod() - intersection)
    return intersection / union if union > 0 else 0.0

def candidate_groups(candidates):
    """Confidence-priority, class-specific greedy NMS with explicit aliases."""
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]['confidence'], candidates[i]['raw_id']))
    groups = []
    while order:
        representative = order.pop(0)
        item = candidates[representative]
        aliases = [representative]
        remaining = []
        for index in order:
            other = candidates[index]
            if item['cls'] == other['cls'] and overlap(item['box'], other['box']) > NMS_IOU:
                aliases.append(index)
            else:
                remaining.append(index)
        order = remaining
        groups.append(dict(representative_index=representative, representative_raw_id=item['raw_id'],
                           alias_indices=aliases, alias_raw_ids=[candidates[i]['raw_id'] for i in aliases]))
    return groups

def confirm_recovery(rows, candidates, reference_vectors, current_vectors, baseline_pairs, *, competition_complete, enabled=True):
    """Return continue/defer/refuse proposals for complete same-frame row pools.

    rows: tid, state (2 lost is the only recovery-eligible state), cls, predicted_xyxy.
    candidates: raw_id, box, confidence, cls. No label fields are consumed.
    baseline_pairs: native (tid, raw_id), frozen before this optional branch.
    Vectors correspond exactly to input order; absence of competitors defers.
    """
    assert len({r['tid'] for r in rows}) == len(rows)
    assert len({c['raw_id'] for c in candidates}) == len(candidates)
    groups = candidate_groups(candidates)
    frozen_pairs = [(int(tid), int(raw)) for tid, raw in baseline_pairs]
    assigned_rows = {tid for tid, _ in frozen_pairs}
    occupied_raw = {raw for _, raw in frozen_pairs}
    decisions = []
    proposals = []
    base = dict(groups=groups, baseline_pairs=[list(p) for p in frozen_pairs])
    if not enabled or not competition_complete or reference_vectors is None or current_vectors is None:
        reason = 'branch_disabled' if not enabled else 'incomplete_competitor_descriptors'
        return dict(**base, decisions=[dict(tid=int(r['tid']), action='defer', reason=reason, raw_id=None) for r in rows], proposals=[], cosine=None, group_iou=None)
    a = np.asarray(reference_vectors, dtype=float)
    b = np.asarray(current_vectors, dtype=float)
    assert a.ndim == b.ndim == 2 and len(a) == len(rows) and len(b) == len(candidates) and a.shape[1] == b.shape[1]
    assert np.isfinite(a).all() and np.isfinite(b).all()
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    assert np.all(an > 0) and np.all(bn > 0)
    representatives = [g['representative_index'] for g in groups]
    cosine = np.clip((a / an[:, None]) @ (b[representatives] / bn[representatives, None]).T, -1.0, 1.0)
    proximity = np.array([[overlap(row['predicted_xyxy'], candidates[j]['box']) for j in representatives] for row in rows])
    for i, row in enumerate(rows):
        decision = dict(tid=int(row['tid']), action='defer', reason='', raw_id=None)
        if row['tid'] in assigned_rows:
            decision['reason'] = 'native_assignment_frozen'
        elif row['state'] != 2:
            decision['reason'] = 'row_not_lost'
        elif not groups:
            decision.update(action='refuse', reason='no_raw_candidate')
        else:
            same_class = np.array([candidates[j]['cls'] == row['cls'] for j in representatives])
            low = np.array([candidates[j]['confidence'] <= NATIVE_LOW for j in representatives])
            appearance_cost = (1.0 - cosine[i]) / 2.0
            absolute = same_class & low & (appearance_cost <= 1.0 - APPEARANCE_THRESH) & (proximity[i] >= PROXIMITY_THRESH)
            if not absolute.any():
                decision.update(action='refuse', reason='no_excluded_low_candidate_passes_fixed_appearance_and_proximity')
            else:
                scores = np.where(same_class, cosine[i], -np.inf)
                maxima = np.flatnonzero(scores == scores.max())
                if len(maxima) != 1:
                    decision['reason'] = 'tied_best_candidate_group'
                else:
                    j = int(maxima[0])
                    group = groups[j]
                    candidate = candidates[representatives[j]]
                    reciprocal = np.array([r['cls'] == candidate['cls'] for r in rows])
                    reverse_scores = np.where(reciprocal, cosine[:, j], -np.inf)
                    reverse = np.flatnonzero(reverse_scores == reverse_scores.max())
                    decision.update(best_raw_id=int(candidate['raw_id']), cosine=float(cosine[i, j]), appearance_cost=float(appearance_cost[j]), proximity_iou=float(proximity[i, j]), alias_raw_ids=group['alias_raw_ids'])
                    if any(raw in occupied_raw for raw in group['alias_raw_ids']):
                        decision['reason'] = 'candidate_alias_owned_by_frozen_native_assignment'
                    elif not low[j]:
                        decision['reason'] = 'best_group_is_native_candidate_not_recovery_branch'
                    elif not absolute[j]:
                        decision['reason'] = 'best_group_fails_fixed_absolute_gates'
                    elif len(reverse) != 1 or int(reverse[0]) != i:
                        decision['reason'] = 'not_strict_mutual_nearest_against_all_competitor_rows'
                    else:
                        decision.update(action='continue', reason='unassigned_mutual_low_candidate_passes_fixed_gates', raw_id=int(candidate['raw_id']))
                        proposals.append(dict(tid=int(row['tid']), raw_id=int(candidate['raw_id']), alias_raw_ids=group['alias_raw_ids'], appearance_cost=float(appearance_cost[j]), cosine=float(cosine[i, j]), proximity_iou=float(proximity[i, j])))
        decisions.append(decision)
    assert len({p['tid'] for p in proposals}) == len(proposals)
    assert all(p['tid'] not in assigned_rows and not any(k in occupied_raw for k in p['alias_raw_ids']) for p in proposals)
    return dict(**base, decisions=decisions, proposals=proposals, cosine=cosine.tolist(), group_iou=proximity.tolist())
