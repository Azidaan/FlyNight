"""Scalar audit implementation; no worker or production-policy imports."""
import math

def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.

def scalar_cosines(references, current):
    left_norms = [math.sqrt(math.fsum(float(x) ** 2 for x in row)) for row in references]
    right_norms = [math.sqrt(math.fsum(float(x) ** 2 for x in row)) for row in current]
    assert all(norm > 0 for norm in left_norms + right_norms)
    return [[min(1., max(-1., math.fsum(float(x) * float(y) for x, y in zip(left, right)) / (left_norms[i] * right_norms[j]))) for j, right in enumerate(current)] for i, left in enumerate(references)]

def groups(candidates):
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]['confidence'], candidates[i]['raw_id']))
    result = []
    while order:
        chosen = order[0]
        aliases = [chosen]
        remaining = []
        for index in order[1:]:
            if candidates[index]['cls'] == candidates[chosen]['cls'] and iou(candidates[index]['box'], candidates[chosen]['box']) > .7:
                aliases.append(index)
            else:
                remaining.append(index)
        result.append(dict(representative_index=chosen, representative_raw_id=candidates[chosen]['raw_id'], alias_indices=aliases, alias_raw_ids=[candidates[i]['raw_id'] for i in aliases]))
        order = remaining
    return result

def reconstruct(rows, candidates, full_cosine, frozen, eligible, *, complete=True, enabled=True, no_descriptors=False):
    grouped = groups(candidates)
    if not enabled or not complete or no_descriptors:
        reason = 'branch_disabled' if not enabled else 'incomplete_competitor_descriptors'
        return grouped, [(r['tid'], 'defer', reason, None) for r in rows], [], None, None
    representatives = [group['representative_index'] for group in grouped]
    similarity = [[values[j] for j in representatives] for values in full_cosine]
    proximity = [[iou(row['predicted_xyxy'], candidates[j]['box']) for j in representatives] for row in rows]
    occupied_rows = {tid for tid, _ in frozen}
    occupied_raw = {raw for _, raw in frozen}
    decisions = []
    proposals = []
    for i, row in enumerate(rows):
        action, selected = 'defer', None
        if row['tid'] in occupied_rows:
            reason = 'native_assignment_frozen'
        elif row['state'] != 2:
            reason = 'row_not_lost'
        elif row['tid'] not in eligible:
            reason = 'outside_observable_native_recovery_episode'
        elif not grouped:
            action, reason = 'refuse', 'no_raw_candidate'
        else:
            potential = [j for j, raw_index in enumerate(representatives) if candidates[raw_index]['cls'] == row['cls'] and candidates[raw_index]['confidence'] <= .1 and (1 - similarity[i][j]) / 2 <= 1 - .8 and proximity[i][j] >= .5]
            if not potential:
                action, reason = 'refuse', 'no_excluded_low_candidate_passes_fixed_appearance_and_proximity'
            else:
                possible = [j for j, raw_index in enumerate(representatives) if candidates[raw_index]['cls'] == row['cls']]
                forward_maximum = max(similarity[i][j] for j in possible)
                maxima = [j for j in possible if similarity[i][j] == forward_maximum]
                if len(maxima) != 1:
                    reason = 'tied_best_candidate_group'
                else:
                    j = maxima[0]
                    candidate = candidates[representatives[j]]
                    reverse_rows = [k for k, competitor in enumerate(rows) if competitor['cls'] == candidate['cls']]
                    reverse_maximum = max(similarity[k][j] for k in reverse_rows)
                    reverse = [k for k in reverse_rows if similarity[k][j] == reverse_maximum]
                    if any(raw in occupied_raw for raw in grouped[j]['alias_raw_ids']):
                        reason = 'candidate_alias_owned_by_frozen_native_assignment'
                    elif candidate['confidence'] > .1:
                        reason = 'best_group_is_native_candidate_not_recovery_branch'
                    elif j not in potential:
                        reason = 'best_group_fails_fixed_absolute_gates'
                    elif len(reverse) != 1 or reverse[0] != i:
                        reason = 'not_strict_mutual_nearest_against_all_competitor_rows'
                    else:
                        action, reason, selected = 'continue', 'unassigned_mutual_low_candidate_passes_fixed_gates', candidate['raw_id']
                        proposals.append((row['tid'], selected, grouped[j]['alias_raw_ids']))
        decisions.append((row['tid'], action, reason, selected))
    return grouped, decisions, proposals, similarity, proximity
