"""Read-only supervision provenance and privileged visibility diagnostic.

The oracle uses annotations for a diagnostic only; this is never inference code.
"""
from collections import Counter
import math
import numpy as np


def label_reasons(kind, center, crop, identity, distance):
    kind = np.asarray(kind)
    center, crop = np.asarray(center), np.asarray(crop)
    valid_id = (np.asarray(identity).astype(str) != '') & (np.asarray(distance) >= 0) & (np.asarray(distance) <= 12)
    finite = np.isfinite(center).all(1)
    if not set(kind.tolist()) <= {-1, 0, 1}:
        raise ValueError('Unexpected kind')
    if np.any((kind >= 0) & ~valid_id) or np.any((kind == 1) & ~finite):
        raise ValueError('Known target needs a valid observed identity and finite positive center')
    inside = ((center >= crop[:, :2]) & (center <= crop[:, 2:])).all(1)
    reasons = np.full(len(kind), '', dtype=object)
    reasons[kind == 1] = 'target_center_in_eye'
    reasons[(kind == 0) & finite & ~inside] = 'annotated_target_outside_square'
    reasons[(kind == 0) & finite & inside] = 'annotated_target_outside_hex_field'
    reasons[(kind == 0) & ~finite] = 'target_not_annotated_in_query'
    reasons[(kind == -1) & ~valid_id] = 'unknown_observed_identity_or_stale'
    reasons[(kind == -1) & valid_id] = 'unknown_query_annotation_gap'
    if any(x == '' for x in reasons):
        raise ValueError('Incomplete label partition')
    return reasons.tolist()


def select_known(rows, oracle=False, fraction=.01):
    known = [r for r in rows if r['kind'] >= 0]
    if len({r['label'] for r in known}) != len(known):
        raise ValueError('Duplicate label')
    n = math.ceil(len(known) * fraction)
    if oracle:
        # Privileged target presence first; original score within each group.
        order = sorted(known, key=lambda r: (-int(r['kind'] == 1), -r['score'], r['label']))
    else:
        order = sorted(known, key=lambda r: (-r['score'], r['label']))
    return order[:n]


def counts(rows):
    out = Counter(r['outcome'] if r['kind'] >= 0 else 'unknown' for r in rows)
    return {k: out[k] for k in ('beneficial', 'harmful', 'neutral', 'unknown')} | {'selected': len(rows)}
