"""Frozen-A execution adapter. No labels, fitting, tracker, or model access."""
import math
from tracking.identity_repair_adapter import compose, VARIANTS


def baseline(snapshot, variant='baseline', failure=None):
    record = dict(event_key=snapshot['event_key'], variant=variant,
                  video=snapshot['video'], frame=snapshot['frame'],
                  base='corrected_cost', edge_updates=[], append_columns=[],
                  freeze='all' if variant in VARIANTS[2:] else 'none',
                  defer_tids=[], alias_groups=[])
    if failure is not None:
        record['execution_failure'] = failure
    return record


def compose_event(snapshot, rows, candidates, references, current,
                  original_score, census_score, variant):
    assert variant in VARIANTS
    if variant == 'baseline':
        return baseline(snapshot), dict(action='literal_saved_matrix')
    missing = [name for name, value in [('original_score', original_score),
                                       ('census_score', census_score)]
               if value is None or not math.isfinite(float(value))]
    if missing:
        failure = dict(kind='frozen_production_requires_finite_scores',
                       unavailable=missing, fallback='literal_saved_matrix',
                       no_numeric_substitution=True,
                       scope='Appearance-only frozen API also casts both scores before branching; this is execution incompleteness, not a negative scientific vote.')
        return baseline(snapshot, variant, failure), dict(action='execution_failure_preserve_baseline', **failure)
    record, note = compose(snapshot, rows, candidates, references, current,
                           original_score, census_score, variant)
    record['event_key'] = snapshot['event_key']
    return record, note
