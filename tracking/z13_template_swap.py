"""Readout-only template intervention; query activity/state never enter matching."""
import hashlib
import numpy as np


def choose_donors(recipient, templates, count=3):
    """Exact observed mask, known different identity/video, deterministic choice.

    Only observed metadata are accepted; neither query labels nor scores are used.
    Distinct donor identities avoid counting repeated episodes as three controls.
    """
    if not recipient['identity']:
        return []
    candidates = [x for x in templates if x['mask'] == recipient['mask']
                  and x['identity'] and x['identity'] != recipient['identity']
                  and x['video'] != recipient['video']]
    salt = (recipient['video'] + '/' + str(recipient['episode'])).encode()
    candidates.sort(key=lambda x: hashlib.sha256(salt + x['split'].encode()
                    + str(x['episode']).encode()).digest())
    result, identities = [], set()
    for candidate in candidates:
        if candidate['identity'] in identities:
            continue
        result.append(candidate)
        identities.add(candidate['identity'])
        if len(result) == count:
            return result
    return []


def project_models(search_features, weights):
    """Full 721-column projections, not rescoring original template's peaks.

    A per-model constant intercept is omitted: the existing standardized score
    and local weighted center are invariant to a constant map offset.
    """
    x, w = np.asarray(search_features, np.float64), np.asarray(weights, np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError('Search features and complete target-filter weights required')
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Nonfinite projection inputs')
    return w @ x.T


def interpretation(pair):
    """Dependence is a diagnostic, never a proof of an intrinsic identity defect."""
    lower, upper = pair['ci95']
    if lower > 0:
        return 'own_template_advantage_detected'
    if upper < .02:
        return 'useful_center_advantage_below_2pp_in_this_test'
    return 'inconclusive'
