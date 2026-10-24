"""Explicit coordinate-only replay guard; scientific outputs are never modified."""
import numpy as np


def compare_outputs(actual, reference, tolerance=0.0001):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.ndim == reference.ndim == 2 and actual.shape == reference.shape
    assert actual.shape[1] == 8, 'Expected frame,x1,y1,x2,y2,id,score,class'
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert np.array_equal(actual[:, [0, 5, 6, 7]], reference[:, [0, 5, 6, 7]]), 'Frame/ID/confidence/class changed'
    delta = np.abs(actual[:, 1:5].astype(np.float64) - reference[:, 1:5].astype(np.float64))
    maximum = float(delta.max()) if delta.size else 0.0
    assert maximum <= tolerance, ('Coordinate replay exceeds approved absolute tolerance', maximum, tolerance)
    return dict(accepted=True, exact=bool(np.array_equal(actual, reference)),
                coordinate_max_abs=maximum, coordinate_tolerance=float(tolerance),
                different_rows=int(np.count_nonzero(np.any(actual != reference, axis=1))),
                row_count=int(actual.shape[0]), frames_ids_scores_classes_exact=True)
