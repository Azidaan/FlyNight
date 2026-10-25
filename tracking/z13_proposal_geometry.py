"""Shadow proposal actions; truth is confined to explicitly named oracle arms."""
import numpy as np


def xywh(box):
    box = np.asarray(box, dtype=float)
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]


def in_crop(center, crop):
    x, y = center
    a, b, c, d = crop
    return bool(a <= x <= c and b <= y <= d)


def geometry_actions(track, detections, original_cost, get_dists, crop, oracle_box=None):
    """Keep all other rows fixed. Restore the full original mean, including on error."""
    original_cost = np.asarray(original_cost).copy()
    saved = track.mean.copy()
    names = ('null', 'centers_eye', 'boxes_eye', 'centers_frame', 'boxes_frame',
             'oracle_gt_center_eye', 'oracle_gt_box_eye')
    result = {k: [original_cost.copy()] for k in names}
    result['null'].append(np.ones_like(original_cost))
    proposals = []

    def evaluate(value, full):
        track.mean[:] = saved
        track.mean[:4 if full else 2] = value[:4 if full else 2]
        return np.asarray(get_dists([track], detections)[0]).copy()

    try:
        for j, detection in enumerate(detections):
            if int(detection.cls) != int(track.cls):
                continue
            box = np.asarray(detection.xyxy, dtype=float)
            value = xywh(box)
            if not np.isfinite(value).all() or np.any(value[2:] <= 0):
                continue
            inside = in_crop(value[:2], crop)
            center_cost, box_cost = evaluate(value, False), evaluate(value, True)
            result['centers_frame'].append(center_cost)
            result['boxes_frame'].append(box_cost)
            if inside:
                result['centers_eye'].append(center_cost)
                result['boxes_eye'].append(box_cost)
            proposals.append(dict(index=j, inside_eye=inside, xywh=value.tolist()))
        if oracle_box is not None:
            value = xywh(oracle_box)
            if in_crop(value[:2], crop):
                result['oracle_gt_center_eye'].append(evaluate(value, False))
                result['oracle_gt_box_eye'].append(evaluate(value, True))
    finally:
        track.mean[:] = saved
    assert np.array_equal(track.mean, saved)
    return result, dict(proposals=proposals, original_xywh=saved[:4].tolist(),
                        oracle_box=None if oracle_box is None else np.asarray(oracle_box).tolist())
