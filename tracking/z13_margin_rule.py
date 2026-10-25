"""Compare fixed-score policies; calibration never sees validation-video outcomes."""
import numpy as np


PROBABILITY_GRID = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01)
MARGIN_GRID = (0., .01, .02, .05, .1, .2, .3, .5, .75, 1., None)


def actions(logits, rule, threshold):
    result = []
    for z in logits:
        z = np.asarray(z, float)
        if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
            raise ValueError('Finite nonempty scores required')
        k = int(z.argmax())
        if rule == 'margin':
            take = threshold is not None and z[k] > threshold
        elif rule == 'probability':
            offset = max(0., float(z[k]))
            probability = np.exp(z[k]-offset)/(np.exp(-offset)+np.exp(z-offset).sum())
            take = z[k] > 0 and probability >= threshold
        else:
            raise ValueError(rule)
        result.append(k if take else -1)
    return np.asarray(result, int)


def outcomes(chosen, good, harm, kind):
    chosen = np.asarray(chosen)
    known = np.asarray(kind) >= 0
    take = (chosen >= 0) & known
    result = np.zeros(len(chosen), float)
    idx = np.flatnonzero(take)
    result[idx] = good[idx, chosen[idx]].astype(float)-harm[idx, chosen[idx]].astype(float)
    return result


def crossfit(logits, good, harm, kind, video, videos, folds, rule):
    grid = MARGIN_GRID if rule == 'margin' else PROBABILITY_GRID
    video = np.asarray(video); kind = np.asarray(kind)
    if set(folds) != set(videos) or len(set(videos)) != len(videos):
        raise ValueError('Complete unique video-fold binding required')
    all_actions = [actions(logits, rule, t) for t in grid]
    all_values = [outcomes(a, good, harm, kind) for a in all_actions]
    scores = np.array([[vals[(video == i) & (kind >= 0)].mean()
                       for i in range(len(videos))] for vals in all_values])
    if not np.isfinite(scores).all():
        raise ValueError('Each video requires known outcomes')
    chosen = np.full(len(kind), -2, int); decisions = []
    for fold in sorted(set(folds.values())):
        train_v = [i for i, v in enumerate(videos) if folds[v] != fold]
        val_v = [i for i, v in enumerate(videos) if folds[v] == fold]
        train_rows = np.isin(video, train_v); val_rows = np.isin(video, val_v)
        best = max(range(len(grid)), key=lambda j: (
            float(scores[j, train_v].mean()),
            -int((all_actions[j][train_rows] >= 0).sum()), j))
        chosen[val_rows] = all_actions[best][val_rows]
        decisions.append(dict(fold=fold, threshold=grid[best],
                              calibration_videos=[videos[i] for i in train_v],
                              validation_videos=[videos[i] for i in val_v],
                              calibration_net=float(scores[best, train_v].mean())))
    assert (chosen >= -1).all()
    net = outcomes(chosen, good, harm, kind)
    per = {v: float(net[(video == i) & (kind >= 0)].mean()) for i, v in enumerate(videos)}
    counts = dict(rows=len(kind), known=int((kind >= 0).sum()),
                  selected=int((chosen >= 0).sum()), unknown_selected=int(((chosen >= 0) & (kind < 0)).sum()),
                  beneficial=int((net > 0).sum()), harmful=int((net < 0).sum()))
    return dict(per_video=per, counts=counts, folds=decisions), chosen
