"""Fit the existing fixed utility estimator on training videos only."""
import numpy as np


def fit_training_only(features, targets, videos, evaluation_videos):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=int)
    videos = np.asarray(videos)
    if set(videos) & set(evaluation_videos):
        raise ValueError('Training and evaluation videos overlap')
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(videos):
        raise ValueError('Unmatched feature/target/video arrays')
    if not np.isfinite(x).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Invalid features or utility labels')
    known = y >= 0
    if set(y[known]) != {0, 1}:
        raise ValueError('Both known utility outcomes required')
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=500, random_state=0))
    model.fit(x[known], y[known])
    if model[-1].n_iter_.max() >= 500:
        raise RuntimeError('Utility optimizer did not converge')
    return model, dict(fit_videos=sorted(set(videos)), fit_known=int(known.sum()),
                      fit_useful=int((y[known] == 1).sum()),
                      excluded_unknown=int((~known).sum()), features=x.shape[1])
