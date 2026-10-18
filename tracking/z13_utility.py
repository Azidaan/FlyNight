"""Fixed observable features and video-held-out utility prediction."""
import numpy as np

GEOMETRY_NAMES = ('shift_x_crop', 'shift_y_crop', 'shift_norm_crop',
                  'log_crop_width', 'log_crop_height', 'log_search_position')
FULL_NAMES = GEOMETRY_NAMES + ('center_response_peak_z',)


def observable_features(row, crop):
    """Explicit whitelist: ignores outcome, GT, GT-derived scale and error fields."""
    crop = np.asarray(crop, dtype=float)
    width, height = np.maximum(crop[2:] - crop[:2], 1)
    delta = (np.asarray(row['center']) - row['pred']) / [width, height]
    geometry = np.array([*delta, np.linalg.norm(delta), np.log(width),
                         np.log(height), np.log1p(row['pos'])], dtype=float)
    full = np.r_[geometry, float(row['p_present'])]
    if not np.isfinite(full).all():
        raise ValueError('Nonfinite observable features')
    return geometry, full


def leave_video_out(features, targets, videos, on_fold=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=int)
    videos = np.asarray(videos)
    scores = np.full(len(targets), np.nan)
    records = []
    for i, video in enumerate(sorted(set(videos))):
        test = videos == video
        fit = (~test) & (targets >= 0)
        if len(np.unique(targets[fit])) != 2:
            raise ValueError('Both utility classes required in each training fold')
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=500, random_state=0))
        model.fit(features[fit], targets[fit])
        if model[-1].n_iter_.max() >= 500:
            raise RuntimeError('Utility optimizer did not converge')
        scores[test] = model.predict_proba(features[test])[:, 1]
        fit_videos = sorted(set(videos[fit]))
        assert video not in fit_videos
        records.append(dict(test_video=str(video), fit_videos=fit_videos,
                            fit_known=int(fit.sum()), test_rows=int(test.sum())))
        if on_fold:
            on_fold(i + 1, len(set(videos)), str(video))
    assert np.isfinite(scores).all()
    return scores, records
