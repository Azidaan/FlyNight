"""Эквивалентное пакетное сравнение кандидатов без циклов по столбцам."""
import numpy as np


def relative_features(features, mask):
    features = np.asarray(features)
    real = np.asarray(mask, bool).copy()
    real[0] = False
    values = np.where(real[:, None], np.nan_to_num(features, nan=-1e9), -np.inf)
    winner = values.argmax(axis=0)
    columns = np.arange(values.shape[1])
    maximum = values[winner, columns]
    second = values.copy()
    second[winner, columns] = -np.inf
    best_other = np.broadcast_to(maximum, values.shape).copy()
    best_other[winner, columns] = second.max(axis=0)
    with np.errstate(invalid='ignore'):
        diff = np.where(np.isfinite(best_other) & real[:, None] & np.isfinite(features),
                        features - best_other, np.nan)
    rank = np.where(real[:, None], (-values).argsort(axis=0).argsort(axis=0), np.nan)
    return np.stack((diff, rank), axis=-1).reshape(len(mask), -1).astype(np.float32)
