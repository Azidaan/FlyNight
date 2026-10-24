"""Expressible extreme outputs of a categorical region readout, without truth."""
import numpy as np


def categorical_rows(original, confidence, valid):
    original=np.asarray(original,float);confidence=np.asarray(confidence,float);valid=np.asarray(valid,bool)
    if original.shape!=confidence.shape or valid.shape!=original.shape or original.ndim!=1:
        raise ValueError('Aligned detection vectors required')
    if not np.isfinite(original).all() or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):
        raise ValueError('Finite costs and detection probabilities required')
    rows=[original.copy(),np.ones(len(original))];choices=[-2,-1]
    for j in np.flatnonzero(valid):
        row=np.ones(len(original));row[j]=1-confidence[j]
        rows.append(row);choices.append(int(j))
    return rows,choices
