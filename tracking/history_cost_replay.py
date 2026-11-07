"""Повтор фактического порядка типов исходной NumPyоперации без допуска."""
import numpy as np


def replay_cost(original,score,alpha,dtype):
    dtype=np.dtype(dtype)
    if not np.issubdtype(dtype,np.floating) or not np.isfinite([original,score,alpha]).all():
        raise ValueError('Invalid recorded cost/score/dtype')
    value=dtype.type(original)
    if alpha==0:return value
    delta=float(float(alpha)*(.5-1/(1+np.exp(-np.clip(float(score),-80,80)))))
    # ВNumPy2np.float32scalar+Pythonfloat сначала приводитdeltaкFloat32.
    # Нельзя заменять это сложением двухPythonfloatсоднимокруглениемвконце.
    return dtype.type(value+delta)
