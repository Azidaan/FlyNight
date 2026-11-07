"""Измерения переходного отклика; нормирование не является обучением сети."""
import numpy as np


def first_sustained(values, predicate, count=3):
    good = np.asarray(predicate(np.asarray(values)), bool)
    for i in range(len(good) - count + 1):
        if good[i:i + count].all():
            return int(i)
    return None


def response_metrics(flow, axis, speed, event, static_end=30, transition=55):
    """flow[t] относится к перемещению изображения t-1 -> t; событие в transition."""
    flow = np.asarray(flow, np.float64)
    q = flow[:, axis]
    baseline = float(q[21:31].mean())
    noise = float(q[21:31].std())
    signal = q - baseline
    plateau = float(signal[45:55].mean())
    eligible = bool(plateau > max(3 * noise, 3e-5))
    residual = signal[transition:] / plateau if eligible else np.full(len(q) - transition, np.nan)
    latency = None
    if eligible:
        latency = first_sustained(residual, (lambda x: x < 0) if event == 'reverse' else (lambda x: np.abs(x) <= .2))
    target = np.zeros_like(flow)
    target[31:55, axis] = speed
    if event == 'reverse':
        target[55:, axis] = -speed
    epe = np.linalg.norm(flow[31:] - target[31:], axis=1)
    return dict(eligible=eligible, baseline=baseline, static_std=noise,
                plateau=plateau, plateau_over_speed=plateau / speed,
                latency_frames=latency, right_censored=bool(eligible and latency is None),
                residual_first=float(residual[0]) if eligible else None,
                residual_last=float(np.mean(residual[-5:])) if eligible else None,
                epe_nominal_px=float(epe.mean()))
