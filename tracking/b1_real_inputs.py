"""Входы Б1: причинные указатели и поправка вероятности без смены действий."""
from __future__ import annotations

import numpy as np


def bind_observation(row, archive, choices):
    """Никаких истинных личностей: только эпизод, наблюдаемый образец и области."""
    episode = int(row['episode_local'])
    start, stop = map(int, archive['ep_offset'][episode:episode + 2])
    current = int(row['current_array_index'])
    reference = start + 3
    if not start <= reference < current < stop:
        raise ValueError('Current/reference observation lies outside its original episode')
    if list(choices) != [-1, *row['current_choices']]:
        raise ValueError('Changed NULL/detection action binding')
    frames, crops = archive['fr_frame'], archive['fr_crop']
    if int(frames[current]) != int(row['current_frame']) or not np.array_equal(crops[current], row['current_crop']):
        raise ValueError('Changed current frame or crop')
    masks = np.zeros((len(choices), archive['ex_mask'].shape[1]), bool)
    for k, columns in enumerate(row['current_masks'], start=1):
        masks[k, columns] = True
    if not masks[1:].any(-1).all():
        raise ValueError('Unsupported real region')
    ref_mask = archive['ex_mask'][episode].astype(bool)
    history = bool(ref_mask.any() and int(frames[reference]) < int(frames[current]))
    return dict(uid=int(row['uid']), video=row['video'], episode_local=episode,
                retinal_source=row['retinal_source'], prefix_start=start, current_array_index=current,
                reference_array_index=reference, reference_index=3, current_frame=int(frames[current]),
                reference_frame=int(frames[reference]), reference_crop=crops[reference].tolist(),
                current_crop=crops[current].tolist(), has_history=history,
                reference_mask=np.flatnonzero(ref_mask).tolist(), current_masks=row['current_masks'],
                choices=list(choices), baseline_action='implicit unchanged action; not an extra neural slot',
                null_slot=0, null_action=-1, sequence_frames=current - start + 1)


def corrected_identity_probability(probability, delta, real_mask):
    """Сдвиг логарифма шансов: нулевая поправка сохраняет float побитно.

    p' = p*exp(delta)/(1-p+p*exp(delta)), записано через expm1.
    В отличие от sigmoid(logit(p)), при delta=0 нет ошибки округления.
    NULL и дополненные слоты сохраняют прежние NaN; градиент в delta живой.
    """
    import torch
    safe = torch.where(real_mask, probability, torch.full_like(probability, .5))
    change = torch.expm1(delta)
    updated = safe + safe * (1 - safe) * change / (1 + safe * change)
    return torch.where(real_mask, updated, probability)
