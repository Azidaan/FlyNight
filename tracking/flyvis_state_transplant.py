"""Replace selected last-axis state coordinates without aliasing either source."""
import torch

def transplant_nodes(recipient, donor, indices):
    if recipient.shape != donor.shape or recipient.dtype != donor.dtype or recipient.device != donor.device:
        raise ValueError('State shape, dtype and device must match')
    if recipient.ndim < 1 or not recipient.is_floating_point():
        raise ValueError('Expected floating node state')
    if not torch.isfinite(recipient).all() or not torch.isfinite(donor).all():
        raise ValueError('Non-finite state')
    ix = torch.as_tensor(indices, dtype=torch.long, device=recipient.device)
    if ix.ndim != 1 or ix.numel() == 0 or ix.unique().numel() != ix.numel():
        raise ValueError('Indices must be nonempty, unique and one-dimensional')
    if ix.min() < 0 or ix.max() >= recipient.shape[-1]:
        raise IndexError('Node index out of range')
    result = recipient.clone()
    result[..., ix] = donor[..., ix]
    return result
