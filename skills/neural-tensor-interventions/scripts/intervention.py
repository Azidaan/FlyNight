"""Small non-mutating, indexed PyTorch intervention with invariant checks."""
import torch

def scale_nodes(activity, indices, factor):
    if not isinstance(activity, torch.Tensor) or activity.ndim < 1 or not activity.is_floating_point():
        raise ValueError('expected a floating tensor with nodes on the last axis')
    if not torch.isfinite(activity).all() or not 0 <= factor <= 1:
        raise ValueError('non-finite input or factor outside [0,1]')
    idx = torch.as_tensor(indices, dtype=torch.long, device=activity.device)
    if idx.ndim != 1 or idx.numel() == 0 or idx.unique().numel() != idx.numel():
        raise ValueError('indices must be a nonempty unique vector')
    if int(idx.min()) < 0 or int(idx.max()) >= activity.shape[-1]:
        raise IndexError('node index out of range')
    result = activity.clone()
    result[..., idx] = activity[..., idx] * factor
    if not torch.isfinite(result).all():
        raise ValueError('intervention produced non-finite output')
    return result
