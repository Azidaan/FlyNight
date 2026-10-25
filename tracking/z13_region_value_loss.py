"""Learn the signed observed assignment outcome, retaining neutral action targets."""
def signed_action_loss(torch,scores,good,harm,mask):
    if scores.shape!=good.shape or scores.shape!=harm.shape or scores.shape!=mask.shape:raise ValueError('Aligned action arrays required')
    if bool((good&harm&mask).any()) or not bool(mask.any(1).all()):raise ValueError('Exclusive outcomes and nonempty query actions required')
    target=good.to(scores.dtype)-harm.to(scores.dtype)
    error=(scores-target).square().masked_fill(~mask,0)
    return (error.sum(1)/mask.sum(1)).mean()
