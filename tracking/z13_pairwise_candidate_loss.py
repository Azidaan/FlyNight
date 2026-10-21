"""Train within-frame candidate ordering, not an absolute presence probability."""


def pairwise_candidate_loss(torch, scores, target, mask):
    """RankNet logistic loss, useful candidate above nonuseful in the same frame.

    Tied labels and padded candidates create no pairs. Unknown frames must be
    excluded by the caller exactly as in the matched BCE training control.
    An empty pair set has a finite zero loss and zero gradient.
    """
    if scores.ndim != 2 or scores.shape != target.shape or scores.shape != mask.shape:
        raise ValueError('Matched frame/candidate arrays required')
    valid=mask.bool();pairs=valid[:,:,None]&valid[:,None,:]&(target[:,:,None]>target[:,None,:])
    difference=scores[:,None,:]-scores[:,:,None]
    loss=torch.nn.functional.softplus(difference)
    return (loss*pairs).sum()/pairs.sum().clamp(min=1),pairs.sum()
