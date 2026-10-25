"""Marginal likelihood of correct, nonharmful region actions, including unchanged."""


def semantic_choice_loss(torch,scores,correct,baseline_correct,mask):
    if scores.shape!=correct.shape or scores.shape!=mask.shape or scores.ndim!=2 or baseline_correct.shape!=(len(scores),) or not bool(mask.any(1).all()):
        raise ValueError('Aligned regional actions and baseline correctness required')
    s=scores.masked_fill(~mask,-torch.inf);good=correct.bool()&mask
    # If no regional action can be correct, the unchanged fallback is the only
    # expressible non-forced choice. Correct existing decisions always remain valid.
    base=baseline_correct.bool()|~good.any(1)
    zero=torch.zeros_like(s[:,:1]);all_scores=torch.cat([s,zero],1)
    acceptable=torch.cat([good,base[:,None]],1)
    return (torch.logsumexp(all_scores,1)-torch.logsumexp(all_scores.masked_fill(~acceptable,-torch.inf),1)).mean()
