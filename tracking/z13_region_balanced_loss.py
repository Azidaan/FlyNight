"""Class-balanced action loss, normalized to the current batch weight sum."""
def balanced_action_loss(torch,scores,targets,mask,positive_weight):
    from tracking.z13_competing_candidates import marginal_choice_loss
    has=((targets>0)&mask).any(1)
    weights=torch.where(has,torch.full_like(scores[:,0],float(positive_weight)),torch.ones_like(scores[:,0]))
    return marginal_choice_loss(torch,scores,targets,mask,positive_weight)/weights.mean()
