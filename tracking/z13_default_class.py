"""Matched readouts with an explicit default score and action-cost supervision."""


def make_model(torch, activity=True):
    from tracking.z13_region_assignment_context import make_context_ranker
    nn = torch.nn
    class DefaultReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.regions = make_context_ranker(torch, activity=activity)
            self.default = nn.Sequential(nn.Linear(46, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, seed, geo, mask):
            if not activity:
                x = torch.zeros_like(x); seed = torch.zeros_like(seed)
            region = self.regions(x, seed, geo, mask)
            z = self.regions.embed(x)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (z * mask[..., None]).sum(1) / denom
            pooled_geo = (geo * mask[..., None]).sum(1) / denom
            h = torch.cat([pooled, self.regions.target(seed), pooled_geo], -1)
            default = self.default(h)
            # Differences are invariant to adding any common logit shift.
            return (region-default).masked_fill(~mask, -1e4)
    return DefaultReadout()


def action_loss(torch, difference, good, harm, mask, loss_name, epsilon):
    if difference.shape != good.shape or good.shape != harm.shape or harm.shape != mask.shape:
        raise ValueError('Aligned action outcomes required')
    if bool((good & harm & mask).any()) or not bool(mask.any(1).all()):
        raise ValueError('Exclusive outcomes and nonempty candidates required')
    target = good.to(difference.dtype)-harm.to(difference.dtype)-epsilon
    if loss_name == 'mse':
        return ((difference-target).square().masked_fill(~mask, 0).sum(1)/mask.sum(1)).mean()
    if loss_name != 'cost_ce':
        raise ValueError(loss_name)
    # Cost(default)=0. Unnormalised max(cost)-cost weights preserve expected-cost ordering.
    costs = torch.cat([torch.zeros_like(target[:, :1]), -target], 1)
    valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
    max_cost = costs.masked_fill(~valid, -float('inf')).max(1, keepdim=True).values
    weights = (max_cost-costs).masked_fill(~valid, 0)
    logits = torch.cat([torch.zeros_like(difference[:, :1]), difference], 1).masked_fill(~valid, -1e4)
    return -(weights * torch.log_softmax(logits, 1)).sum(1).mean()
