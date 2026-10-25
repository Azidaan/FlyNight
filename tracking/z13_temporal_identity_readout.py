"""Internal regional association with partial true-ID supervision and an unmatched option."""


def make_model(torch, activity=True, history=True):
    from tracking.z13_region_assignment_context import make_context_ranker
    nn = torch.nn

    class TemporalIdentityReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.activity = activity
            self.history = history
            self.base = make_context_ranker(torch, activity=activity)
            self.association = nn.Sequential(nn.Linear(37, 16), nn.Tanh(), nn.Linear(16, 1))
            self.unmatched = nn.Parameter(torch.zeros(()))
            self.delta = nn.Sequential(nn.Linear(116, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)

        def forward(self, x, seed, geo, mask, previous, previous_mask, pair_geometry):
            if not self.activity:
                x = torch.zeros_like(x)
                seed = torch.zeros_like(seed)
                previous = torch.zeros_like(previous)
            base_score = self.base(x, seed, geo, mask)
            z = self.base.embed(x)
            past = self.base.embed(previous)
            target = self.base.target(seed)[:, None].expand_as(z)
            context = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            h = torch.cat([z, target, (z-target).abs(), z*target,
                           context[:, None].expand_as(z), geo], -1)
            a, b = z[:, :, None], past[:, None]
            pair = torch.cat([(a-b).abs(), a*b, pair_geometry], -1)
            logits = self.association(pair).squeeze(-1)
            pmask = previous_mask if self.history else torch.zeros_like(previous_mask)
            valid = mask[:, :, None] & pmask[:, None]
            # Index0 is the explicit unmatched ACTION, not a physical candidate identity.
            valid = valid.clone()
            valid[:, 0] = False
            scores = logits.masked_fill(~valid, -1e4)
            weights = torch.cat([scores, self.unmatched.expand(*scores.shape[:2], 1)], -1).softmax(-1)
            message = weights[..., :-1] @ past
            displacement = (weights[..., :-1, None] * pair_geometry).sum(2)
            delta = self.delta(torch.cat([h, message, displacement, weights[..., -1:]], -1)).squeeze(-1)
            delta = delta * pmask.any(1, keepdim=True)
            return (base_score + delta).masked_fill(~mask, -1e4), logits, valid

    return TemporalIdentityReadout()


def identity_loss(torch, logits, targets, known, valid):
    use = known & valid
    counts = use.sum((1, 2))
    errors = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    per_row = errors.masked_fill(~use, 0).sum((1, 2)) / counts.clamp(min=1)
    return per_row[counts > 0].mean() if bool((counts > 0).any()) else logits.sum() * 0
