"""Causal two-frame candidate attention inside the FlyVis readout.

No labels or true centers enter the feature/model APIs. This is soft context
exchange, not supervised identity assignment or the full KeepTrack algorithm.
"""
import numpy as np
from tracking.z13_candidate_ranker import candidate_inputs, make_ranker


def previous_inputs(queries, sequences):
    lookup = {(episode, f['t']): f for episode, frames in sequences.items() for f in frames}
    xs, gs, masks = [], [], []
    for episode, pos, pred in queries:
        current = lookup[(episode, pos + 4)]
        previous = lookup[(episode, pos + 3)]
        assert previous['t'] == current['t'] - 1
        x, _, g, mask = candidate_inputs(previous['centers'], previous['features'],
            previous['scores'], current['seed_center'], current['seed_features'],
            pred, current['width'], pos)
        xs.append(x); gs.append(g); masks.append(mask)
    return tuple(np.asarray(a) for a in (xs, gs, masks))


def make_temporal(torch, activity=True):
    nn = torch.nn
    class Temporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = make_ranker(torch, activity)
            self.match = nn.Sequential(nn.Linear(35, 16), nn.Tanh(), nn.Linear(16, 1))
            self.null = nn.Parameter(torch.zeros(()))
            self.delta = nn.Sequential(nn.Linear(107, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)

        def forward(self, x, seed, geom, mask, px, pg, pm):
            if not self.base.activity:
                x = torch.zeros_like(x); seed = torch.zeros_like(seed); px = torch.zeros_like(px)
            z = self.base.embed(x); s = self.base.embed(seed)[:, None].expand_as(z)
            context = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            h = torch.cat([z, s, (z-s).abs(), z*s, context[:, None].expand_as(z), geom], -1)
            pz = self.base.embed(px)
            a = z[:, :, None]; b = pz[:, None]
            pair = torch.cat([(a-b).abs(), a*b,
                geom[:, :, None, :2]-pg[:, None, :, :2],
                pg[:, None, :, 4:5].expand(-1, z.shape[1], -1, -1)], -1)
            logits = self.match(pair).squeeze(-1).masked_fill(~pm[:, None], -1e4)
            weights = torch.cat([logits, self.null.expand(*logits.shape[:2], 1)], -1).softmax(-1)
            message = weights[..., :-1] @ pz
            position = weights[..., :-1] @ pg[..., :2]
            extra = torch.cat([h, message, position, weights[..., -1:]], -1)
            return (self.base.score(h).squeeze(-1) + self.delta(extra).squeeze(-1)).masked_fill(~mask, -1e4)
    return Temporal()
