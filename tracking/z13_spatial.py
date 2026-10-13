"""Spatial FlyVis readout v2. No labels are accepted by the prediction API."""
from __future__ import annotations

import numpy as np

WIDTH = 32
TEMPLATE_POINTS = 64
QUALITY_FEATURES = 12


def make_spatial_readout(torch):
    nn = torch.nn

    class SpatialReadout(nn.Module):
        def __init__(self, in_dim, points, eye_size, neighbors):
            super().__init__()
            self.register_buffer('mu', torch.zeros(in_dim))
            self.register_buffer('sd', torch.ones(in_dim))
            self.register_buffer('xy', torch.as_tensor(points / eye_size * 2 - 1, dtype=torch.float32))
            self.register_buffer('neighbors', torch.as_tensor(neighbors, dtype=torch.long))
            self.emb = nn.Linear(in_dim, WIDTH)
            self.local = nn.Linear(WIDTH * 2, WIDTH)
            self.pos = nn.Linear(2, WIDTH, bias=False)
            self.q = nn.Linear(WIDTH, WIDTH)
            self.k = nn.Linear(WIDTH, WIDTH)
            self.v = nn.Linear(WIDTH, WIDTH)
            self.norm = nn.LayerNorm(WIDTH)
            self.score = nn.Sequential(nn.Linear(WIDTH * 3, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
            self.absent = nn.Sequential(nn.Linear(WIDTH * 2, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))

        def embed(self, feats):
            x = self.emb((feats.transpose(1, 2) - self.mu) / self.sd)
            adjacent = x[:, self.neighbors].mean(2)
            return torch.nn.functional.gelu(self.local(torch.cat((x, adjacent), -1)))

        def exemplar(self, feats, mask):
            x = self.embed(feats)
            if not bool(mask.any(1).all()):
                raise ValueError('Empty exemplar mask')
            # Select a deterministic spread of real columns. Padded slots are masked.
            indices, valid = [], []
            for m in mask:
                ids = torch.where(m)[0]
                n = min(len(ids), TEMPLATE_POINTS)
                pick = torch.linspace(0, len(ids) - 1, n, device=x.device).round().long()
                chosen = ids[pick]
                indices.append(torch.cat((chosen, chosen[-1:].expand(TEMPLATE_POINTS - n))))
                valid.append(torch.arange(TEMPLATE_POINTS, device=x.device) < n)
            idx = torch.stack(indices)
            ok = torch.stack(valid)
            features = x.gather(1, idx[..., None].expand(-1, -1, WIDTH))
            xy = self.xy[idx]
            lo = self.xy[None].expand(len(mask), -1, -1).masked_fill(~mask[..., None], float('inf')).amin(1)
            hi = self.xy[None].expand(len(mask), -1, -1).masked_fill(~mask[..., None], -float('inf')).amax(1)
            relative = (xy - (hi + lo)[:, None] / 2) / ((hi - lo)[:, None] / 2).clamp_min(1e-3)
            return features + self.pos(relative), ok

        def logits(self, ex, feats):
            template, valid = ex
            search = self.embed(feats)
            query = self.q(search + self.pos(self.xy)[None])
            key = self.k(template)
            scores = query @ key.transpose(1, 2) / WIDTH ** .5
            weights = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
            context = weights @ self.v(template)
            joint = self.norm(search + context)
            col = self.score(torch.cat((joint, context, search * context), -1)).squeeze(-1)
            absent = self.absent(torch.cat((joint.mean(1), joint.amax(1)), -1))
            return torch.cat((col, absent), -1)

    return SpatialReadout


def quality_features(probabilities, geom, center_eye, center_frame, prediction, crop):
    """Only observable map statistics and geometry; no target box/identity/error."""
    p = np.asarray(probabilities, np.float64)
    spatial = p[:721] / max(float(p[:721].sum()), 1e-20)
    order = np.sort(spatial)
    points = geom.points / geom.eye_size
    mean = (spatial[:, None] * points).sum(0)
    variance = (spatial[:, None] * (points - mean) ** 2).sum(0)
    k = int(spatial.argmax())
    local = float(spatial[np.unique(geom.neighbors[k])].sum())
    entropy = -float((spatial * np.log(spatial.clip(1e-20))).sum()) / np.log(721)
    xy = np.asarray(center_eye) / geom.eye_size
    crop = np.asarray(crop, float)
    size = np.maximum(crop[2:] - crop[:2], 1)
    shift = (np.asarray(center_frame) - np.asarray(prediction)) / size
    return np.asarray([1 - p[721], order[-1], order[-1] - order[-2], entropy,
                       local, variance[0], variance[1], np.linalg.norm(xy - mean),
                       shift[0], shift[1], np.linalg.norm(shift),
                       min(xy[0], xy[1], 1 - xy[0], 1 - xy[1])], np.float32)


def make_quality_head(torch):
    class QualityHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mu', torch.zeros(QUALITY_FEATURES))
            self.register_buffer('sd', torch.ones(QUALITY_FEATURES))
            self.net = torch.nn.Sequential(torch.nn.Linear(QUALITY_FEATURES, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

        def forward(self, x):
            return self.net((x - self.mu) / self.sd).squeeze(-1)
    return QualityHead
