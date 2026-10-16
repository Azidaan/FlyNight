"""Isolated functional candidates; never imported by the running Z13 protocol."""
import numpy as np


def transport_indices(points, spacing, eye_size, old_crop, new_crop):
    from scipy.spatial import cKDTree
    old_crop, new_crop = np.asarray(old_crop), np.asarray(new_crop)
    world = new_crop[:2] + points / eye_size * (new_crop[2:] - new_crop[:2])
    source = (world - old_crop[:2]) / (old_crop[2:] - old_crop[:2]) * eye_size
    distance, index = cKDTree(points).query(source)
    valid = distance <= .6 * spacing
    return index, valid


def make_candidate(torch, base_class):
    class Candidate(base_class):
        def __init__(self, in_dim, points, eye_size, neighbors, mode):
            super().__init__(in_dim, points, eye_size, neighbors)
            self.mode = mode
            if mode == 'foreground':
                self.background_gain = torch.nn.Parameter(torch.zeros(()))
            if mode == 'adaptive':
                self.time_bias = torch.nn.Parameter(torch.zeros(()))
                self.time_slope = torch.nn.Parameter(torch.zeros(()))
            if mode == 'size':
                self.size_head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 2))
                torch.nn.init.zeros_(self.size_head[-1].weight)
                torch.nn.init.zeros_(self.size_head[-1].bias)

        def exemplar(self, features, mask):
            fg = super().exemplar(features, mask)
            if self.mode != 'foreground':
                return fg
            # Outside a one-neighbour dilated observed box: context, NOT labelled pure background.
            outside = ~mask[:, self.neighbors].any(-1)
            valid = outside.any(-1)
            safe = outside.clone()
            safe[~valid, 0] = True
            return fg, super().exemplar(features, safe), valid

        def logits(self, exemplar, features):
            if self.mode != 'foreground':
                return super().logits(exemplar, features)
            fg, bg, valid = exemplar
            a = super().logits(fg, features)
            b = super().logits(bg, features)[:, :721]
            # Relative context response. Does not use current labels or boxes.
            b = b - b.mean(-1, keepdim=True)
            gain = self.background_gain.clamp(0, 2)
            columns = a[:, :721] - gain * valid[:, None] * b
            return torch.cat((columns, a[:, 721:]), -1)

        def size_delta(self, exemplar, features, logits):
            template, valid = exemplar
            search = self.embed(features)
            weights = logits[:, :721].softmax(-1).detach()
            obj = (weights[..., None] * search).sum(1)
            ref = (template * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            return np.log(2.) * self.size_head(torch.cat((obj, ref), -1)).tanh()

    return Candidate


def run_candidate(torch, backbone, readout, hexes, masks, crops, geom, eye_size):
    """Returns logits and optional log-size residuals, with no ground-truth inputs."""
    state = backbone.init(hexes[:, :1]); last = hexes[:, 0]
    with torch.no_grad():
        for j in range(1, 4):
            state, feats = backbone.step(state, last, hexes[:, j]); last = hexes[:, j]
    ex = readout.exemplar(feats.detach(), masks)
    previous = None
    outputs, sizes = [], []
    for j in range(4, hexes.shape[1]):
        with torch.no_grad():
            state, current = backbone.step(state, last, hexes[:, j]); last = hexes[:, j]
        current = current.detach()
        if readout.mode in ('fixed', 'adaptive') and previous is not None:
            maps = [transport_indices(geom.points, geom.spacing, eye_size, c[j-1], c[j]) for c in crops]
            indices = torch.as_tensor(np.stack([m[0] for m in maps]), device=current.device)
            valid = torch.as_tensor(np.stack([m[1] for m in maps]), device=current.device)
            aligned = previous.gather(2, indices[:, None].expand(-1, current.shape[1], -1))
            if readout.mode == 'adaptive':
                diff = ((current - aligned).abs() / readout.sd[None, :, None]).mean(1, keepdim=True).clamp(max=10)
                alpha = (readout.time_bias + readout.time_slope * diff).sigmoid()
            else:
                alpha = .5
            current = torch.where(valid[:, None], alpha * current + (1-alpha) * aligned, current)
        previous = current
        logits = readout.logits(ex, current)
        outputs.append(logits)
        if readout.mode == 'size':
            sizes.append(readout.size_delta(ex, current, logits))
    return torch.stack(outputs, 1), torch.stack(sizes, 1) if sizes else None


def box_iou(center, size, target):
    center, size, target = map(lambda x: np.asarray(x, dtype=float), (center, size, target))
    lo, hi = center - size / 2, center + size / 2
    inter = np.maximum(0, np.minimum(hi, target[2:]) - np.maximum(lo, target[:2])).prod()
    union = size.prod() + np.maximum(0, target[2:] - target[:2]).prod() - inter
    return float(inter / max(union, 1e-12))
