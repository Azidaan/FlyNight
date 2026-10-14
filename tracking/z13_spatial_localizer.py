"""Closed-loop application of frozen Z13 v2 models; no annotation inputs."""
from collections import deque
import numpy as np

from tracking.z13_readout import (Localizer, HexGeometry, FlyBackbone, make_modules,
                                  N_EXEMPLAR, to_frame, center_estimate)
from tracking.z13_spatial import make_spatial_readout, make_quality_head, quality_features


def training_integration():
    from tracking.flyvis_motion import integration_plan
    return integration_plan(30., .02)


def percentile_threshold(scores, fraction=.01):
    """Top fraction including all ties, using observed scores only."""
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite nonempty scores required')
    if not 0 < fraction <= 1:
        raise ValueError('Invalid fraction')
    rank = max(1, int(np.ceil(fraction * len(scores))))
    threshold = float(np.sort(scores)[::-1][rank - 1])
    return {'tau': threshold, 'target_rank': rank,
            'selected': int((scores >= threshold).sum()), 'total': len(scores)}


class SpatialLocalizer(Localizer):
    def __init__(self, ckpt_path, bank, frame_size=(1280, 720)):
        import torch
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ck.get('format') != 'z13-spatial-hota-policy-v1':
            raise ValueError('Explicit spatial HOTA policy checkpoint required')
        self.torch, self.bank, self.frame_size, self.deque = torch, bank, frame_size, deque
        self.arm, self.tau = ck['arm'], ck['tau']['tau']
        self.geom = HexGeometry(bank.points_local, bank.eye_size)
        self.dev = bank.estimator.flyvis.device
        _, HexGRU = make_modules(torch)
        if self.arm == 'B':
            self.bb = HexGRU(self.geom.neighbors).to(self.dev)
            self.bb.load_state_dict(ck['backbone']); self.bb.eval()
        elif self.arm in ('C', 'D'):
            bank.estimator.network.load_state_dict(ck['backbone'])
            self.bb = FlyBackbone(bank.estimator, trainable=False)
            # Episodes were simulated at 30 Hz. The environment's retina still
            # uses the video's measured rate, exactly as in the original stand.
            self.bb.steps, self.bb.dt = training_integration()
        else:
            raise ValueError('Unknown arm')
        self.ro = make_spatial_readout(torch)(34, self.geom.points, bank.eye_size,
                                             self.geom.neighbors).to(self.dev)
        self.ro.load_state_dict(ck['readout']); self.ro.eval()
        if not torch.equal(self.ro.neighbors.cpu(), torch.as_tensor(self.geom.neighbors, device='cpu')):
            raise ValueError('Checkpoint geometry differs from runtime')
        expected_xy = torch.as_tensor(self.geom.points / bank.eye_size * 2 - 1, device='cpu')
        if not torch.equal(self.ro.xy.cpu(), expected_xy):
            raise ValueError('Checkpoint coordinates differ from runtime')
        self.quality = make_quality_head(torch)().cpu()
        self.quality.load_state_dict(ck['quality']); self.quality.eval()
        self.hist, self.eyes = {}, {}
        self.stats = {'steps': 0, 'interventions': 0, 'refusals': 0, 'births': 0,
                      'nonfinite_refusals': 0}
        self.max_batch = 4

    def _new(self, tracks):
        from tracking.foveated_eyes import box_to_crop_frame
        torch = self.torch
        histories = []
        for t in tracks:
            h = list(self.hist[int(t.track_id)])
            histories.append([h[0]] * (N_EXEMPLAR - len(h)) + h[-N_EXEMPLAR:])
        hexes = torch.stack([torch.stack([x[0] for x in h]) for h in histories])
        # Training episodes were saved in fp16, then loaded into fp32.
        hexes = hexes.half().float()
        masks = torch.as_tensor(np.stack([self.geom.mask_in_box(box_to_crop_frame(
            h[-1][1], h[-1][2], self.bank.eye_size)) for h in histories]), device=self.dev)
        state = self.bb.init(hexes[:, 0:1]); last = hexes[:, 0]
        for j in range(1, N_EXEMPLAR):
            state, feats = self.bb.step(state, last, hexes[:, j]); last = hexes[:, j]
        ex, valid = self.ro.exemplar(feats, masks)
        packed = self._pack(state)
        for row, t in enumerate(tracks):
            self.eyes[int(t.track_id)] = (packed[row].clone(), last[row].clone(),
                                           (ex[row].clone(), valid[row].clone()))
        self.stats['births'] += len(tracks)

    def before_associate(self, pool, eye_frame, tracked_state):
        from tracking.foveated_eyes import crop_box
        torch = self.torch
        lost = [t for t in pool if t.state != tracked_state and int(t.track_id) in self.hist]
        alive = {int(t.track_id) for t in lost}
        for key in [key for key in self.eyes if key not in alive]:
            self.eyes.pop(key)
        with torch.no_grad():
            new = [t for t in lost if int(t.track_id) not in self.eyes]
            for left in range(0, len(new), self.max_batch):
                self._new(new[left:left + self.max_batch])
            for left in range(0, len(lost), self.max_batch):
                group = lost[left:left + self.max_batch]
                ids = [int(t.track_id) for t in group]
                predictions = [np.asarray(t.mean[:2], np.float64).copy() for t in group]
                crops = [crop_box(tuple(float(v) for v in t.mean[:4]), self.frame_size) for t in group]
                current = self.bank._render(eye_frame, crops)[:, 0].half().float()
                state = self._unpack(torch.stack([self.eyes[i][0] for i in ids]))
                last = torch.stack([self.eyes[i][1] for i in ids])
                exemplar = (torch.stack([self.eyes[i][2][0] for i in ids]),
                            torch.stack([self.eyes[i][2][1] for i in ids]))
                state, feats = self.bb.step(state, last, current)
                p = self.ro.logits(exemplar, feats).softmax(-1).cpu().numpy()
                packed = self._pack(state)
                for row, (t, tid, crop, prediction) in enumerate(zip(group, ids, crops, predictions)):
                    self.eyes[tid] = (packed[row].clone(), current[row].clone(), self.eyes[tid][2])
                    self.stats['steps'] += 1
                    if not np.isfinite(p[row]).all():
                        self.stats['refusals'] += 1; self.stats['nonfinite_refusals'] += 1
                        continue
                    eye = center_estimate(p[row, :721], self.geom)
                    center = np.asarray(to_frame(eye, crop, self.bank.eye_size))
                    features = quality_features(p[row], self.geom, eye, center, prediction, crop)
                    score = float(self.quality(torch.from_numpy(features)[None]).sigmoid()[0])
                    if np.isfinite(score) and self.tau is not None and score >= self.tau:
                        t.mean[:2] = center
                        self.stats['interventions'] += 1
                    else:
                        self.stats['refusals'] += 1

