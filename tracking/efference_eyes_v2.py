"""Transport the birth reference together with neural state (fly_efference_eye_v2).

The reference denotes the activity at birth in scene coordinates, not a scalar
decoder bias. Both states undergo exactly the same resampling, including newly
exposed boundary columns. Decode the transported reference before subtraction.
This is one explicit reference convention, not proof of full coordinate invariance.
"""
from __future__ import annotations

from tracking.efference_eyes import EfferenceEyeBank


class ReferencedEfferenceEyeBank(EfferenceEyeBank):
    def __init__(self, estimator, **kwargs):
        super().__init__(estimator, **kwargs)
        self._reference_activity = {}

    def _birth(self, boxes_xywh, frame):
        super()._birth(boxes_xywh, frame)
        for tid in boxes_xywh:
            self._reference_activity[tid] = self._activity[tid].clone()

    def _remap(self, tid, old_crop, new_crop):
        torch = self.torch
        current = self._activity[tid]
        reference = self._reference_activity[tid]
        mapped_current, mapped_reference = current.clone(), reference.clone()
        for group in self._groups:
            indices, weights = self._weights(group, old_crop, new_crop)
            indices = torch.as_tensor(indices, device=current.device)
            weights = torch.as_tensor(weights, device=current.device, dtype=current.dtype)
            nodes = group['nodes']
            for source, target in ((current, mapped_current), (reference, mapped_reference)):
                target[nodes] = (source[nodes][:, indices] * weights[None]).sum(-1)
        self._activity[tid] = mapped_current
        self._reference_activity[tid] = mapped_reference
        self._baseline[tid] = self._by_type(mapped_reference[None])
        if self.decode_flow:
            self._baseline_flow[tid] = self._decode(mapped_reference[None, None])[0].clone()

    def remove(self, track_ids):
        track_ids = list(track_ids)
        super().remove(track_ids)
        for tid in track_ids:
            self._reference_activity.pop(tid, None)
