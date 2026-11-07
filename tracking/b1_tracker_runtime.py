"""Б1: сохранённая база и временная поправка внутри первого назначения.

Модуль получает только уже наблюдённые изображения, детекции и состояние
трекера. Разметка и расписание размеченных кадров здесь отсутствуют.
"""
from collections import deque
from pathlib import Path
import numpy as np
from ultralytics.trackers.basetrack import BaseTrack, TrackState
from ultralytics.trackers.utils import matching
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.b1_appearance import appearance_features, observe_tag, gate_features
from tracking.b1_appearance_fast import relative_features
from tracking.b1_learning import apply_gate_policy, categorical_cost

ROOT = Path(__file__).resolve().parents[1]


class CausalMasa:
    """Общий неизменный извлекатель; состояния и решения вариантов раздельны."""
    def __init__(self, model):
        self.model = model
        self.frame = -1
        self.images, self.detections, self.vectors = {}, {}, {}
        self.feature_frame = None
        self.feature_maps = None
        self.encoded_frames = 0

    def start(self, frame, image, boxes, confidence):
        assert frame > self.frame
        self.frame = int(frame)
        self.images[frame] = image
        self.detections[frame] = np.asarray(boxes)[np.asarray(confidence) >= .25].copy()
        for f in list(self.images):
            if f < frame - 4:
                del self.images[f]
                del self.detections[f]
        self.vectors = {k: v for k, v in self.vectors.items() if k[0] >= frame - 4}

    def embed(self, frame, boxes):
        import torch
        from tracking.masa_r50 import preprocess
        assert frame <= self.frame and frame in self.images
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        keys = [(int(frame), *b.tolist()) for b in boxes]
        missing = list(dict.fromkeys(k for k in keys if k not in self.vectors))
        if missing:
            with torch.no_grad():
                if self.feature_frame != frame:
                    image, self.scale = preprocess(self.images[frame])
                    self.feature_maps = self.model.features(image)
                    self.feature_frame = frame
                    self.encoded_frames += 1
                for left in range(0, len(missing), 16):
                    group = missing[left:left + 16]
                    b = torch.tensor([k[1:] for k in group], dtype=torch.float32, device='cuda')
                    z = torch.nn.functional.normalize(self.model.embed_boxes(self.feature_maps, b, self.scale), dim=1)
                    # Именно сохранённая точность старого банка, а не FP16-сеть.
                    z = z.cpu().numpy().astype(np.float16).astype(np.float32)
                    assert np.isfinite(z).all()
                    self.vectors.update(zip(group, z))
        return np.stack([self.vectors[k] for k in keys]) if keys else np.zeros((0, 256), np.float32)


def geometry_inputs(geom, crop, prediction, detections, cls, matrix, row, assignment):
    from tracking.z13_region_assignment import regions
    from tracking.z13_region_assignment_context import assignment_state_context
    choices, masks, first = regions(geom, crop, prediction,
        [d.xyxy for d in detections], [d.score for d in detections], [d.cls for d in detections], cls)
    context = assignment_state_context(matrix, row, choices, assignment)
    return choices, masks, np.concatenate([first, context], axis=1)


class FrozenBase:
    def __init__(self):
        import joblib
        folder = ROOT / 'runs/b1_real_inputs_20261026'
        self.identity = joblib.load(folder / 'identity_fold4.joblib')
        self.gate = joblib.load(folder / 'gate.joblib')
        with np.load(ROOT / 'runs/b123_20261027/b1_base_export_vectorized/recovery1/outputs/pca.npz') as z:
            self.mean, self.projection = z['mean'], z['projection']

    def probability(self, geometry, region, coarse, snapped, previous, gap):
        import tracking.b1_appearance as appearance
        # Только эквивалентная векторизация старых относительных признаков.
        appearance.relative_features = relative_features
        count = len(geometry)
        width = max(18, count)
        mask = np.arange(width) < count
        geo = np.zeros((width, geometry.shape[1]), np.float32)
        geo[:count] = geometry
        emb = np.zeros((width, 256), np.float32)
        emb[1:count] = region
        features, bank = appearance_features(geo, mask, emb, coarse, snapped,
                                             self.mean, self.projection, previous, gap)
        # Внешнее дополнение старых чанков различалось по split. Оно никогда
        # не подаётся identity-модели: проверяем и используем реальные строки.
        p = np.full(width, np.nan, np.float64)
        if count > 1:
            p[1:count] = self.identity.predict_proba(features[1:count])[:, 1]
        tag = observe_tag(geo, mask, emb, bank)
        return p, geo, mask, tag, features

    def decision(self, probability, geo, mask):
        features = gate_features(geo, mask, probability)
        valid = mask & (geo[:, -1] < .5)
        gate = np.full(len(mask), np.nan)
        if valid.any():
            gate[valid] = self.gate.predict_proba(features[valid])[:, 1]
        return apply_gate_policy(gate, geo, .5), gate


class ResidualEngine:
    def __init__(self, arm, bank, geom):
        import torch
        from tracking.z13_readout import FlyBackbone, make_modules
        from tracking.b1_learning import ReceptorHistory
        from tracking.flyvis_temporal_residual import TemporalResidualAssociation
        from tracking.b1_fast_stimulus import fast_init
        from tracking.b1_index_copy_stimulus import index_copy_step
        FlyBackbone.init = fast_init
        FlyBackbone.step = index_copy_step
        self.arm = arm
        ck = torch.load(ROOT / f'runs/b123_20261027/b1_training/{arm}/checkpoint.pt', map_location='cuda', weights_only=False)
        assert ck['arm'] == arm
        self.head = TemporalResidualAssociation().cuda().eval()
        self.head.load_state_dict(ck['head'])
        if arm in ('C', 'D'):
            # Отдельный estimator каждого FlyVis-варианта: D не меняет C.
            from scripts.run_bdd_video_rate_experiment import make_motion_estimator
            estimator = make_motion_estimator('flyvis_tiled', 30., .02)
            self.bb = FlyBackbone(estimator, trainable=False)
            self.bb.net.load_state_dict(ck['backbone'])
        elif arm == 'B':
            _, cls = make_modules(torch)
            self.bb = cls(geom.neighbors).cuda().eval()
            self.bb.load_state_dict(ck['backbone'])
            self.bb.dev = torch.device('cuda')
        else:
            assert arm == 'R1'
            self.bb = ReceptorHistory('cuda')
        self.head.requires_grad_(False)
        self.cursor = ck['cursor']

    def _check_candidate(self, state, current, activity=None, *, check_activity=False):
        """Reject invalid backend values before they enter the object's history.

        The official backends return new states. This does not undo arbitrary
        in-place writes by a replacement backend into the preceding state.
        """
        import torch
        from collections.abc import Mapping
        batch, columns = current.shape
        if self.arm in ('C', 'D'):
            values = getattr(getattr(state, 'nodes', None), 'activity', None)
            state_shape = (batch, self.bb.net.stimulus.n_nodes)
            channels = self.bb.index.shape[0]
        elif self.arm == 'B':
            values = state.get('h') if isinstance(state, Mapping) else None
            state_shape = (batch, self.bb.out.w.shape[1], columns)
            channels = self.bb.out.w.shape[0]
        else:
            values = state
            state_shape = (batch, self.bb.alpha.shape[1], columns)
            channels = 2 * self.bb.alpha.shape[1]
        if (not torch.is_tensor(values) or tuple(values.shape) != state_shape
                or values.device != current.device or not torch.isfinite(values).all()):
            raise ValueError(f'Expected finite recurrent state with shape {state_shape} on the observation device')
        if check_activity:
            shape = (batch, channels, columns)
            if (not torch.is_tensor(activity) or tuple(activity.shape) != shape
                    or activity.device != current.device or not torch.isfinite(activity).all()):
                raise ValueError(f'Expected finite recurrent activity with shape {shape} on the observation device')

    def start(self, history):
        import torch
        if len(history) != 4:
            raise ValueError('Historical residual engine requires exactly four observations')
        hx = torch.stack([h['hex'] for h in history])[None]
        if hx.ndim != 3 or not hx.shape[-1] or not torch.isfinite(hx).all():
            raise ValueError('Four finite nonempty receptor vectors required')
        with torch.no_grad():
            state = self.bb.init(hx[:, :1])
            self._check_candidate(state, hx[:, 0])
            for i in range(1, 4):
                state, reference = self.bb.step(state, hx[:, i - 1], hx[:, i])
                self._check_candidate(state, hx[:, i], reference, check_activity=True)
        return dict(state=state, last=hx[:, -1], reference=reference)

    def advance(self, eye, current):
        import torch
        previous = eye['last']
        if (not torch.is_tensor(current) or current.ndim != 1
                or previous.ndim != 2 or tuple(current.shape) != tuple(previous.shape[1:])
                or current.device != previous.device or current.dtype != previous.dtype
                or not torch.isfinite(previous).all() or not torch.isfinite(current).all()):
            raise ValueError('Matching finite receptor observations on the same device and dtype required')
        observed = current[None]
        with torch.no_grad():
            state, activity = self.bb.step(eye['state'], previous, observed)
            self._check_candidate(state, observed, activity, check_activity=True)
        eye.update(state=state, current=activity, last=observed)

    def correction(self, eye, masks, reference_mask, probability, zero=False):
        import torch
        from tracking.b1_real_inputs import corrected_identity_probability
        count = len(masks)
        p = torch.as_tensor(probability[:count][None], device='cuda', dtype=torch.float64)
        real = torch.ones_like(p, dtype=torch.bool)
        real[:, 0] = False
        with torch.no_grad():
            safe = torch.nan_to_num(p, nan=.5).clamp(1e-8, 1 - 1e-8).float()
            _, delta = self.head(torch.logit(safe), eye['reference'], eye['current'],
                torch.as_tensor(reference_mask[None], device='cuda'),
                torch.as_tensor(masks[None], device='cuda'),
                torch.ones_like(real), torch.tensor([True], device='cuda'))
            if zero:
                delta = torch.zeros_like(delta)
            updated = corrected_identity_probability(p, delta.double(), real)[0].cpu().numpy()
            self.last_pooled = dict(
                reference=self.head.pool(eye['reference'], torch.as_tensor(reference_mask[None, None], device='cuda'))[0, 0].cpu().tolist(),
                current=self.head.pool(eye['current'], torch.as_tensor(masks[None], device='cuda'))[0].cpu().tolist())
        if zero:
            assert np.array_equal(updated, probability[:count], equal_nan=True)
        assert np.max(np.abs(delta.cpu().numpy())) <= 1
        return updated, delta[0].cpu().numpy()


class B1Tracker(XYWHByteTrack):
    """Все вмешательства меняют настоящую матрицу до LAP и следующие состояния."""
    def __init__(self, args, bank=None, geom=None, server=None, base=None, engine=None,
                 disabled=False, zero=False):
        super().__init__(args)
        self.bank, self.geom, self.server, self.base, self.engine = bank, geom, server, base, engine
        self.disabled, self.zero = disabled, zero
        self.hist, self.eyes, self.previous = {}, {}, {}
        self._first = False
        self._id_count = 0
        self.events = []

    def update(self, *args, **kwargs):
        external = BaseTrack._count
        BaseTrack._count = self._id_count
        try:
            return super().update(*args, **kwargs)
        finally:
            self._id_count = BaseTrack._count
            BaseTrack._count = external

    def begin(self, frame, eye_frame):
        self.frame, self.eye_frame = int(frame), eye_frame
        self.size = (eye_frame.shape[1], eye_frame.shape[0])

    def after(self):
        if self.disabled:
            return
        from tracking.foveated_eyes import crop_box
        import torch
        tracks = [t for t in self.tracked_stracks if t.frame_id == self.frame_id]
        for t in self.tracked_stracks:
            self.eyes.pop(int(t.track_id), None)
        if not tracks:
            return
        crops = [crop_box(t.mean[:4], self.size) for t in tracks]
        with torch.no_grad():
            rendered = self.bank._render(self.eye_frame, crops)[:, 0].half().float()
        for t, crop, hx in zip(tracks, crops, rendered):
            self.hist.setdefault(int(t.track_id), deque(maxlen=4)).append(
                dict(frame=self.frame, box=np.asarray(t.xyxy).copy(), crop=crop, hex=hx))
        alive = {int(t.track_id) for t in self.tracked_stracks + self.lost_stracks}
        for store in (self.hist, self.previous):
            for tid in set(store) - alive:
                del store[tid]

    def _first_association(self, *args):
        self._first = True
        try:
            return super()._first_association(*args)
        finally:
            self._first = False

    def get_dists(self, tracks, detections):
        matrix = super().get_dists(tracks, detections)
        if self.disabled or not self._first or not len(tracks):
            return matrix
        from tracking.foveated_eyes import crop_box, box_to_crop_frame
        from tracking.z13_readout import to_frame
        from scripts.z13_masa_exemplar_snap_20261026 import iou
        import torch
        self.size = (self.eye_frame.shape[1], self.eye_frame.shape[0])
        assignment = np.full(len(tracks), -1, int)
        if len(detections):
            matches, _, _ = matching.linear_assignment(matrix, thresh=self.args.match_thresh)
            for i, j in matches:
                assignment[i] = j
        answer = matrix.copy()
        lost = [i for i, t in enumerate(tracks) if t.state != TrackState.Tracked and int(t.track_id) in self.hist]
        alive = {int(tracks[i].track_id) for i in lost}
        for tid in set(self.eyes) - alive:
            del self.eyes[tid]
        for i in lost:
            track = tracks[i]
            tid = int(track.track_id)
            if tid not in self.eyes:
                h = list(self.hist[tid]); h = [h[0]] * (4 - len(h)) + h[-4:]
                ref = h[-1]
                assert ref['frame'] < self.frame
                mask = self.geom.mask_in_box(box_to_crop_frame(ref['box'], ref['crop'], self.bank.eye_size))
                pts = self.geom.points[mask]
                a = to_frame(pts.min(0) - self.geom.spacing / 2, ref['crop'], self.geom.eye_size)
                b = to_frame(pts.max(0) + self.geom.spacing / 2, ref['crop'], self.geom.eye_size)
                coarse_box = np.array([*a, *b], np.float32)
                snapped = coarse_box.copy()
                det = self.server.detections[ref['frame']]
                if len(det):
                    overlaps = iou(coarse_box, det); best = int(overlaps.argmax())
                    if overlaps[best] >= .3:
                        snapped = det[best]
                embeddings = self.server.embed(ref['frame'], [coarse_box, snapped])
                eye = dict(ref_frame=ref['frame'], ref_mask=mask, coarse=embeddings[0], snapped=embeddings[1])
                if self.engine is not None:
                    eye.update(self.engine.start(h))
                self.eyes[tid] = eye
            eye = self.eyes[tid]
            crop = crop_box(track.mean[:4], self.size)
            if self.engine is not None:
                with torch.no_grad():
                    hx = self.bank._render(self.eye_frame, [crop])[:, 0].half().float()[0]
                    self.engine.advance(eye, hx)
            choices, masks, geo = geometry_inputs(self.geom, crop, track.mean[:2], detections, track.cls, matrix, i, assignment)
            current = self.server.embed(self.frame, [detections[j].xyxy for j in choices[1:]])
            old = self.previous.get(tid)
            gap = self.frame - old['frame'] if old is not None else 0
            probability, padded_geo, valid, tag, features = self.base.probability(geo, current, eye['coarse'], eye['snapped'], old, gap)
            original_p = probability.copy()
            delta = np.zeros(len(choices), np.float32)
            if self.engine is not None:
                probability[:len(choices)], delta = self.engine.correction(eye, masks, eye['ref_mask'], probability, self.zero)
            action, gate = self.base.decision(probability, padded_geo, valid)
            answer[i] = categorical_cost(matrix[i], choices, action)
            tag['frame'] = self.frame
            self.previous[tid] = tag
            assert eye['ref_frame'] < self.frame
            self.events.append(dict(frame=self.frame, track=tid, reference_frame=eye['ref_frame'],
                previous_query_frame=None if old is None else old['frame'], choices=choices, action=action,
                crop=list(crop), prediction=track.mean[:2].tolist(), track_class=int(track.cls),
                original=matrix[i].tolist(), updated=answer[i].tolist(),
                identity_base=original_p[:len(choices)].tolist(), identity=probability[:len(choices)].tolist(),
                delta=delta.tolist(), gate=gate[:len(choices)].tolist(), geometry=geo.tolist(),
                base_features=features[:len(choices)].tolist(),
                pooled=None if self.engine is None else self.engine.last_pooled))
        return answer
