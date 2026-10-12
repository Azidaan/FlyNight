"""З13 (карточка fly_object_readout_z13_v1): объектное считывание FlyVis — найти образец цели в области глаза или отказаться.

Состав:
  * HexGeometry — колонки глаза (721) в пикселях поля глаза, 6 соседей, перевод «точка ↔ колонка»;
  * Readout — считыватель, одинаковый во всех вариантах: стандартизация признаков колонки, линейное вложение в 16 чисел,
    образец — среднее вложений колонок внутри рамки трека, балл колонки = κ·cos + β, «отказ» — обучаемый балл (722 варианта);
  * HexGRU — контроль B без FlyVis: свёрточный GRU по колонке и 6 соседям (вход 1, состояние 16) + свёртка 16 → 34;
  * FlyBackbone — FlyVis (flow/0000/000): fade-in без градиента, шаги с градиентом; frozen — штатные параметры,
    trainable — 734 свободных (nodes_bias, nodes_time_const, edges_syn_strength) с притяжением к штатному решению;
  * center_estimate — колонка-максимум и её 6 соседей (взвешенное среднее только по 7 колонкам);
  * Localizer — применение в трекере: глаз рождается при потере трека на 4 последних ведомых кадрах (образец),
    затем работает на каждом кадре потери на вырезке вокруг прогноза; решение «найден ≥ τ» → новый центр.
Модуль не читает разметку.
"""
from __future__ import annotations

import numpy as np

EMB = 16
HIDDEN = 16
N_EXEMPLAR = 4
FREE = ("nodes_bias", "nodes_time_const", "edges_syn_strength")
CORE_LR = {"nodes_bias": 1e-4, "nodes_time_const": 5e-6, "edges_syn_strength": 5e-6}   # как К1 v2 (scripts/fly_k1_v2.py)


class HexGeometry:
    """Колонки глаза: points (721, 2) — x, y в пикселях поля глаза eye_size × eye_size; neighbors (721, 7) — сама + 6 соседей."""

    def __init__(self, points_xy: np.ndarray, eye_size: int) -> None:
        from scipy.spatial import cKDTree
        self.points = np.asarray(points_xy, np.float32)
        self.eye_size = int(eye_size)
        tree = cKDTree(self.points)
        d, _ = tree.query(self.points, k=2)
        self.spacing = float(np.median(d[:, 1]))
        nb = np.tile(np.arange(len(self.points))[:, None], (1, 7))
        for i, p in enumerate(self.points):
            ids = [j for j in tree.query_ball_point(p, 1.2 * self.spacing) if j != i]
            ids = sorted(ids, key=lambda j: float(np.linalg.norm(self.points[j] - p)))[:6]
            nb[i, 1:1 + len(ids)] = ids   # у краевых колонок недостающие соседи — сама колонка
        self.neighbors = nb

    def nearest(self, xy_eye) -> int:
        return int(np.argmin(np.linalg.norm(self.points - np.asarray(xy_eye, np.float32)[None], axis=1)))

    def mask_in_box(self, box_eye) -> np.ndarray:
        x1, y1, x2, y2 = box_eye
        m = (self.points[:, 0] >= x1) & (self.points[:, 0] <= x2) & (self.points[:, 1] >= y1) & (self.points[:, 1] <= y2)
        if not m.any():   # рамка мельче шага колонки — ближайшая к центру колонка
            m[self.nearest(((x1 + x2) / 2, (y1 + y2) / 2))] = True
        return m


def to_eye(xy_frame, crop, eye_size: int) -> tuple[float, float]:
    """Точка кадра → пиксели поля глаза вырезки (как tracking/foveated_eyes.box_to_crop_frame)."""
    x1, y1, x2, y2 = crop
    return (float(xy_frame[0]) - x1) * eye_size / max(x2 - x1, 1), (float(xy_frame[1]) - y1) * eye_size / max(y2 - y1, 1)


def to_frame(xy_eye, crop, eye_size: int) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return x1 + float(xy_eye[0]) * max(x2 - x1, 1) / eye_size, y1 + float(xy_eye[1]) * max(y2 - y1, 1) / eye_size


def make_modules(torch):
    nn = torch.nn

    class Readout(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.register_buffer("mu", torch.zeros(in_dim)); self.register_buffer("sd", torch.ones(in_dim))
            self.emb = nn.Linear(in_dim, EMB)
            self.kappa = nn.Parameter(torch.tensor(5.0)); self.beta = nn.Parameter(torch.tensor(0.0)); self.absent = nn.Parameter(torch.tensor(0.0))

        def embed(self, feats):   # (B, D, 721) → (B, 721, EMB)
            z = (feats.transpose(1, 2) - self.mu) / self.sd
            return self.emb(z)

        def exemplar(self, feats, mask):   # mask (B, 721) bool → (B, EMB)
            e = self.embed(feats); m = mask.float().unsqueeze(-1)
            return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

        def logits(self, ex, feats):   # → (B, 722): 721 колонка + «отказ»
            f = self.embed(feats)
            cos = torch.nn.functional.cosine_similarity(f, ex.unsqueeze(1).expand_as(f), dim=-1, eps=1e-6)
            col = self.kappa * cos + self.beta
            return torch.cat([col, self.absent.expand(col.shape[0], 1)], dim=1)

    class HexConv(nn.Module):
        def __init__(self, cin: int, cout: int, neighbors: np.ndarray) -> None:
            super().__init__()
            self.register_buffer("nb", torch.as_tensor(neighbors, dtype=torch.long))
            self.w = nn.Parameter(torch.randn(cout, cin, 7) * (1.0 / np.sqrt(cin * 7))); self.b = nn.Parameter(torch.zeros(cout))

        def forward(self, x):   # (B, C, N) → (B, cout, N)
            g = x[:, :, self.nb]   # (B, C, N, 7)
            return torch.einsum("bcnk,ock->bon", g, self.w) + self.b[None, :, None]

    class HexGRU(nn.Module):
        """Контроль B: свёрточный GRU по колонке и 6 соседям, вход — изображение глаза (B, 1, 721)."""
        out_dim = 34

        def __init__(self, neighbors: np.ndarray) -> None:
            super().__init__()
            self.zr = HexConv(1 + HIDDEN, 2 * HIDDEN, neighbors); self.hc = HexConv(1 + HIDDEN, HIDDEN, neighbors); self.out = HexConv(HIDDEN, self.out_dim, neighbors)

        def init(self, hex0):   # нулевое состояние + шаг на первом кадре (у мухи на его месте fade-in)
            state = {"h": torch.zeros(hex0.shape[0], HIDDEN, hex0.shape[-1], device=hex0.device)}
            return self.step(state, hex0, hex0)[0]

        def step(self, state, last_hex, cur_hex):
            x = cur_hex.reshape(cur_hex.shape[0], 1, -1); h = state["h"]
            z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
            hn = torch.tanh(self.hc(torch.cat([x, r * h], 1)))
            h = (1 - z) * h + z * hn
            return {"h": h}, torch.tanh(self.out(h))

        def penalty(self):
            return torch.zeros((), device=self.out.w.device)

    return Readout, HexGRU


class FlyBackbone:
    """FlyVis в виде для градиента (как scripts/fly_finetune_k1.py:Fly.step) на вырезках одного глаза на пример."""

    out_dim = 34

    def __init__(self, estimator, trainable: bool) -> None:
        import torch
        from tracking.flyvis_activity import OUTPUT_CELL_TYPES, type_column_indices
        from tracking.foveated_eyes import _patch_constant_dt
        self.torch = torch; self.est = estimator; self.net = estimator.network
        _patch_constant_dt(self.net)
        self.dev = estimator.flyvis.device
        self.dt, self.steps = float(estimator.dt), int(estimator.integration_steps)
        self.fade_in_seconds = float(estimator.fade_in_seconds)
        idx = type_column_indices(self.net.connectome.nodes.type[:], OUTPUT_CELL_TYPES)
        self.index = torch.as_tensor(np.stack(list(idx.values())), device=self.dev)   # (34, 721)
        self.trainable = bool(trainable)
        self.params, self.p0 = [], []
        for name, p in self.net.named_parameters():
            p.requires_grad_(self.trainable and name in FREE)
            if self.trainable and name in FREE:
                self.params.append(p); self.p0.append(p.detach().clone())

    def param_groups(self):
        names = [n for n, _ in self.net.named_parameters() if n in FREE]
        return [{"params": [p], "lr": CORE_LR[n]} for n, p in zip(names, self.params)]

    def penalty(self):
        t = self.torch
        if not self.params:
            return t.zeros((), device=self.dev)
        return sum(((p - q) / (q.abs() + 0.1)).pow(2).mean() for p, q in zip(self.params, self.p0)) / len(self.params)

    def clamp(self):
        if self.params:
            with self.torch.no_grad():
                self.net.clamp()

    def init(self, hex0):   # (B, 1, 721) → состояние после fade-in (без градиента)
        t = self.torch; net = self.net
        steps = int(self.fade_in_seconds / self.dt); batch = hex0.shape[0]
        net.stimulus.zero(batch, steps)
        frames = t.linspace(0, 1, steps, device=hex0.device)[None, :, None] * (hex0.reshape(batch, 1, -1).repeat(1, steps, 1) - 0.5) + 0.5
        net.stimulus.add_input(frames[:, :, None])
        with t.no_grad():
            x = net.stimulus(); net.clamp(); params = net._param_api(); state = net._initial_state(params, batch)
            for i in range(x.shape[1]):
                state = net._next_state(params, state, x[:, i], self.dt)
        return state

    def step(self, state, last_hex, cur_hex):
        t = self.torch
        alpha = t.arange(1, self.steps + 1, device=self.dev, dtype=t.float32).view(1, self.steps, 1, 1) / self.steps
        stim = last_hex.reshape(-1, 1, 1, last_hex.shape[-1]) * (1 - alpha) + cur_hex.reshape(-1, 1, 1, cur_hex.shape[-1]) * alpha
        st = self.net.stimulus
        idx = t.as_tensor(np.asarray(st.input_index), device=self.dev)
        buf = t.zeros((stim.shape[0], stim.shape[1], st.n_nodes), device=self.dev, dtype=t.float32)
        buf[:, :, idx] = stim.expand(-1, -1, idx.shape[0], -1)
        states = self.net.forward(buf, self.dt, state, as_states=True)
        act = states[-1].nodes.activity   # (B, nodes) — последний шаг интегрирования
        return states[-1], act[:, self.index]   # (B, 34, 721)


class Localizer:
    """Применение в трекере (вариант B, C или D из runs/z13/<arm>/model.pt). Перед первым сопоставлением — before_associate
    (потерянные треки в пуле), после обновления — after_update (ведомые треки обновлённые на кадре: история образца).
    Состояние глаза потерянного трека хранится тензором (активность узлов мухи или скрытое состояние B)."""

    def __init__(self, ckpt_path, bank, frame_size=(1280, 720)) -> None:
        import torch
        from collections import deque
        self.torch, self.bank, self.frame_size, self.deque = torch, bank, frame_size, deque
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.arm = ck["arm"]; self.tau = (ck["tau"] or {}).get("tau")
        Readout, HexGRU = make_modules(torch)
        self.geom = HexGeometry(bank.points_local, bank.eye_size); self.dev = bank.estimator.flyvis.device
        if self.arm == "B":
            self.bb = HexGRU(self.geom.neighbors).to(self.dev); self.bb.load_state_dict(ck["backbone"])
        else:
            from scripts.run_bdd_video_rate_experiment import make_motion_estimator
            est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)   # как при обучении (scripts/z13_train.py:build)
            if self.arm == "D":
                est.network.load_state_dict({k: v.to(self.dev) for k, v in ck["network"].items()})
            self.bb = FlyBackbone(est, trainable=False)
        self.ro = Readout(self.bb.out_dim).to(self.dev); self.ro.load_state_dict(ck["readout"]); self.ro.eval()
        self.hist: dict = {}      # трек → последние ведомые (hex, рамка, вырезка)
        self.eyes: dict = {}      # трек → (состояние, последний hex, образец)
        self.stats = {"steps": 0, "interventions": 0, "refusals": 0, "births": 0}

    def _pack(self, state):
        return state["h"] if self.arm == "B" else state.nodes.activity

    def _unpack(self, x):
        if self.arm == "B":
            return {"h": x}
        from flyvis.utils.tensor_utils import AutoDeref
        net = self.bb.net
        return net._state_api(AutoDeref(nodes=AutoDeref(activity=x), edges=AutoDeref()))

    def after_update(self, tracker, eye_frame) -> None:
        from tracking.foveated_eyes import crop_box
        upd = [t for t in tracker.tracked_stracks if t.frame_id == tracker.frame_id]
        for t in tracker.tracked_stracks:
            self.eyes.pop(int(t.track_id), None)   # снова ведомый — глаз потери закрыт
        if not upd:
            return
        crops = [crop_box(tuple(float(v) for v in t.mean[:4]), self.frame_size) for t in upd]
        with self.torch.no_grad():
            hx = self.bank._render(eye_frame, crops)[:, 0]
        for t, crop, h in zip(upd, crops, hx):
            self.hist.setdefault(int(t.track_id), self.deque(maxlen=N_EXEMPLAR)).append((h, np.asarray(t.xyxy, np.float64), crop))

    def before_associate(self, pool, eye_frame, tracked_state) -> None:
        from tracking.foveated_eyes import box_to_crop_frame, crop_box
        torch = self.torch
        lost = [t for t in pool if t.state != tracked_state and int(t.track_id) in self.hist]
        alive = {int(t.track_id) for t in lost}
        for k in [k for k in self.eyes if k not in alive]:
            self.eyes.pop(k)
        if not lost:
            return
        new = [t for t in lost if int(t.track_id) not in self.eyes]
        with torch.no_grad():
            if new:   # рождение: fade-in на первом из 4 последних ведомых кадров, затем остальные; образец — на последнем
                H = []
                for t in new:
                    h = list(self.hist[int(t.track_id)]); h = [h[0]] * (N_EXEMPLAR - len(h)) + h[-N_EXEMPLAR:]; H.append(h)
                hexes = torch.stack([torch.stack([x[0] for x in h]) for h in H])   # (B, 4, 721)
                masks = torch.as_tensor(np.stack([self.geom.mask_in_box(box_to_crop_frame(h[-1][1], h[-1][2], self.bank.eye_size)) for h in H]), device=self.dev)
                st = self.bb.init(hexes[:, 0:1]); last = hexes[:, 0]; feats = None
                for j in range(1, N_EXEMPLAR):
                    st, feats = self.bb.step(st, last, hexes[:, j]); last = hexes[:, j]
                ex = self.ro.exemplar(feats, masks)
                x = self._pack(st)
                for r, t in enumerate(new):
                    self.eyes[int(t.track_id)] = (x[r].clone(), last[r].clone(), ex[r].clone())
                self.stats["births"] += len(new)
            ids = [int(t.track_id) for t in lost]
            crops = [crop_box(tuple(float(v) for v in t.mean[:4]), self.frame_size) for t in lost]
            cur = self.bank._render(eye_frame, crops)[:, 0]
            st = self._unpack(torch.stack([self.eyes[i][0] for i in ids])); last = torch.stack([self.eyes[i][1] for i in ids]); ex = torch.stack([self.eyes[i][2] for i in ids])
            st, feats = self.bb.step(st, last, cur)
            P = torch.softmax(self.ro.logits(ex, feats), -1).cpu().numpy()
            x = self._pack(st)
        for r, (t, i, crop) in enumerate(zip(lost, ids, crops)):
            self.eyes[i] = (x[r].clone(), cur[r].clone(), self.eyes[i][2])
            self.stats["steps"] += 1
            if self.tau is not None and 1.0 - float(P[r, 721]) >= self.tau:
                c = to_frame(center_estimate(P[r, :721], self.geom), crop, self.bank.eye_size)
                t.mean[0], t.mean[1] = float(c[0]), float(c[1]); self.stats["interventions"] += 1
            else:
                self.stats["refusals"] += 1


def center_estimate(probs_cols: np.ndarray, geom: HexGeometry) -> np.ndarray:
    """Колонка-максимум и её 6 соседей: взвешенное среднее только по этим 7 колонкам (два максимума не усредняются)."""
    k = int(np.argmax(probs_cols)); nb = np.unique(geom.neighbors[k])
    w = probs_cols[nb]
    return (geom.points[nb] * w[:, None]).sum(0) / max(float(w.sum()), 1e-12)


def run_sequence(torch, backbone, readout, hexes, masks):
    """Один проход эпизода: hexes (B, T, 721) — 4 кадра образца + кадры поиска; masks (B, 721) — колонки рамки на последнем кадре образца.
    Возвращает логиты (B, T − 4, 722) для каждого кадра поиска."""
    state = backbone.init(hexes[:, 0:1]); last = hexes[:, 0]
    feats = None
    for j in range(1, N_EXEMPLAR):
        state, feats = backbone.step(state, last, hexes[:, j]); last = hexes[:, j]
    if feats is None:
        state, feats = backbone.step(state, last, hexes[:, 0])
    ex = readout.exemplar(feats, masks)
    out = []
    for j in range(N_EXEMPLAR, hexes.shape[1]):
        state, feats = backbone.step(state, last, hexes[:, j]); last = hexes[:, j]
        out.append(readout.logits(ex, feats))
    return torch.stack(out, 1)
