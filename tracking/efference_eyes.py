"""Фовеальный глаз FlyVis с эфферентной копией перемещения вырезки (карточка fly_efference_eye_v1).

Действующий глаз (tracking/foveated_eyes.EyeBank) каждый кадр переносит вырезку к прогнозу рамки; изображение внутри
глаза тогда движется и от движения объекта, и от перемещения / изменения размера самой вырезки, а состояние нейронов
накоплено в координатах прежней вырезки. Здесь глаз использует известную геометрию вырезки (эфферентную копию
собственного перемещения) внутри пути мухи:
  * состояние — перед шагом активность каждого типа клеток переносится в координаты новой вырезки: точка колонки новой
    вырезки → кадр → поле прежней вырезки, барицентрическая интерполяция по колонкам того же типа (решётки 721 и 123
    колонки); колонки, вошедшие в поле с края, берут значение ближайшей колонки;
  * вход — прошлый стимул заново отрисовывается из прошлого кадра в текущей вырезке (а не берётся из прежней вырезки),
    поэтому перемещение и масштаб вырезки не создают видимого движения;
  * выход — остаточный декодированный поток (как у EyeBank) — движение изображения при неподвижной сцене-координате,
    т. е. движение объекта в кадре; перевод в px кадра — калибровкой 2 × 2 (scripts/fly_efference_eye.py).
Базовый отклик после разгона и всё остальное — как у EyeBank.
"""
from __future__ import annotations

import numpy as np

from tracking.foveated_eyes import EyeBank, crop_box


class EfferenceEyeBank(EyeBank):
    def __init__(self, estimator, **kwargs) -> None:
        super().__init__(estimator, **kwargs)
        from scipy.spatial import Delaunay, cKDTree
        c = self.network.connectome
        t = np.asarray(c.nodes.type[:]); u = np.asarray(c.nodes.u[:]); v = np.asarray(c.nodes.v[:])
        ref = t == t[0]   # полный тип (721 колонка) — порядок узлов совпадает с порядком рецепторов
        centers = estimator._local_centers_yx   # (721, 2) y, x относительно центра
        pos_of = {(int(a), int(b)): (float(cx) + self.eye_size // 2, float(cy) + self.eye_size // 2) for a, b, (cy, cx) in zip(u[ref], v[ref], centers)}
        node_pos = np.array([pos_of[(int(a), int(b))] for a, b in zip(u, v)], np.float64)   # (nodes, 2) x, y в px поля глаза
        self._groups = []   # решётки колонок: (матрица узлов типов × колонки, позиции колонок, триангуляция, дерево)
        by_set: dict[bytes, list] = {}
        for ty in np.unique(t):
            idx = np.flatnonzero(t == ty)
            key = node_pos[idx].round(3).tobytes()
            by_set.setdefault(key, []).append(idx)
        for key, idxs in by_set.items():
            P = node_pos[idxs[0]]
            for ix in idxs:
                assert np.allclose(node_pos[ix], P)
            self._groups.append({"nodes": self.torch.as_tensor(np.stack(idxs), device=self._dev()), "P": P, "tri": Delaunay(P), "tree": cKDTree(P)})
        self._prev_gray = None

    def _dev(self):
        return self.estimator.flyvis.device

    def _weights(self, g, old_crop, new_crop):
        """Барицентрические веса: колонка новой вырезки → позиция в поле прежней вырезки → три колонки прежней."""
        E = float(self.eye_size)
        ox1, oy1, ox2, oy2 = old_crop; nx1, ny1, nx2, ny2 = new_crop
        P = g["P"]
        X = nx1 + P[:, 0] * (nx2 - nx1) / E; Y = ny1 + P[:, 1] * (ny2 - ny1) / E
        Q = np.column_stack([(X - ox1) * E / max(ox2 - ox1, 1), (Y - oy1) * E / max(oy2 - oy1, 1)])
        s = g["tri"].find_simplex(Q)
        idx = np.zeros((len(Q), 3), np.int64); w = np.zeros((len(Q), 3), np.float64)
        inside = s >= 0
        if inside.any():
            T = g["tri"].transform[s[inside]]
            b = np.einsum("nij,nj->ni", T[:, :2, :], Q[inside] - T[:, 2, :])
            idx[inside] = g["tri"].simplices[s[inside]]; w[inside] = np.column_stack([b, 1 - b.sum(1)])
        if (~inside).any():
            _, nn = g["tree"].query(Q[~inside]); idx[~inside] = nn[:, None]; w[~inside] = [1.0, 0.0, 0.0]
        return idx, w

    def _remap(self, tid, old_crop, new_crop):
        torch = self.torch
        act = self._activity[tid]
        new = act.clone()
        for g in self._groups:
            idx, w = self._weights(g, old_crop, new_crop)
            A = act[g["nodes"]]   # (types, columns)
            it = torch.as_tensor(idx, device=act.device); wt = torch.as_tensor(w, device=act.device, dtype=act.dtype)
            new[g["nodes"]] = (A[:, it] * wt[None]).sum(-1)
        self._activity[tid] = new

    def step(self, frame, predictions_xywh):
        gray = self._gray(frame)
        ids = [tid for tid in predictions_xywh if tid in self._activity]
        if ids and self._prev_gray is not None:
            crops = {tid: crop_box(predictions_xywh[tid], (gray.shape[1], gray.shape[0]), context=self.context, min_side=self.min_side) for tid in ids}
            with self.gpu_lock:
                for tid in ids:
                    if crops[tid] != self.crops.get(tid):
                        self._remap(tid, self.crops[tid], crops[tid])
                hexes = self._render(self._prev_gray, [crops[t] for t in ids])   # прошлый кадр в текущей вырезке
            for row, tid in enumerate(ids):
                self._last_hex[tid] = hexes[row].clone(); self.crops[tid] = crops[tid]
        out = super().step(frame, predictions_xywh)
        self._prev_gray = gray
        return out

    def birth(self, frame, boxes_xywh):
        super().birth(frame, boxes_xywh)
        self._prev_gray = self._gray(frame)
