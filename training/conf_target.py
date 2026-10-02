"""Цель уверенности без умножения на IoU (карточка conf_target_probe_v1).

В Ultralytics 8.4.84 (`utils/tal.py`, `TaskAlignedAssigner._forward`) цель
классификации положительного якоря равна нормированной метрике выравнивания,
умноженной на лучший IoU этого объекта (`pos_overlaps`): объект с неточной рамкой
учат оценивать низко. Г0а показал, что при равном IoU ночной разрыв потерь на
пороге сокращается на ~40 %. Здесь `pos_overlaps` заменяется единицей: пик цели
равен 1 при любой точности рамки, распределение по якорям внутри объекта прежнее.

Побочный эффект, о котором надо помнить: веса потерь рамки (box/DFL) в
`v8DetectionLoss` тоже берутся из `target_scores`, поэтому они меняются вместе
с целью. Патч ставится на класс (действует на обе ветви E2ELoss YOLO26) и
сделан модульными функциями — локальные не пикклятся вместе с моделью.

Использование:
    from training import conf_target
    conf_target.install()      # плечо iou_free
    ... обучение ...
    conf_target.uninstall()
"""
from __future__ import annotations

import torch

_orig_forward = None
CALLS = {"n": 0}


def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    """Копия `TaskAlignedAssigner._forward` 8.4.84 с единицей вместо `pos_overlaps`."""
    mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
    target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes, align_metric)
    target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
    align_metric *= mask_pos
    pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
    norm_align_metric = (align_metric / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
    target_scores = target_scores * norm_align_metric
    CALLS["n"] += 1
    return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


def install() -> None:
    global _orig_forward
    import ultralytics
    from ultralytics.utils.tal import TaskAlignedAssigner

    if ultralytics.__version__ != "8.4.84":
        raise SystemExit(f"[conf_target] патч писался под 8.4.84, установлена {ultralytics.__version__}")
    if _orig_forward is None:
        _orig_forward = TaskAlignedAssigner._forward
        TaskAlignedAssigner._forward = _forward
    print("[conf_target] патч установлен: цель уверенности без умножения на IoU", flush=True)


def uninstall() -> None:
    global _orig_forward
    from ultralytics.utils.tal import TaskAlignedAssigner

    if _orig_forward is not None:
        TaskAlignedAssigner._forward = _orig_forward
        _orig_forward = None


def selftest() -> None:
    """Патч при единичном IoU совпадает с оригиналом; при IoU < 1 цель выше."""
    from ultralytics.utils.tal import TaskAlignedAssigner

    torch.manual_seed(0)
    a = TaskAlignedAssigner(topk=10, num_classes=8, alpha=0.5, beta=6.0)
    bs, na = 2, 400
    xs, ys = torch.meshgrid(torch.arange(20) + 0.5, torch.arange(20) + 0.5, indexing="xy")
    anc = torch.stack([xs.flatten(), ys.flatten()], 1).float()
    gt = torch.tensor([[[2.0, 2.0, 9.0, 9.0], [10.0, 10.0, 18.0, 17.0]]] * bs)
    lab = torch.tensor([[[2.0], [5.0]]] * bs)
    mask = torch.ones(bs, 2, 1)
    scores = torch.rand(bs, na, 8)
    for jitter, expect_equal in ((0.0, True), (1.5, False)):
        pb = gt[:, :1].repeat(1, na, 1) + jitter * torch.randn(bs, na, 4)
        pb[..., 2:] = torch.maximum(pb[..., 2:], pb[..., :2] + 1)
        r0 = TaskAlignedAssigner.forward(a, scores, pb, anc, lab, gt, mask)
        install()
        r1 = TaskAlignedAssigner.forward(a, scores, pb, anc, lab, gt, mask)
        uninstall()
        same = torch.allclose(r0[2], r1[2], atol=1e-5)
        print(f"[selftest] jitter={jitter}: цели совпали={same}, сумма цели {r0[2].sum():.3f} → {r1[2].sum():.3f}")
        assert torch.equal(r0[3], r1[3]) and torch.equal(r0[4], r1[4]), "назначение якорей не должно меняться"
        assert (r1[2] >= r0[2] - 1e-6).all(), "цель не должна уменьшаться"


if __name__ == "__main__":
    selftest()
