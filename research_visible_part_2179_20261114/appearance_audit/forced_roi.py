"""Diagnostic-only ROI routing override, outside production source.

Parent supplies the already-built model and full, unchanged original ROI batch.
Only selection of the feature-pyramid level changes. Canvas, boxes, coordinates,
strides, sample ratio, alignment, embedding head, batch count/order are retained.
None exactly replicates the current natural route and must reproduce it first.
No model import, initialization or execution happens merely by loading this file.
"""

def embed_boxes_forced(model, feats, boxes, scale, forced_level=None):
    import torch
    from torchvision.ops import roi_align
    if forced_level is not None:
        assert type(forced_level) is int and 0 <= forced_level <= 3
    if not len(boxes):
        return torch.zeros((0,256),device=feats[0].device)
    b = boxes * scale
    area = ((b[:,2]-b[:,0])*(b[:,3]-b[:,1])).clamp(min=1e-6).sqrt()
    lvl = torch.floor(torch.log2(area/56+1e-6)).clamp(0,3).long()
    if forced_level is not None:
        lvl = torch.full_like(lvl,forced_level)
    out = torch.zeros((len(b),256,7,7),device=b.device,dtype=feats[0].dtype)
    for i,s in enumerate((4,8,16,32)):
        m = lvl == i
        if m.any():
            rois = torch.cat([torch.zeros((int(m.sum()),1),device=b.device,dtype=b.dtype),b[m]],1)
            out[m] = roi_align(feats[i],rois.to(feats[i].dtype),7,
                spatial_scale=1.0/s,sampling_ratio=0,aligned=True)
    return model.embed(out)
