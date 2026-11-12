from common import *
import numpy as np
import torch,ultralytics
from ultralytics.utils.nms import non_max_suppression,TorchNMS
p=verify();torch.set_num_threads(2);checkpoint=torch.load(ROOT/'weights/baseline_seed42/best.pt',map_location='cpu',weights_only=False);model=checkpoint.get('ema') or checkpoint['model'];head=model.model[-1]
# The forward methods are not invoked. Record both stored attributes and current properties.
metadata=dict(checkpoint_keys=sorted(checkpoint),checkpoint_date=checkpoint.get('date'),checkpoint_version=checkpoint.get('version'),loaded_class=type(model).__module__+'.'+type(model).__name__,head_class=type(head).__module__+'.'+type(head).__name__,stored_end2end=head.__dict__.get('_end2end'),stored_end2end_legacy=head.__dict__.get('end2end'),resolved_end2end=bool(head.end2end),module_names=list(head._modules),has_one2one_cv2='one2one_cv2' in head._modules,has_one2one_cv3='one2one_cv3' in head._modules,model_yaml=model.yaml,loaded_vendor_version=ultralytics.__version__,model_forward_calls=0,cuda_initialized=torch.cuda.is_initialized())
assert not torch.cuda.is_initialized();save('checkpoint.json',metadata)
case=p['cases'];path=ROOT/'runs/z13_val20_matched_20261024/cache'/case['video']/'detections/raw_video.npz'
with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
ids=np.flatnonzero(raw['frame_idx']==case['frame']);x=np.column_stack([raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids]]).astype(np.float32);tensor=torch.from_numpy(x.copy())
passed=non_max_suppression(tensor[None].clone(),conf_thres=p['fixed']['conf'],iou_thres=p['fixed']['iou'],end2end=bool(head.end2end))[0].numpy();assert np.array_equal(passed,x)
assert len(ids)<=300 and all(raw['conf'][ids]>.01)
offsets=torch.from_numpy(raw['cls'][ids].astype(np.float32))[:,None]*7680
kept=TorchNMS.nms(tensor[:,:4]+offsets,tensor[:,4],p['fixed']['iou']).numpy();keepids=ids[kept].tolist()
selected=[]
for rid in case['raw_ids']:
    j=int(np.flatnonzero(ids==rid)[0]);selected.append(dict(raw_id=rid,box=x[j,:4].tolist(),score=float(x[j,4]),cls=int(x[j,5]),preserved_by_final_output_branch=True,preserved_by_explicit_NMS=rid in keepids))
verify();save('postprocess.json',dict(passed=True,frame_detections=len(ids),final_output_count=len(passed),final_output_exact=True,explicit_NMS_count=len(kept),kept_raw_ids=keepids,removed_raw_ids=sorted(set(ids.tolist())-set(keepids)),selected=selected,scope='Input is the already saved final detection array, not an invented raw neural prediction. Idempotent final-output branch and explicit fixed-threshold suppression control only. Original detector network not executed.',script_sha256=sha(__file__)))
print('CHECKPOINT',metadata['head_class'],'END2END',metadata['resolved_end2end'],'STORED',metadata['stored_end2end'],'VERSION',metadata['checkpoint_version']);print('POSTPROCESS',len(ids),len(passed),len(kept),selected)
