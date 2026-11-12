from common import *
import ast
import numpy as np
import torch
from ultralytics.utils import nms
p=verify();result=read(OUT/'postprocess.json');ck=read(OUT/'checkpoint.json');assert result['passed'];case=p['cases'];path=ROOT/'runs/z13_val20_matched_20261024/cache'/case['video']/'detections/raw_video.npz'
with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
ids=np.flatnonzero(raw['frame_idx']==case['frame']);assert len(ids)==result['frame_detections'];order=sorted(ids.tolist(),key=lambda i:(-float(raw['conf'][i]),i));kept=[];suppressed=[]
def iou(a,b):
    it=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]));u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-it
    return it/u if u>0 else 0.
for rid in order:
    candidates=[(j,iou(raw['xyxy'][rid].astype(float),raw['xyxy'][j].astype(float))) for j in kept if raw['cls'][rid]==raw['cls'][j]];blockers=[(j,s) for j,s in candidates if s>p['fixed']['iou']]
    if blockers:suppressed.append(dict(raw_id=rid,suppressed_by=blockers[0][0],iou=blockers[0][1],score=float(raw['conf'][rid]),class_id=int(raw['cls'][rid])))
    else:kept.append(rid)
assert kept==result['kept_raw_ids'];pair_iou=iou(raw['xyxy'][52351].astype(float),raw['xyxy'][52356].astype(float));assert pair_iou>.7;assert next(x for x in suppressed if x['raw_id']==52356)['suppressed_by']==52351
x=np.column_stack([raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids]]).astype(np.float32);calls=[];old=nms.TorchNMS.nms
def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('Overlap suppression unexpectedly called')
nms.TorchNMS.nms=forbidden
try:y=nms.non_max_suppression(torch.from_numpy(x.copy())[None],conf_thres=.01,iou_thres=.7,end2end=ck['resolved_end2end'])[0].numpy()
finally:nms.TorchNMS.nms=old
assert not calls and np.array_equal(x,y);assert ck['model_yaml']['end2end'] is True and ck['has_one2one_cv2'] and ck['has_one2one_cv3']
tree=ast.parse((OUT/'archived_detection_cache.py').read_text(encoding='utf8'));func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_detection_cache');predict=[]
for node in ast.walk(func):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='predict':predict.append(dict(line=node.lineno,keywords=[k.arg for k in node.keywords],iou=ast.unparse(next(k.value for k in node.keywords if k.arg=='iou'))))
assert len(predict)==2 and all('end2end' not in x['keywords'] and 'nms' not in x['keywords'] and x['iou']=='iou' for x in predict)
current=ast.parse((ROOT/'scripts/run_bdd_video_rate_experiment.py').read_text(encoding='utf8'));cf=next(x for x in current.body if isinstance(x,ast.FunctionDef) and x.name=='build_detection_cache');route=[]
for node in ast.walk(cf):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='predict':route.append([k.arg for k in node.keywords])
assert route==[x['keywords'] for x in predict]
verify();save('audit.json',dict(passed=True,frame=case['frame'],counts=dict(original=len(ids),native_final_output=len(y),explicit_fixed_NMS=len(kept),suppressed=len(suppressed)),pair_iou=pair_iou,explicit_suppressed=suppressed,greedy_keep_exact=True,overlap_suppression_invocations=0,old_predict_calls=predict,current_predict_keywords_same=True,checkpoint_nms_free_structure_verified=True,input_hashes_unchanged=True,script_sha256=sha(__file__),limitations='Independent scalar greedy suppression uses saved final detections. The historical raw pre-postprocessing output and old vendor file hashes were not captured, so no claim of bitwise historical neural-forward replay. No safety or tracking gain is inferred from deleting21boxes.'))
print('AUDIT_PASS',len(ids),len(kept),'PAIR_IOU',pair_iou,'REMOVED_BY',next(x for x in suppressed if x['raw_id']==52356))
