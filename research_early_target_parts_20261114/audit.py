from common import *
import numpy as np,cv2
r=read(OUT/'result.json');old=ROOT/'research_mask_propagation_20261114';truth=np.load(old/'evaluation_truth.npz');masks=np.load(OUT/'predictions.npz');old1=np.load(old/'predictions.npz');old2=np.load(old/'memory_predictions.npz')
for row in r['records']:
 s,mode,fr=row['source'],row['mode'],row['frame'];key=f'{s}__{mode}__{fr}';mask=masks[key];t=truth[str(fr)];k=t>=0;cf=np.bincount(2*(t[k]==1).astype('int64')+mask[k],minlength=4);tn,fp,fn,tp=map(int,cf);assert (row['tp'],row['fp'],row['fn'])==(tp,fp,fn);assert row['iou']==tp/(tp+fn+fp)
 if s=='late':assert np.array_equal(mask,(old1['sam__'+str(fr)] if mode=='ordinary' else old2['initial_memory_only__'+str(fr)]))
for name,v in r['means'].items():
 a=[x['iou'] for x in r['records'] if name==x['source']+'__'+x['mode']];assert len(a)==5 and abs(sum(a)/5-v)<1e-14
assert len(r['records'])==40 and len(masks.files)==40;part=np.load(OUT/'prior_parts.npz');assert part['front_equal'].sum()==part['rear_equal'].sum() and not np.any(part['front_equal'] & part['rear_equal'])
save('audit.json',dict(passed=True,separate_confusion_matrices_checked=40,late_replay_checks=10,group_means=8,matched_pixels=int(part['front_equal'].sum()),result_sha256=sha(OUT/'result.json'),input_hashes_unchanged=all(sha(ROOT/n)==h for n,h in read(OUT/'protocol.json')['inputs'].items())))
