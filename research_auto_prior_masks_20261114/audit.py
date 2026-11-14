from common import *
import cv2,numpy as np
p=read(OUT/'protocol.json');r=read(OUT/'result.json');e=read(OUT/'execution_result.json');m=np.load(OUT/'predictions.npz');all_y=np.load(OUT/'all_teacher_masks.npz');prompts=np.load(OUT/'prompts.npz')
for c in p['cases']:
 truth=np.load(ROOT/c['truth'])
 for row in [v for v in r['records'] if v['case']==c['id']]:
  fr=row['frame'];t=truth[c['truth_prefix']+str(fr)];k=t>=0;v=m[f"{c['id']}__{row['source']}__{fr}"];tn,fp,fn,tp=np.bincount(2*(t[k]==1).astype(int)+v[k],minlength=4);assert (fp,fn,tp)==(row['fp'],row['fn'],row['tp']);assert (row['iou'] is None)==(tp+fn==0)
for sel in e['selections']:
 b=sel['source_box'];vals=[]
 for a in sel['boxes']:
  inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));vals.append(inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))
 assert np.allclose(vals,sel['box_iou'],atol=2e-7);i=max(range(len(vals)),key=vals.__getitem__) if vals and max(vals)>0 else None;assert i==sel['selected']
 if i is not None:assert np.array_equal(all_y[sel['key']+'__'+str(i)],prompts[sel['key']+'__yolo'])
n=cv2.fillPoly(np.zeros((720,1280),'uint8'),[np.rint(np.array([[867,185],[1135,181],[1135,266],[864,267]])/3+[900,200]).astype('int32')],1).astype(bool)
save('audit.json',dict(passed=True,confusion_matrices=len(r['records']),selections=len(e['selections']),neighbor_1171={s:int((n&m['old2179__'+s+'__1171']).sum()) for s in ['manual','yolo','box']},negative_core_pixels=int(n.sum()),result_sha256=sha(OUT/'result.json')))
