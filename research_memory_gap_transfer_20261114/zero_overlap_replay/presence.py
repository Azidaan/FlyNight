from pathlib import Path
import hashlib, json, sys
D=Path(__file__).resolve().parent
P=D.parent
sys.path.insert(0,str(P))
from common import ROOT,now,sha,rel
def write(name,x):
    with (D/name).open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def rd(x):return json.loads(x.read_bytes())
src=P/'run.py'
prev=ROOT/'research_ownership_conflict_2179_20261114/twenty_gap_outcomes.json'
ids=[x['case'] for x in rd(prev)['episodes'] if x['reason']=='zero_overlap_all_candidates']
assert len(ids)==4
root_base=ROOT/'tools/sam2-runtime-20261027/sam2/modeling/sam2_base.py';root_pred=ROOT/'tools/sam2-runtime-20261027/sam2/sam2_video_predictor.py'
if sys.argv[1]=='prepare':
    write('protocol_scores.json',dict(registered=now(),inputs={rel(f):sha(f) for f in [src,P/'protocol.json',P/'actions.json',prev,Path(__file__),root_base,root_pred]},cases=ids,scope='Exact replay of ordinary and fixed registered masks on all four zero-overlap target rows; CPU independent pixel counts per frame and equality to original overlaps and actions. Distinguish empty predictions from nonempty disjoint ones; record native per-object presence score and range of returned mask before binarization, without changing inference or registered thresholds.'))
    sys.exit()
proto=rd(D/'protocol_scores.json')
assert proto['cases']==ids and all(sha(ROOT/f)==h for f,h in proto['inputs'].items())
original=src.read_text(encoding='utf8');a=original.index('with GpuLock');b=original.index("with (OUT/'predictions.npz')")
loop=original[a:b].replace(' for c in cases:'," for c in cases:\n  if c['id'] not in audit_ids:continue")
tag="   traces.append(dict(case=cid,frame=k"
assert loop.count(tag)==1
loop=loop.replace(tag,"""   if k==N:
    for mode in ['fixed','ordinary']:
     mm=priors[mode][0][k]
     raw=method[mode][0]
     oo=oracle[0]; correct=np.flatnonzero(oo>.5); selected=int(correct[np.argmax(oo[correct])])
     details.append(dict(case=cid,mode=mode,raw_predictions_by_frame=[int(priors[mode][0][i].sum()) for i in range(N+1)],target_pixels=int(g[0][0].sum()),final_prediction_pixels=int(mm.sum()),pixels_on_official_target=int((mm&g[0][0]).sum()),pixels_on_other_evaluated_cars=int((mm&g[0][2]).sum()),pixels_on_all_current_masks=int((mm&np.logical_or.reduce(masks_now)).sum()),current_mask_pixels=int(masks_now[selected].sum()),current_target_iou=float(oo[selected]),all_pairs_equal=bool(np.array_equal(method[mode],np.array(saved_trace[(cid,k)]['scores'][mode])))))
"""+tag)
key="     m0[k]=(l[0,0]>0).cpu().numpy()"
assert loop.count(key)==1
loop=loop.replace(key,key+"""
     if len(prior)==0 and k==N:
      out=st['output_dict_per_obj'][0]['non_cond_frame_outputs'][k]
      gate_rows.append(dict(case=cid,mode=mode,score=float(out['object_score_logits'].item()),minimum=float(l.min().item()),maximum=float(l.max().item()),pred_obj_scores=bool(vp.pred_obj_scores),runtime_class=type(vp).__name__))
""")
base=rd(P/'actions.json');saved_trace={(x['case'],x['frame']):x for x in base['traces']}
env=dict(gate_rows=[],audit_ids=ids,details=[],saved_trace=saved_trace)
exec(compile(original[:a]+loop,str(src),'exec'),env)
for x in env['traces']:
    assert json.loads(json.dumps(x))==saved_trace[x['case'],x['frame']]
rows=env['details'];assert len(rows)==8 and all(q['all_pairs_equal'] for q in rows)
assert all(q['pixels_on_all_current_masks']==0 for q in rows if q['mode']=='ordinary')
assert len(env['gate_rows'])==len(rows)
write('presence_receipt.json',dict(gate_rows=env['gate_rows'],ended=now(),checked_original_traces=len(env['traces']),inputs_unchanged=all(sha(ROOT/f)==h for f,h in proto['inputs'].items()),records=rows))
