"""Evaluation-only independent exact anchors; never used in weight computation."""
from common import *
import numpy as np
def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));den=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/den if den>0 else 0
def evaluation_tables():
    p=read(OUT/'evaluation_plan.json');anchors=read(ROOT/p['clock'])['anchors'];labels=read(ROOT/p['labels']);by={int(r.get('frameIndex',i)):r for i,r in enumerate(labels)}
    with np.load(ROOT/p['detector_file'],allow_pickle=False) as f:boxes=f['xyxy'].copy()
    def hits(frame,box):
        key=str(frame)
        if key not in anchors:return None
        a=anchors[key];row=by[int(a['frameIndex'])];assert row['name']==a['name']
        return [str(o['id']) for o in row.get('labels') or [] if o.get('box2d') and not (o.get('attributes') or {}).get('crowd',False) and overlap(box,[o['box2d'][k] for k in ['x1','y1','x2','y2']])>=.5]
    tables=[]
    for e in read(OUT/'runtime.json')['events']:
        expected={};witness={}
        for r in e['rows']:
            past=[]
            for obs in p['past_anchored_assignments']:
                if not e['frame']-12<=obs['frame']<e['frame']:continue
                rid=obs['assigned'].get(str(r['tid']))
                if rid is not None:past.append(dict(frame=obs['frame'],raw_id=int(rid),hits=hits(obs['frame'],boxes[int(rid)])))
            ids=[x['hits'][0] for x in past if x['hits'] is not None and len(x['hits'])==1]
            valid=bool(past) and len(ids)==len(past) and len(set(ids))==1
            expected[str(r['tid'])]=ids[0] if valid else None;witness[str(r['tid'])]=past
        tables.append(dict(frame=e['frame'],current_exact=str(e['frame']) in anchors,expected=expected,expected_witness=witness,current={str(c['raw_id']):hits(e['frame'],c['box']) for c in e['candidates']}))
    return tables
def classify(table,pairs):
    out=[]
    for t,r in pairs:
        expected=table['expected'][str(t)];hits=table['current'][str(r)];kind='unknown' if expected is None or hits is None or len(hits)!=1 else 'correct' if expected==hits[0] else 'wrong'
        out.append(dict(tid=t,raw_id=r,outcome=kind,expected_identity=expected,current_hits=hits))
    return out
