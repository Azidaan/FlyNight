"""Read-only target check before starting the bounded sampling experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_semantic_sampling_20261023 import RUN,OLD,CACHE,ARMS,read,write,verify_sources
from scripts.z13_semantic_audit_20261023 import metadata,audit
from scripts.z13_candidate_ranker_20261021 import pack
import scripts.z13_candidate_identity_20261021 as reader
import numpy as np

if __name__=='__main__':
    p=read(RUN/'protocol.json');verify_sources(p);base=read(OLD/'protocol.json')
    a=read(CACHE/'fit/labels.json');b=read(OLD/'evaluate/labels.json');labels=a+b
    ma,ha,ca=metadata(base['fit'],a);mb,hb,cb=metadata(base['evaluate'],b)
    union=np.zeros(len(labels),bool);details={}
    for arm in ARMS:
        seq,_=reader.load('fit',arm);first=pack(a,seq)
        old=reader.CACHE
        try:reader.CACHE=OLD;seq,_=reader.load('evaluate',arm)
        finally:reader.CACHE=old
        last=pack(b,seq);arrays=tuple(np.concatenate([x,y]) for x,y in zip(first,last))
        negative,summary=audit(labels,arrays[5],arrays[3],ma+mb)
        assert not np.any(negative & (arrays[4]>0));union|=negative.any(1);details[arm]=summary
    known=np.asarray([l['kind']>=0 for l in labels]);assert not np.any(union & ~known)
    result=dict(arms=details,gt_centers_checked=ca+cb,common_hard_rows=int(union.sum()),common_indices=np.flatnonzero(union[known]).tolist())
    write(RUN/'preflight_target.json',result)
    print({a:(d['hard_rows'],d['hard_candidates']) for a,d in details.items()},'union',result['common_hard_rows'])
