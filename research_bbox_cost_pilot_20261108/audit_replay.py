"""Independent canonical-ID and SciPy LAP reconstruction; alpha is explicit."""
from collections import Counter
from research_bbox_singleton_pilot_20261108.audit import independently_proposed


def replay(native,data,cfg,ledger,alpha):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    aliases={};owners={};seen=set();logs={r['frame']:r for r in ledger};events={e['frame']:e for e in data['events']};result=native.copy();checks=Counter();counts=np.bincount(native[:,0].astype(int),minlength=data['frames']);offset=np.r_[0,counts.cumsum()]
    for f in range(data['frames']):
        lo,hi=offset[f:f+2];ids=list(map(int,native[lo:hi,5]));root=lambda t:aliases.get(t,t);event=events.get(f,dict(frame=f,rows=[],newly_confirmed=[]));d=logs.get(f,dict(links=[],revoked=[],allowed=[]))
        for r in d['revoked']:
            assert r['native_id'] in ids and r['kept_native_id'] in ids and aliases[r['native_id']]==r['old_canonical_id']==root(r['kept_native_id']);del aliases[r['native_id']]
        current={root(t) for t in ids};fresh=set(event['newly_confirmed']) & set(ids)-seen;eligible={}
        for e in independently_proposed(event,cfg) if cfg else []:
            old,new=e['old_tid'],e['new_tid'];canonical=root(old)
            if new not in fresh or old not in seen or old in ids or owners.get(canonical)!=old or canonical in current or e['cost']>=1:continue
            eligible[(old,new)]=dict(old_tid=old,new_tid=new,canonical_id=canonical,cost=e['cost'],corrected_cost=e['cost']-alpha)
        assert sorted(eligible)==sorted((e['old_tid'],e['new_tid']) for e in d['allowed'])
        rr=sorted({a for a,b in eligible});cc=sorted({b for a,b in eligible});R,C=len(rr),len(cc)
        if R and C:
            matrix=np.full((R+C,R+C),np.inf);matrix[R:,C:]=0
            for i in range(R):matrix[i,C+i]=.4
            for j in range(C):matrix[R+j,j]=.4
            for (a,b),e in eligible.items():matrix[rr.index(a),cc.index(b)]=np.float32(e['corrected_cost'])
            ii,jj=linear_sum_assignment(matrix);actual=sum(float(np.float32(e['corrected_cost'])) for e in d['links'])+.4*(R+C-2*len(d['links']));assert abs(actual-matrix[ii,jj].sum())<1e-5;checks['independent_optional_assignments']+=1
        for e in d['links']:assert e==eligible[(e['old_tid'],e['new_tid'])];aliases[e['new_tid']]=e['canonical_id']
        canonical=[root(t) for t in ids];assert len(canonical)==len(set(canonical));result[lo:hi,5]=canonical
        for t in ids:owners[root(t)]=t
        seen.update(ids);checks['causal_frames']+=1
    return result,checks
