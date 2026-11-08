"""Recover a sole admissible detection only from the native unmatched remainder.

The ordinary first-stage matches stay fixed. Geometry and optional visual
corrections act only on free Lost rows and free high-confidence detections.
No box state, historical ID or duplicate-removal policy is changed here.
"""
from __future__ import annotations

import numpy as np
from ultralytics.trackers.utils import matching
from tracking.spatial_identity_assignment import corrected_cost


def free_singleton_matches(cost, native_matches, unmatched_rows, unmatched_columns,
                           plan, alpha, beta, arm, states, track_classes,
                           detection_classes, threshold=.8):
    base=np.asarray(cost)
    if base.ndim!=2 or np.isnan(base).any():raise ValueError('Expected a two-dimensional cost matrix without NaN.')
    if not np.isfinite([alpha,beta,threshold]).all() or min(alpha,beta)<0:raise ValueError('Finite nonnegative corrections required.')
    if len(states)!=base.shape[0] or len(track_classes)!=base.shape[0] or len(detection_classes)!=base.shape[1]:raise ValueError('Track/detection metadata shape mismatch.')
    native=np.asarray(native_matches,dtype=int).reshape(-1,2)
    rr=np.asarray(unmatched_rows,dtype=int);cc=np.asarray(unmatched_columns,dtype=int)
    assert len(set(rr.tolist()))==len(rr) and len(set(cc.tolist()))==len(cc)
    assert not set(native[:,0])&set(rr) and not set(native[:,1])&set(cc)
    result=base.copy();allowed=[]
    if len(rr) and len(cc):
        if arm=='U_THRESHOLD':
            for i in rr:
                if int(states[i])!=2:continue
                for j in cc:
                    if track_classes[i]==detection_classes[j] and base[i,j]<1:
                        allowed.append((int(i),int(j)));result[i,j]-=alpha
        else:
            row_set=set(rr.tolist());column_set=set(cc.tolist())
            mapped_arm=arm.replace('U_','G_',1) if arm.startswith('U_') else arm
            if arm=='U_GEO':mapped_arm='G'
            proposal=corrected_cost(base,plan['votes'],alpha,beta,mapped_arm,plan['columns'])
            for i,j in plan['votes']:
                if i in row_set and j in column_set and int(states[i])==2 and track_classes[i]==detection_classes[j] and base[i,j]<1:
                    allowed.append((int(i),int(j)));result[i,j]=proposal[i,j]
        if np.array_equal(result,base):added=np.empty((0,2),int)
        else:
            residual_cost=np.full((len(rr),len(cc)),np.inf,dtype=base.dtype)
            ri={int(v):k for k,v in enumerate(rr)};ci={int(v):k for k,v in enumerate(cc)}
            for i,j in allowed:residual_cost[ri[i],ci[j]]=result[i,j]
            residual=matching.linear_assignment(residual_cost,thresh=threshold)[0]
            residual=np.asarray(residual,dtype=int).reshape(-1,2)
            added=np.column_stack([rr[residual[:,0]],cc[residual[:,1]]]) if len(residual) else np.empty((0,2),int)
            # Every new edge has passed an explicit proposal/threshold rule.
            assert all((int(i),int(j)) in allowed for i,j in added)
        matches=np.concatenate([native,added],axis=0)
    else:matches=native.copy()
    if len(matches):matches=matches[np.argsort(matches[:,0],kind='stable')]
    used_rows=set(matches[:,0]);used_columns=set(matches[:,1])
    assert len(used_rows)==len(matches) and len(used_columns)==len(matches)
    return matches,[int(i) for i in rr if i not in used_rows],[int(j) for j in cc if j not in used_columns],result,allowed


class FreeSingletonAssociation:
    """Mixin for the existing spatial tracker; the all-zero path is unchanged."""
    def _first_association(self,pool,detections,activated,refind):
        if self.identity_alpha==0 and self.identity_beta==0:
            return super()._first_association(pool,detections,activated,refind)
        cost=self.get_dists(pool,detections)
        native,rr,cc=matching.linear_assignment(cost,thresh=self.args.match_thresh)
        plan=dict(votes=[],columns={})
        if len(rr) and len(cc) and self.identity_provider is not None and self.identity_arm!='U_THRESHOLD':
            plan=self.identity_provider(self,pool,detections,cost)
        matches,ut,ud,changed,allowed=free_singleton_matches(cost,native,rr,cc,plan,
            self.identity_alpha,self.identity_beta,self.identity_arm,[t.state for t in pool],
            [t.cls for t in pool],[d.cls for d in detections],self.args.match_thresh)
        self.identity_events.append(dict(frame=self.frame_id,track_ids=[int(t.track_id) for t in pool],
            votes=plan['votes'],matches=matches.tolist(),native_matches=np.asarray(native).reshape(-1,2).tolist(),
            eligible_free_edges=[list(p) for p in allowed],changed_costs=int(np.count_nonzero(changed!=cost))))
        self._apply_matches(matches,pool,detections,activated,refind)
        return ut,ud
