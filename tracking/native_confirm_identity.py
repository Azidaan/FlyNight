"""Causal public-ID linking after native confirmation; native track states stay intact.

The caller supplies only already confirmed, first-public native tracks and
pre-association Lost-to-detection proposals. This module never creates boxes,
changes Kalman states, or revises an earlier output frame.
"""
from __future__ import annotations
import numpy as np
from ultralytics.trackers.utils import matching


class NativeConfirmedIdentity:
    def __init__(self,alpha=.2,threshold=.8,enabled=True):
        if not np.isfinite([alpha,threshold]).all() or alpha<0:raise ValueError('Finite nonnegative alpha required.')
        self.alpha=float(alpha);self.threshold=float(threshold);self.enabled=bool(enabled)
        self.alias={};self.owner={};self.seen=set();self.first_public={};self.last_frame=-1

    def canonical(self,tid):return self.alias.get(int(tid),int(tid))

    def _collisions(self,ids):
        groups={};revoked=[]
        for tid in ids:groups.setdefault(self.canonical(tid),[]).append(tid)
        for root,group in groups.items():
            if len(group)<2:continue
            keep=root if root in group else self.owner.get(root)
            if keep not in group:keep=min(group,key=lambda t:(self.first_public.get(t,float('inf')),t))
            for tid in group:
                if tid==keep:continue
                assert tid in self.alias and self.alias[tid]==root
                del self.alias[tid];revoked.append(dict(native_id=tid,old_canonical_id=root,kept_native_id=keep))
        assert len({self.canonical(t) for t in ids})==len(ids)
        return revoked

    def update(self,frame,output,newly_confirmed,edges):
        if frame<=self.last_frame:raise ValueError('Frames must be strictly increasing.')
        self.last_frame=int(frame);out=np.asarray(output);ids=[int(v) for v in out[:,4]] if len(out) else []
        if len(ids)!=len(set(ids)):raise ValueError('Native public IDs must be unique in a frame.')
        if not self.enabled:
            assert not self.alias
            for tid in ids:self.first_public.setdefault(tid,frame);self.owner[tid]=tid
            self.seen.update(ids);return out.copy(),dict(links=[],revoked=[],allowed=[])
        revoked=self._collisions(ids);emitted={self.canonical(t) for t in ids};fresh=set(map(int,newly_confirmed)) & set(ids)-self.seen
        allowed=[]
        for e in edges:
            old=int(e['old_tid']);new=int(e['new_tid']);cost=float(e['cost']);root=self.canonical(old)
            if not bool(e.get('accepted',False)) or new not in fresh or old not in self.seen or old in ids:continue
            if self.owner.get(root)!=old or root in emitted or old==new:continue
            if not np.isfinite(cost) or cost>=1:continue
            allowed.append(dict(old_tid=old,new_tid=new,canonical_id=root,cost=cost,corrected_cost=cost-self.alpha))
        rows=sorted({e['old_tid'] for e in allowed});columns=sorted({e['new_tid'] for e in allowed});links=[]
        if rows and columns:
            ri={v:i for i,v in enumerate(rows)};ci={v:i for i,v in enumerate(columns)};C=np.full((len(rows),len(columns)),np.inf,dtype=np.float32)
            by_pair={}
            for e in allowed:
                key=(e['old_tid'],e['new_tid'])
                if key in by_pair:raise ValueError('Duplicate old/new proposal.')
                by_pair[key]=e;C[ri[key[0]],ci[key[1]]]=e['corrected_cost']
            pairs=matching.linear_assignment(C,thresh=self.threshold)[0]
            for i,j in np.asarray(pairs,dtype=int).reshape(-1,2):
                e=by_pair[(rows[i],columns[j])];root=e['canonical_id'];new=e['new_tid']
                assert root not in emitted and new not in self.seen
                self.alias[new]=root;emitted.add(root);links.append(e)
        assert len({self.canonical(t) for t in ids})==len(ids)
        result=out.copy()
        if len(result):result[:,4]=[self.canonical(t) for t in ids]
        for tid in ids:self.first_public.setdefault(tid,frame);self.owner[self.canonical(tid)]=tid
        self.seen.update(ids)
        return result,dict(links=links,revoked=revoked,allowed=allowed)
