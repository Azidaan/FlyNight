"""Check predicates independently, reconstruct the ID collisions and compare paired ledgers."""
from run import P,ROOT,INV,L,sha,j,save
from collections import Counter,defaultdict
from statistics import median
from math import log,exp,isclose
import numpy as np
from scipy.optimize import linear_sum_assignment
def v(b):
    x,y,X,Y=map(float,b);return [(x+X)/2,(y+Y)/2,log(X-x),log(Y-y)]
def overlap(a,b):
    x,y,X,Y=a;t,u,T,U=b;i=max(0,min(X,T)-max(x,t))*max(0,min(Y,U)-max(y,u));return i/((X-x)*(Y-y)+(T-t)*(U-u)-i)
def audit():
    f=j(P/'frozen.json');end=j(P/'complete.json');assert end['frozen_sha256']==sha(P/'frozen.json')
    agg=defaultdict(Counter); changed=[]; verified=Counter()
    for vid,meta in f['videos'].items():
        path=P/'videos'/f'{vid}.json';assert sha(path)==end['outputs'][vid];data=j(path);a=data['candidates'];s=data['split'];matched={}; expected={}
        na=j(INV/'native'/f'{vid}.json');events={r['frame']:r for r in na['events']}
        with np.load(ROOT/meta['raw']) as z:fr,bb,cl,cf=[z[k] for k in ('frame_idx','xyxy','cls','conf')]
        for t,event in events.items():
            for r in event['rows']:
                cs=[c for c in r['candidates'] if c['forward_iou']>=.2]
                if r['age']>2 or len(cs)!=1 or cs[0]['new_tid'] is None:continue
                c=cs[0];k=f"{t}|{r['old_tid']}|{c['new_tid']}";old={u['frame']:v(u['box']) for u in r['row_data']['history'][-4:]};ts=sorted(old)
                if len(ts)<2:expected[k]=(None,[]);continue
                last=ts[-1];speed=[median((old[b][i]-old[a][i])/(b-a) for a,b in zip(ts,ts[1:])) for i in range(4)]
                d=0
                for u,b in [(t-1,c['birth']['box']),(t,c['box'])]:
                    p=[old[last][i]+(u-last)*speed[i] for i in range(4)];obs=v(b);sc=[exp(p[2])/2,exp(p[3])/2,log(2),log(2)];d=max(d,*[abs(x-y)/z for x,y,z in zip(obs,p,sc)])
                    if u==t-1:
                        x,y,w,h=p;w=exp(w)/2;h=exp(h)/2;pred=(x-w,y-h,x+w,y+h)
                b=c['birth']['box'];alts=[];th=overlap(b,pred)
                if d<=1:
                    for i in np.flatnonzero((fr==t-1)&(cl==r['row_data']['cls'])&(cf>=.1)):
                        b0=tuple(map(float,bb[i]))
                        if overlap(b0,b)<.5 and overlap(b0,pred)>th:alts.append(int(i))
                expected[k]=(d,alts)
        assert set(expected)==set(a)
        labels=j(L/f'{vid}.json')['rows']
        for k,(d,alts) in expected.items():
            r=a[k];assert ((d is None and r['score'] is None) or isclose(d,r['score'],rel_tol=1e-8,abs_tol=1e-8)) and alts==r['alternatives'] and bool(alts)==r['veto'];l=labels[k];assert r['kind']==('unknown' if not l['known'] else 'correct' if l['positive'] else 'foreign')
        with np.load(INV/'native'/f'{vid}.npz') as z:o=z['outputs']
        ids=defaultdict(set)
        for t,i in o[:,[0,5]].astype(int):ids[t].add(i)
        for arm in ('H','V'):
            logrows=data['arms'][arm];links=logrows['links'];at=defaultdict(list);candidates=defaultdict(list)
            for k in links:at[a[k]['t']].append(k)
            for k,(d,alt) in expected.items():
                if d is not None and d<=1 and (arm=='H' or not alt):candidates[a[k]['t']].append(k)
            alias={};owner={};seen=set();born={};rev={}
            for t in sorted(ids):
                cur=ids[t];groups=defaultdict(list)
                for i in cur:groups[alias[i][0] if i in alias else i].append(i)
                for root,ls in groups.items():
                    if len(ls)<2:continue
                    keep=root if root in ls else owner.get(root)
                    if keep not in ls:keep=min(ls,key=lambda x:(born.get(x,1e100),x))
                    for i in ls:
                        if i!=keep:_,k=alias.pop(i);rev[k]=dict(frame=int(t),kept=int(keep))
                assigned={alias[i][0] if i in alias else i for i in cur};eligible={}
                for k in candidates[t]:
                    r=a[k];old,new=r['old'],r['new'];root=alias.get(old,(old,None))[0]
                    if old in seen-cur and new in cur-seen and root not in assigned and owner.get(root)==old and new in events[t]['newly_confirmed']:eligible[k]=root
                if eligible:
                    rr=sorted({a[k]['old'] for k in eligible});cc=sorted({a[k]['new'] for k in eligible});m,n=len(rr),len(cc);M=np.zeros((m+n,m+n));M[:m,:n]=1e6;M[:m,n:]=.4;M[m:,:n]=.4; costs={}
                    for k in eligible:
                        d=expected[k][0];costs[k]=float(np.float32(1.6*d/(1+d)));M[rr.index(a[k]['old']),cc.index(a[k]['new'])]=costs[k]
                    ii,jj=linear_sum_assignment(M);assert abs(M[ii,jj].sum()-(sum(costs[k] for k in at[t])+.4*(m+n-2*len(at[t]))))<1e-6;verified['assignment_objectives']+=1
                for k in at[t]:
                    assert k in eligible and links[k]==eligible[k];alias[a[k]['new']]=(links[k],k)
                assert len({alias.get(i,(i,None))[0] for i in cur})==len(cur)
                for i in cur:owner[alias.get(i,(i,None))[0]]=i;born.setdefault(i,t)
                seen.update(cur)
            assert rev==logrows['revocations']
        arms=data['arms'];h,veto=[set(arms[z]['links']) for z in ('H','V')];rh,rv=[set(arms[z]['revocations']) for z in ('H','V')];count=Counter(videos=1,pairs=len(a),flagged=sum(r['veto'] for r in a.values()),H=len(h),V=len(veto),rev_H=len(rh),rev_V=len(rv),prevented=len(rh-rv),new_conflicts=len(rv-rh),retained_conflicts=len(rv&rh),videos_prevented=int(bool(rh-rv)),videos_lost_correct=int(any(a[k]['kind']=='correct' for k in h-veto)))
        for group,items in [('lost',h-veto),('gained',veto-h),('H',h),('V',veto)]:
            for k in items:count[group+'_'+a[k]['kind']]+=1
        for k in sorted((h^veto)|(rh^rv)):
            changed.append(dict(video=vid,split=s,key=k,kind=a[k]['kind'],veto=a[k]['veto'],H=k in h,V=k in veto,H_rev=arms['H']['revocations'].get(k),V_rev=arms['V']['revocations'].get(k)))
        agg[s].update(count);agg['all'].update(count)
    assert all(sha(ROOT/k)==s for k,s in f['inputs'].items())
    result=dict(passed=True,counts=agg,changed=changed,checks=verified,gate=agg['all']['prevented']>0 and agg['all']['new_conflicts']==0 and agg['all']['lost_correct']==0)
    save(P/'result.json',result);print(dict(agg),result['gate'])
if __name__=='__main__':audit()
