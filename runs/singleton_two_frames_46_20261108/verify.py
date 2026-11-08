from compare import P,D,ROOT,INV,sha,j,save,Counter,defaultdict
from statistics import median
from math import exp,log
import numpy as np
from scipy.optimize import linear_sum_assignment
def vec(b):
    x,y,X,Y=b;return ((x+X)/2,(y+Y)/2,log(X-x),log(Y-y))
def iou(a,b):
    x,y,X,Y=map(float,a);u,v,U,V=map(float,b);i=max(0,min(X,U)-max(x,u))*max(0,min(Y,V)-max(y,v));return i/((X-x)*(Y-y)+(U-u)*(V-v)-i)
def main():
    manifest=j(D/'frozen.json');dec=j(P/'decisions.json');checks=Counter();tot=defaultdict(Counter);diff=[]
    for vid,r in dec.items():
        prev=j(D/'videos'/f'{vid}.json');a=prev['candidates'];events={r['frame']:r for r in j(INV/f'{vid}.json')['events']};ww=r['confirmation_alternatives_on_birth_warnings'];assert set(ww)=={k for k in a if a[k]['veto']}
        with np.load(ROOT/manifest['videos'][vid]['raw']) as z:rf,rb,cl,cf=[z[k] for k in ('frame_idx','xyxy','cls','conf')]
        for k,alt in ww.items():
            x=a[k];t=x['t'];row,=[u for u in events[t]['rows'] if u['old_tid']==x['old']];c,=[u for u in row['candidates'] if u['new_tid']==x['new']]
            h={u['frame']:vec(u['box']) for u in row['row_data']['history'][-4:]};ts=sorted(h);last=ts[-1]
            p=[h[last][i]+(t-last)*median((h[b][i]-h[a][i])/(b-a) for a,b in zip(ts,ts[1:])) for i in range(4)]
            x0,y,w,hh=p;w=exp(w)/2;hh=exp(hh)/2;pred=[x0-w,y-hh,x0+w,y+hh];own=iou(c['box'],pred)
            ids=np.flatnonzero((rf==t)&(cf>=.1)&(cl==row['row_data']['cls']));assert alt==[int(i) for i in ids if iou(rb[i],c['box'])<.5 and iou(rb[i],pred)>own];checks['warning_checks']+=1
        n=r['arm'];ls=n['links'];pending=defaultdict(list);selected=defaultdict(list)
        for k,x in a.items():
            if x['score'] is not None and x['score']<=1 and not ww.get(k):pending[x['t']].append(k)
        for k in ls:selected[a[k]['t']].append(k)
        with np.load(INV/f'{vid}.npz') as z:o=z['outputs']
        ids=defaultdict(set)
        for t,i in o[:,[0,5]].astype(int):ids[int(t)].add(int(i))
        alias={};seen=set();born={};owner={};rev={}
        for t,cur in sorted(ids.items()):
            groups=defaultdict(list)
            for i in cur:groups[alias.get(i,(i,None))[0]].append(i)
            for root,ls0 in groups.items():
                if len(ls0)<2:continue
                keep=root if root in ls0 else owner.get(root)
                if keep not in ls0:keep=min(ls0,key=lambda x:(born.get(x,float('inf')),x))
                for i in ls0:
                    if i!=keep:_,k=alias.pop(i);rev[k]=dict(frame=t,kept=keep)
            emitted={alias.get(i,(i,None))[0] for i in cur};allow={}
            for k in pending[t]:
                x=a[k];old=x['old'];new=x['new'];root=alias.get(old,(old,None))[0]
                if old in seen-cur and new in cur-seen and root not in emitted and owner.get(root)==old and new in events[t]['newly_confirmed']:allow[k]=root
            if allow:
                rr=sorted({a[k]['old'] for k in allow});cc=sorted({a[k]['new'] for k in allow});m,nc=len(rr),len(cc);M=np.zeros((m+nc,m+nc));M[:m,:nc]=1e6;M[:m,nc:]=.4;M[m:,:nc]=.4;cost={}
                for k in allow:
                    s=a[k]['score'];cost[k]=np.float32(1.6*s/(1+s)).item();M[rr.index(a[k]['old']),cc.index(a[k]['new'])]=cost[k]
                ii,jj=linear_sum_assignment(M);assert abs(M[ii,jj].sum()-sum(cost[k] for k in selected[t])-.4*(m+nc-2*len(selected[t])))<1e-6;checks['objectives']+=1
            for k in selected[t]:assert allow[k]==ls[k];alias[a[k]['new']]=(ls[k],k)
            assert len({alias.get(i,(i,None))[0] for i in cur})==len(cur)
            for i in cur:owner[alias.get(i,(i,None))[0]]=i;born.setdefault(i,t)
            seen.update(cur)
        assert n['revocations']==rev
        arms={**prev['arms'],'B':n};out=Counter(videos=1,pairs=len(a))
        for name,arm in arms.items():
            out[name+'_rev']=len(arm['revocations']);out[name+'_links']=len(arm['links'])
            for k in arm['links']:out[name+'_'+a[k]['kind']]+=1
        for name in ('H','V'):
            arm=arms[name]
            for prop in ('links','revocations'):
                for state,keys in [('removed',set(arm[prop])-set(n[prop])),('added',set(n[prop])-set(arm[prop]))]:
                    out[f'{name}_B_{prop}_{state}']=len(keys)
                    for k in sorted(keys):
                        kind=a[k]['kind'];out[f'{name}_B_{prop}_{state}_{kind}']+=1;diff.append(dict(video=vid,key=k,vs=name,prop=prop,change=state,kind=kind))
        tot['all'].update(out);tot[r['split']].update(out)
    for rr in j(D/'five_cases/audit.json')['rows']:
        k='|'.join(map(str,rr['key']));assert bool(dec[rr['video']]['confirmation_alternatives_on_birth_warnings'][k])==rr['birth_and_now']
    assert tot['all']['videos']==46 and tot['all']['pairs']==198 and checks['warning_checks']==31
    save(P/'result.json',dict(totals=tot,differences=diff,checks=checks,decisions_sha256=sha(P/'decisions.json'),before_sha256=sha(P/'before.json')));print(dict(tot),checks)
if __name__=='__main__':main()
