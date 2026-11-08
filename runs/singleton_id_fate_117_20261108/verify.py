"""Independent ledger audit, actual return frames/indices, and sparse labels."""
from analyze import P,D,T,ROOT,INV,read,sha,save
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
def main():
    f=read(P/'frozen.json');rs=read(P/'events.json');summary=read(P/'summary.json');assert len(rs)==117 and sha(P/'events.json')==summary['events_sha256']
    ck=Counter();names=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle']
    for vid in {r['video'] for r in rs}:
        rr=[r for r in rs if r['video']==vid];prev=read(D/'videos'/f'{vid}.json');b=read(T/'decisions.json')[vid]['arm'];arms={**prev['arms'],'B':b,'B0':dict(links={},revocations={})}
        with np.load(INV/f'{vid}.npz') as z:out=z['outputs']
        with np.load(ROOT/f['videos'][vid]['raw']) as z:rf,cf,cl,boxes=[z[k] for k in ('frame_idx','conf','cls','xyxy')]
        ann=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{vid}.json');align=read(ROOT/f'runs/train_cache/{vid}/alignment.json')['matches'];cache={}
        def labels(t):
            if t in cache:return cache[t]
            fr=ann[next(i for i,a in enumerate(align) if a['video_frame_index']==t)]
            objs=[o for o in fr.get('labels') or [] if o['category'] in names and not (o.get('attributes') or {}).get('crowd',False)];native=out[out[:,0]==t];found={}
            for cls in set(map(int,native[:,7])):
                ii=np.flatnonzero(native[:,7]==cls);oo=[o for o in objs if o['category']==names[cls]]
                if not oo:continue
                a=native[ii,1:5].astype(float);bb=np.array([[o['box2d'][k] for k in ('x1','y1','x2','y2')] for o in oo]);s=np.maximum(0,np.minimum(a[:,None,2:],bb[None,:,2:])-np.maximum(a[:,None,:2],bb[None,:,:2]));inter=s.prod(-1);area=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(bb[:,2:]-bb[:,:2]).prod(1)[None,:]-inter;v=inter/np.maximum(area,1e-12);v[v<.5]=0
                row,col=linear_sum_assignment(-v)
                for i,j in zip(row,col):
                    if v[i,j]>0:found[int(native[ii[i],5])]=str(oo[j]['id'])
            cache[t]=(found,{str(o['id']) for o in objs});ck['annotation_frames']+=1;return cache[t]
        def alias_at(arm,t):
            events=[]
            for k,root in arms[arm]['links'].items():events.append((int(k.split('|')[0]),1,int(k.split('|')[2]),root))
            for k,r in arms[arm]['revocations'].items():events.append((r['frame'],0,int(k.split('|')[2]),None))
            a={}
            for fr,kind,n,root in sorted(events):
                if fr>t:break
                if kind:a[n]=root
                else:del a[n]
            return a
        for r in rr:
            t,old,new=map(int,r['key'].split('|'));assert r['key'] in prev['arms']['H']['links']
            possible=out[(out[:,5]==old)&(out[:,0]>t)]
            ret=r['return_info'];assert (ret is not None)==bool(len(possible))
            if ret:
                q=possible[0];u=int(q[0]);assert ret['frame']==u and ret['box']==q[1:5].tolist()
                ids=np.flatnonzero(rf==u);domain=ids[cf[ids]>=.25] if q[6]>=.25 else ids[(cf[ids]>.1)&(cf[ids]<.25)];idx=int(domain[int(q[8])]);assert idx==ret['raw_index'] and np.array_equal(boxes[idx],np.array(ret['raw_box']));ck['return_indices']+=1
            for arm,z in r['arms'].items():
                if r['target'] is None:assert z['state']=='unknown_prior' and not z['observations'];continue
                target=r['target'];anchor=r['anchor'];assert labels(anchor)[0][old]==target;assert alias_at(arm,anchor).get(old,old)==z['wanted']
                future=[int(a['video_frame_index']) for a in align if a['video_frame_index']>t and target in labels(int(a['video_frame_index']))[1]]
                assert [s['frame'] for s in z['observations']]==future
                for s in z['observations']:
                    u=s['frame'];mapped=alias_at(arm,u);lab=labels(u)[0];native=out[out[:,0]==u];hits=[i for i,g in lab.items() if g==target];public=[mapped.get(i,i) for i in hits]
                    assert sorted(s['target_native'])==sorted(hits) and sorted(s['target_public'])==sorted(public)
                    status='saved' if z['wanted'] in public else 'changed_id' if public else 'unmatched_target';assert s['status']==status
                    ck['outcome_points']+=1
                obs=z['observations'][:2];expected='insufficient_future' if len(obs)<2 else 'hijacked' if any(s['hijacked'] for s in obs) else 'saved_both' if all(s['status']=='saved' for s in obs) else 'changed_id' if any(s['status']=='changed_id' for s in obs) else 'unmatched_target';assert z['state']==expected
    for arm in summary['arms']:assert dict(Counter(r['arms'][arm]['state'] for r in rs))==summary['arms'][arm]['states']
    assert all(sha(ROOT/k)==v for k,v in f['files'].items())
    save(P/'audit.json',dict(passed=True,checks=ck,summary_sha256=sha(P/'summary.json'),events_sha256=sha(P/'events.json')));print(ck)
if __name__=='__main__':main()
