"""Geometry only. Bind original candidates and distinguish empty support before filtering."""
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
RUN = Path('runs/flyvis_pair_support_overlap_20261107')
IDENT = 'flyvis_pair_support_overlap_v1'
OLD = Path('runs/z13_spatial_bins_gate_20261025')
LABEL = Path('runs/z13_region_assignment_20261025/labels')
TEMP = Path('runs/z13_temporal_observation_gate_20261025')
GRID = Path('runs/z13_local_peak_20261021/geometry.json')
CAT = Path('experiments/project_history.json')
PLAN = Path('dissertation/ЕДИНЫЙ_ПЛАН.md')
GUARD = Path('runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py')
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
SOURCES = [OLD/'recovery1/protocol.json', OLD/'report.json', LABEL/'report.json', TEMP/'report.json', GRID, SCRIPT, GUARD,
           Path('tracking/z13_region_assignment.py'), Path('tracking/z13_readout.py')]
CATS = ['both_empty', 'one_empty', 'disjoint', 'partial_contained', 'partial_both_unique', 'identical']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def dump(p, d, exclusive=False):
    with p.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False); f.write('\n')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()
def journal(text):
    s=PLAN.read_text(encoding='utf-8')
    anchor='## Журнал изменений\n\n'
    assert s.count(anchor)==1
    PLAN.write_text(s.replace(anchor,anchor+'**'+datetime.now().strftime('%d.%m.%Y, %H:%M')+' MSK** — '+text+'\n\n',1),encoding='utf-8')
def prepare():
    cat=read(CAT); assert not RUN.exists() and not any(c['id']==IDENT for c in cat['experiments'])
    RUN.mkdir()
    p=dict(id=IDENT, registered=datetime.now().astimezone().isoformat(), git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        question='В одной сохранённой области глаза два реально предложенных бокса читают одинаковые, общие или раздельные колонки?',
        sets='Primary frozen evaluate26; fit20 and fit84 separately for context. All original queries including unknown identity; no subset by result.',
        primary='For each query all unordered pairs of distinct saved REAL choices; no keep/null action. Report exactly-two separately and any same/no shared cell per query. Within same eye and time only. Same cell means the same index, not equal activity.',
        pre='Before mask.any: high score >=.25, same class, centre inside eye square and distance to nearest grid point <=.6 spacing as in historical region selector. Also count all same-class centres inside the square BEFORE proximity filter including empty boxes. Outside-eye centres are not candidates.',
        categories=CATS, annotation='Separate previously known beneficial vs incorrect pair (different indices). Geometry for all pairs is label-free.',
        acceptance='Exact original real choices, UID/frame/track and old quadrant totals for every query across 130 videos, 37296 queries/26600 known. Independently count each pair intersection using grid in rectangle intersection; independent CSV recount. No threshold of scientific success.',
        inputs='Read original report manifests and match SHA of each per-video file before use; use only frame/crop indices and raw cached boxes. No replacement by nearest cell, no model, training, decoded frames, GPU, comparison of tracker accuracy or latency.',
        sha256={i.as_posix():sha(i) for i in SOURCES})
    dump(RUN/'protocol.json',p,True)
    cat['experiments'].append(dict(id=IDENT,title='Общие и раздельные колонки FlyVis у конкурирующих рамок',area='flyvis',state='ready',
        started=p['registered'][:10],question=p['question'],hypothesis='В одной области глаза разные рамки могут читать общие колонки. Проверяем геометрию, а не различимость активности.',
        expected='Описательный учёт пустых рамок и пересечений на сохранённых спорных запросах.',acceptance=[dict(criterion=p['acceptance'],met=None)],
        evidence=[dict(kind='json',ref=(RUN/'protocol.json').as_posix(),claim='Срезы, категории, контроль до подсчёта')]))
    dump(CAT,cat)
    journal('постановка диагностики общих колонок конкурирующих рамок. `'+IDENT+'`. '+p['question']+' Основной срез — прежние 26 видео; пустые области до фильтра отдельно. Только сохранённая геометрия, без обучения. Протокол: '+(RUN/'protocol.json').as_posix()+'.')

def category(a,b,c):
    if min(a,b)==0: return 'one_empty' if max(a,b) else 'both_empty'
    if not c:return 'disjoint'
    if c==a==b:return 'identical'
    return 'partial_contained' if c==min(a,b) else 'partial_both_unique'

def compute():
    sys.path.insert(0,str(GUARD.parent))
    from system_memory_guard import guard_entrypoint
    guard=guard_entrypoint(timeout=900); t=time.perf_counter()
    import numpy as np
    import psutil
    from tracking.z13_readout import HexGeometry
    proc=psutil.Process()
    p=read(RUN/'protocol.json')
    for s,h in p['sha256'].items(): assert sha(Path(s))==h,s
    assert not (RUN/'report.json').exists()
    old=read(SOURCES[0]); oldreport=read(OLD/'report.json'); labels=read(LABEL/'report.json')['video_sha256']; temporal=read(TEMP/'report.json')['manifest']
    grid=read(GRID); g=HexGeometry(np.array(grid['points'],np.float32),grid['eye_size']); pts=g.points.astype(float)
    assert pts.shape==(721,2) and len(set(map(tuple,pts)))==721
    per={}; used={}; path=RUN/'pairs.csv.gz'
    fields=['split','video','uid','j','k','na','nb','shared','category','admitted','exactly_two_admitted','beneficial_vs_incorrect','near_a','near_b']
    with gzip.open(path,'xt',encoding='ascii',newline='') as f:
        out=csv.writer(f); out.writerow(fields)
        for split in ['evaluate26','fit20','fit84']:
            for v in old[split]:
                cache=Path('runs/z13_data')/(v+'.npz'); raw=Path('runs/train_cache')/v/'detections/raw_video.npz'
                rows=LABEL/'videos'/(v+'.json'); tmp=TEMP/'videos'/(v+'.json'); bins=OLD/'videos'/(v+'.json')
                for fn,h in [(cache,old['sha256'][cache.as_posix()]),(raw,old['sha256'][raw.as_posix()]),(rows,labels[v]),(tmp,temporal[tmp.as_posix()]),(bins,oldreport['video_sha256'][v])]:
                    assert sha(fn)==h,fn;used[fn.as_posix()]=h
                with np.load(cache) as z: crops=z['fr_crop']; frames=z['fr_frame']; tids=z['ep_tid']; off=z['ep_offset']
                with np.load(raw) as z: xy=z['xyxy']; score=z['conf']; cls=z['cls']; fi=z['frame_idx']
                rr=read(rows)['rows']; tr=read(tmp)['rows']; br=read(bins)['cases']; n=Counter(); cc=defaultdict(Counter)
                assert len(rr)==len(tr)==len(br) and len(set(r['uid'] for r in rr))==len(rr)
                frame_map={}
                for r,m,b in zip(rr,tr,br):
                    ix=m['current_array_index']; frame=r['frame']; e=m['episode_local']
                    assert r['uid']==m['uid']==b['uid'] and r['kind']==b['kind'] and frame==m['current_frame']==frames[ix]
                    assert v==r['video']==m['video'] and r['tid']==m['track_id']==tids[e] and off[e]<=ix<off[e+1]
                    if frame not in frame_map:frame_map[frame]=np.flatnonzero((fi==frame)&(score>=.25))
                    ids=frame_map[frame]; rg=r['regions']; assert len(ids)==rg['high_detections']
                    choices=rg['choices'][2:]; assert rg['choices'][:2]==[-2,-1] and choices==m['current_choices'][1:]
                    assert len(b['quadrant_counts'])==len(choices)+1 and b['quadrant_counts'][0] is None
                    box=xy[ids].astype(float); c=crops[ix].astype(float); div=np.maximum(c[2:]-c[:2],1)
                    corners=(box.reshape(-1,2,2)-c[:2])*g.eye_size/div
                    centers=corners.mean(1); inside=(cls[ids]==rg['track_class'])&((centers>=0)&(centers<=g.eye_size)).all(1)
                    masks={}; counts={}; near={}; admitted=[]
                    for j in np.flatnonzero(inside).tolist():
                        lo,hi=corners[j]; a=((pts>=lo)&(pts<=hi)).all(1)
                        # Historical nearest() uses float32; historical final distance uses float64 centre.
                        cen=centers[j]; near[j]=bool(np.linalg.norm(g.points[g.nearest(cen)]-cen)<=.6*g.spacing)
                        cnt=int(a.sum()); counts[j]=cnt; masks[j]=int.from_bytes(np.packbits(a).tobytes(),'big')
                        n['square_boxes']+=1; n['square_empty']+=int(not cnt)
                        if near[j]:
                            n['pre_boxes']+=1; n['pre_empty']+=int(not cnt)
                            if cnt:admitted.append(j)
                        if j in choices:
                            # Old quadrant code takes the centre in the detector's native dtype.
                            mid=(xy[ids[j],:2]+xy[ids[j],2:])/2
                            qc=(mid.astype(float)-c[:2])*g.eye_size/div
                            bits=(g.points[:,0]>=qc[0]).astype(int)+2*(g.points[:,1]>=qc[1])
                            quad=np.bincount(bits[a],minlength=4).tolist()
                            assert quad==b['quadrant_counts'][1+choices.index(j)],(v,r['uid'],j)
                    assert choices==admitted,(v,r['uid'],choices,admitted)
                    n['queries']+=1; n['known']+=int(r['kind']>=0); n['admitted_boxes']+=len(choices)
                    n['ambiguous']+=int(len(choices)>=2); n['exactly_two']+=int(len(choices)==2)
                    outcomes=dict(zip(rg['choices'],r['arms']['regions'],strict=True))
                    seen=set();admit_set=set(admitted)
                    for j,k in itertools.combinations(counts,2):
                        a=counts[j]; d=counts[k]; common=(masks[j]&masks[k]).bit_count()
                        lo=np.maximum(corners[j,0],corners[k,0]); hi=np.minimum(corners[j,1],corners[k,1])
                        check=int(np.count_nonzero((pts[:,0]>=lo[0])&(pts[:,0]<=hi[0])&(pts[:,1]>=lo[1])&(pts[:,1]<=hi[1])))
                        assert check==common
                        cat=category(a,d,common); both=j in admit_set and k in admit_set; sem=False
                        cc['square'][cat]+=1
                        if near[j] and near[k]:cc['pre'][cat]+=1
                        if both:
                            cc['admitted'][cat]+=1; seen.add(cat)
                            if len(choices)==2:cc['exactly_two'][cat]+=1
                            if r['kind']>=0:
                                u,w=outcomes[j],outcomes[k]
                                sem=bool((u['beneficial'] and (not w['own_after'] or w['harmful'])) or (w['beneficial'] and (not u['own_after'] or u['harmful'])))
                                if sem:cc['beneficial_vs_incorrect'][cat]+=1
                        out.writerow([split,v,r['uid'],j,k,a,d,common,cat,int(both),int(both and len(choices)==2),int(sem),int(near[j]),int(near[k])])
                    n['any_identical_query']+=int('identical' in seen)
                    n['all_disjoint_query']+=int(seen=={'disjoint'})
                per[v]=dict(split=split,queries_boxes=n,pairs=dict(cc))
                if len(per)%15==0:print(f'{len(per)}/130',flush=True)
    assert sum(z['queries_boxes']['queries'] for z in per.values())==37296 and sum(z['queries_boxes']['known'] for z in per.values())==26600
    totals={}
    for group in ['evaluate26','fit20','fit84','all']:
        nn=Counter(); pp=defaultdict(Counter)
        for z in per.values():
            if group!='all' and group!=z['split']:continue
            nn.update(z['queries_boxes'])
            for k,counter in z['pairs'].items():pp[k].update(counter)
        totals[group]=dict(queries_boxes=nn,pairs=pp)
    dump(RUN/'input_sha256.json',used,True);dump(RUN/'by_video.json',per,True)
    dump(RUN/'report.json',dict(complete=True,totals=totals,seconds=time.perf_counter()-t,process_peak_working_set_bytes=proc.memory_info().peak_wset,
        gpu_used=False,guard=guard.read(),sha256={str(f):sha(f) for f in [RUN/'protocol.json',RUN/'input_sha256.json',RUN/'by_video.json',path]}),True)

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--compute' in sys.argv:compute()
    else:raise ValueError('Specify --prepare or --compute.')
