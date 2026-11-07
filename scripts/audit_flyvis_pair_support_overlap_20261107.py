"""Audit persisted numeric pair counts without importing the measurement code."""
import csv
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/flyvis_pair_support_overlap_20261107'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def main():
    r=load(RUN/'report.json')
    for mapping in (r['sha256'],load(RUN/'protocol.json')['sha256'],load(RUN/'input_sha256.json')):
        for f,h in mapping.items():assert digest(ROOT/f)==h,f
    counts=defaultdict(lambda:defaultdict(Counter));queries=defaultdict(lambda:defaultdict(list)); unique=set()
    with gzip.open(RUN/'pairs.csv.gz','rt',encoding='ascii') as f:
        for row in csv.DictReader(f):
            a,b,k=map(int,(row['na'],row['nb'],row['shared'])); assert 0<=k<=min(a,b); x=a-k;y=b-k
            cat=('both_empty' if a==b==0 else 'one_empty' if a*b==0 else 'disjoint' if k==0
                 else 'identical' if x+y==0 else 'partial_contained' if x*y==0 else 'partial_both_unique')
            assert cat==row['category']
            u=int(row['uid']);j=int(row['j']);l=int(row['k']);assert j<l and (u,j,l) not in unique
            unique.add((u,j,l)); both=row['admitted']=='1';near=row['near_a']==row['near_b']=='1'
            assert both==(near and a*b>0)
            cc=counts[row['video']]; cc['square'][cat]+=1
            if near:cc['pre'][cat]+=1
            if both:
                cc['admitted'][cat]+=1;queries[row['video']][u].append(cat)
            if row['exactly_two_admitted']=='1':assert both;cc['exactly_two'][cat]+=1
            if row['beneficial_vs_incorrect']=='1':assert both;cc['beneficial_vs_incorrect'][cat]+=1
    videos=load(RUN/'by_video.json'); assert len(videos)==130
    groups=defaultdict(lambda:defaultdict(Counter))
    for v,d in videos.items():
        q=d['queries_boxes']; assert dict(counts[v])==d['pairs']
        lens=list(queries[v].values());assert len(lens)==q['ambiguous']
        assert sum(len(x)==1 for x in lens)==q['exactly_two']
        assert sum('identical' in x for x in lens)==q['any_identical_query']
        assert sum(set(x)=={'disjoint'} for x in lens)==q['all_disjoint_query']
        assert q['pre_boxes']-q['pre_empty']==q['admitted_boxes']
        for g in ['all',d['split']]:
            groups[g]['queries_boxes'].update(q)
            for n,c in counts[v].items():groups[g][n].update(c)
    for g,res in r['totals'].items():
        assert groups[g].pop('queries_boxes')==res['queries_boxes']
        assert dict(groups[g])==res['pairs']
    result=dict(passed=True, input_hashes_verified=len(load(RUN/'input_sha256.json')), videos=len(videos), unique_pair_rows=len(unique),
        checks='Independent reclassification from set sizes and intersection; by-video, split and query counts. Worker independently verified intersections against intersected rectangles and bound saved choices/quadrants.',
        report_sha256=digest(RUN/'report.json'))
    with (RUN/'audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(result)
if __name__=='__main__': main()
