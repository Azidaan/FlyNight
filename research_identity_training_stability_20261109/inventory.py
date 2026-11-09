"""Inventory existing fold models and exact-history image maps; no inference."""
import json,hashlib,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'runs/domain_matched_native_20261109';OLD=ROOT/'runs/full_template_native_20261109';LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision';TRAIN=ROOT/'research_domain_matched_reader_20261109'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
def item(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
cases=read(ROOT/'research_visible_geometry_head_20261109/deployment_screen_protocol.json')['cases']
features=read(TRAIN/'features.json')['records'];folds=read(TRAIN/'predictions.json')['folds'];records=[]
sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),TRAIN/'features.json',TRAIN/'predictions.json',TRAIN/'audit.json',NATIVE/'deployment.json',ROOT/'research_visible_geometry_head_20261109/deployment_screen_protocol.json']}
for case in cases:
    v=case['video'];events_path=NATIVE/'rollouts/C34'/(v+'.events.json.gz');sources.update({events_path.relative_to(ROOT).as_posix():sha(events_path)})
    with gzip.open(events_path,'rt',encoding='utf8') as f:events=json.load(f)['events']
    event=next(e for e in events if e['frame']==case['frame']);q=next(q for q in event['proposals'] if q['old_id']==case['old_id'] and q['raw_id']==case['raw_id'])
    ref=[[h['frame'],h['crop']] for h in q['history']];seqs=[ref,ref+q['search']];available=[]
    receipts={}
    for arm in ['G_R1','G_FLYVIS','G_PERM']:
        path=LEGACY/'rollouts'/arm/(v+'.receipt.json')
        if path.exists():
            receipts.update(read(path).get('feature_files',{}));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for seq in seqs:
        k=key(v,seq);matches=[]
        for base in [NATIVE,OLD,LEGACY]:
            path=base/'features'/v/(k+'.R1.npy')
            if not path.exists():continue
            digest=sha(path);name=path.relative_to(ROOT).as_posix()
            if base==LEGACY:
                assert receipts.get(name)==digest,name
            else:
                meta=path.with_suffix('.json');m=read(meta);assert m['sha256']==digest and m['execution_sha256']==sha(base/'execution_protocol.json')
                assert m['sequence']==seq and m['video']==v
                sources.update({meta.relative_to(ROOT).as_posix():sha(meta),(base/'execution_protocol.json').relative_to(ROOT).as_posix():sha(base/'execution_protocol.json')})
            matches.append(item(path))
        available.append(dict(key=k,sequence=seq,matches=matches))
    records.append(dict(**case,geometry=q['geometry'],channels=q['channels'],logit=q['logit'],reference_box=q['reference_box'],reference_crop=q['reference_crop'],current_box=q['current_box'],current_crop=q['current_crop'],history=ref,search=q['search'],maps=available))
foreign_videos=sorted({r['video'] for r in features if r['label']=='foreign'})
training_composition={v:[{k:r[k] for k in ['video','frame','old_id','raw_id','label']} for r in features if r['video']==v] for v in foreign_videos}
assert len(folds)==80 and len(foreign_videos)==6
ready=all(a['matches'] for r in records for a in r['maps'])
with (OUT/'inventory.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,ready=ready,records=records,foreign_video_groups=training_composition,sources=sources,folds_ready=80,no_training=True,no_core_or_video=True),f,ensure_ascii=False,indent=2)
print('STABILITY_INVENTORY',ready,'readyMaps',sum(bool(a['matches']) for r in records for a in r['maps']),'of',2*len(records),flush=True)
