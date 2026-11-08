"""Read-only diagnosis of the completed free-remainder attempt."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/identity_assignment_20261108';OLD=ROOT/'runs/singleton_free_20261108'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert read(OLD/'independent_audit.json')['passed'];exports=read(BASE/'export_complete.json')['videos'];allrows=[];sources={}
    for split in ['fit20','evaluate26']:
        files={a:OLD/'outcomes'/(split+'_'+a+'.json') for a in ['B0','U_GEO','U_IMAGE','U_FLYVIS']}
        outcomes={a:{r['uid']:r for r in read(path)['rows']} for a,path in files.items()}
        sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in files.values()})
        videos={r['video'] for r in outcomes['U_GEO'].values()};counts=Counter()
        for video in sorted(videos):
            source=ROOT/exports[video]['path'];d=read(source);score=read(BASE/'scores'/(video+'.json'))
            sources[source.relative_to(ROOT).as_posix()]=sha(source)
            with np.load(source.with_suffix('.npz'),allow_pickle=False) as z:
                for row in outcomes['U_GEO'].values():
                    if row['video']!=video:continue
                    snap=d['snapshots'][str(row['frame'])];i=next(k for k,r in enumerate(snap['rows']) if r['tid']==row['tid']);track=snap['rows'][i]
                    if track['state']!=2 or len(track.get('choices',[]))!=2 or 'geometry' not in track:continue
                    valid=[j for j,g in zip(track['choices'],track['geometry']) if g['valid']]
                    if len(valid)!=1:continue
                    j=valid[0];C=z['cost_'+str(row['frame'])];native=snap['native_matches'];free=i not in {a for a,b in native} and j not in {b for a,b in native}
                    positive=C[i,j]<1;col=score['frames'][str(row['frame'])]['columns'][str(j)]
                    kind=row['outcome']['kind'];prior=row['baseline']['kind'];counts['sole_lost_queries']+=1
                    if free and positive:
                        counts['admissible_free_queries']+=1;counts['free_'+kind]+=1
                        counts['free_visual_defined' if col['geometry_valid'] else 'free_visual_undefined']+=1
                        if row['primary_known']:counts['known_free_'+kind]+=1
                    changed=kind!=prior or row['outcome']['raw_id']!=row['baseline']['raw_id']
                    if not row['primary_known']:continue
                    g=track['geometry'][track['choices'].index(j)]
                    allrows.append(dict(uid=row['uid'],video=video,split=split,frame=row['frame'],tid=row['tid'],free=free,positive_iou=bool(positive),
                        geometric_high_index=j,cost=float(C[i,j]),prior=prior,kind=kind,changed=changed,
                        image_kind=outcomes['U_IMAGE'][row['uid']]['outcome']['kind'],flyvis_kind=outcomes['U_FLYVIS'][row['uid']]['outcome']['kind'],
                        visual_defined=col['geometry_valid'],rivals=col['rows'],score=col['scores'],
                        last_seen=track['history'][-1]['frame'],gap=row['frame']-track['history'][-1]['frame'],
                        scale=g.get('scale'),residual=g.get('residual'),current_nodes=g.get('current_nodes'),reference_nodes=g.get('reference_nodes'),
                        candidate_box=snap['boxes'][j],reference_box=track['history'][-1]['box'],target=row['outcome']['target'],identity=row['outcome']['identity']))
        print(split,dict(counts))
    dest=OLD/'free_verification_diagnosis.json';assert not dest.exists()
    result=dict(passed=True,source_sha256=sources,rows=allrows,
        claim='Existing relative score requires at least two rival tracks. Read-only check; absolute identity compatibility has not been measured here.')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in allrows if r['split']=='evaluate26' and r['changed']],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
