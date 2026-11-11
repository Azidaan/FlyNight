"""Extract recorded decisions only, without solving any new assignment."""
from pathlib import Path
import json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sources={}; records=[]
for arm,video,frames,tid in [('G7','b251064f-5f6b663e',[50],37),
                            ('G_DOMAIN','b251064f-5f6b663e',[50],37),
                            ('C34','b1f4491b-cf446195',[258,261,264],1102)]:
    p=ROOT/'runs/domain_matched_native_20261109/rollouts'/arm/(video+'.events.json.gz')
    raw=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    d=json.loads(gzip.decompress(raw))
    for fr in frames:
        ev=next((r for r in d['events'] if r['frame']==fr),None)
        result=dict(arm=arm,video=video,frame=fr,tid=tid,event_saved=ev is not None,
                    assignments=[r for r in d['assignments'] if r['frame']==fr and r['tid']==tid])
        if ev:
            i=next(i for i,r in enumerate(ev['rows']) if r['tid']==tid)
            ps=[r for r in ev['proposals'] if r['old_id']==tid]
            result.update(row=i,history=ev['rows'][i]['history'],columns=[dict(
                raw_id=p['raw_id'],column=p['column'],accepted=p['accepted'],reason=p['reason'],
                original_cost=ev['C'][i][p['column']],corrected_cost=ev['corrected'][i][p['column']],
                native_matched=[m for m in ev['native_first'] if m[0]==i],
                actual_matched=[m for m in ev['actual_first'] if m[0]==i],
                candidates_below_threshold=[dict(tid=r['tid'],cost=ev['corrected'][k][p['column']])
                    for k,r in enumerate(ev['rows']) if ev['corrected'][k][p['column']]<=.8]) for p in ps])
        records.append(result)
for n,h in sources.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
p=OUT/'event_witnesses.json';assert not p.exists()
p.write_text(json.dumps(dict(passed=True,sources=sources,records=records,
    scope='Saved matrices and assignments, no solver call. Missing event at frame261 cannot identify its first association mechanism.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('SAVED_EVENT_WITNESSES',len(records))
