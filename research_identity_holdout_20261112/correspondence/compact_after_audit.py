"""Publish small score/registration facts after full independent audit only."""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read('audit.json');assert audit['passed'];r=read('result.json');assert r['passed'];p=read('protocol.json');assert audit['selection_sha256']==p['selection_sha256']
records=[{k:v for k,v in z.items() if k not in ['channels','patch_key']} for z in r['records']]
registration=[]
for z in r['registrations']:
    item={k:v for k,v in z.items() if k not in ['source_pixels','candidate_translations','cost_tuples']}
    item['source_pixel_count']=len(z.get('source_pixels',[]));item['candidate_count']=len(z.get('candidate_translations',[]));item['winning_cost_tuple']=z['cost_tuples'][z['winner']] if z['ready'] else None
    if z['ready']:
        w=item['winning_cost_tuple'];item['median_ties']=sum(c[0]==w[0] for c in z['cost_tuples']);item['median_sum_ties']=sum(c[:2]==w[:2] for c in z['cost_tuples'])
    registration.append(item)
out=dict(passed=True,source=dict(path='research_identity_holdout_20261112/correspondence/result.json',sha256=sha(OUT/'result.json'),bytes=(OUT/'result.json').stat().st_size,audit_path='research_identity_holdout_20261112/correspondence/audit.json',audit_sha256=sha(OUT/'audit.json'),selection_sha256=p['selection_sha256']),records=records,registrations=registration,baseline_witness=r['baseline_witness'],core_result_sha256=r['core_result_sha256'],samples_sha256=r['samples_sha256'],frame_hashes=r['frame_hashes'],elapsed_seconds=r['elapsed_seconds'],new_core_calls=r['new_core_calls'],new_fits=r['new_fits'],tracker_calls=r['tracker_calls'],GT_used_for_decision=False,omitted='Per-channel samples, all shift costs/candidatecoordinates/pixelindices remain in immutable full source and samples; no score or case omitted.',scope='Readout decisions and registration diagnostics, not identity correctness or fullmatrix result.')
with (OUT/'compact_result.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('COMPACT_WRITTEN',len(records),len(registration),(OUT/'compact_result.json').stat().st_size)
