from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=(OUT/'worker.py').read_text(encoding='utf8')
s=s.replace("assert (OUT/'GO.json').exists()","assert (OUT/'GO_v2.json').exists()")
s=s.replace("p=verify();start=time.monotonic()","p=verify();addendum=read(OUT/'protocol_addendum_v2.json');\nfor n,h in addendum['additional_inputs'].items():assert sha(ROOT/n)==h,n\nfor n,h in addendum['new_source_hashes'].items():assert sha(OUT/n)==h,n\nstart=time.monotonic()")
s=s.replace("m=maps[f'index_{ci}_{ei}_C34'].astype(float) if arm=='C34' else maps[f'index_{ci}_{ei}_R1'].astype(float)","m=np.load(ROOT/c['endpoints'][ei]['paths']['C34' if arm=='C34' else 'R1'],allow_pickle=False).astype(float)")
ast.parse(s)
with (OUT/'worker_v2.py').open('x',encoding='utf8') as f:f.write(s)
stage=json.loads((ROOT/'research_stage_localization_20261112/protocol.json').read_text(encoding='utf8'));inputs=sorted({p for c in stage['cases'] for e in c['endpoints'] for p in e['paths'].values()})
d=dict(reason='Baseline provenance mismatch diagnosed before any intervention outcome; original replaymaps slightly differ from archived original endpoints. Replace only map source, preserve frozen heads/query/masks/criteria and1e-9 gate.',original_protocol_sha256=sha(OUT/'proposed_protocol.json'),additional_inputs={p:sha(ROOT/p) for p in inputs},new_source_hashes={'worker_v2.py':sha(OUT/'worker_v2.py')},baseline_diagnostic_sha256=sha(OUT/'baseline_diagnostic.json'),prior_failure_preserved=True,outcomes_before_addendum=False)
with (OUT/'protocol_addendum_v2.json').open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print('ADDENDUM',sha(OUT/'protocol_addendum_v2.json'),'INPUTS',len(inputs))
