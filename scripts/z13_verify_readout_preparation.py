"""Verify preserved sources and foreign edits, and that only two new cards were added."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    current=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    before=json.loads(subprocess.check_output(['git','show','HEAD:experiments/project_history.json'],cwd=ROOT).decode('utf-8'))
    cards={c['id']:c for c in current['experiments']}
    assert all(cards[c['id']]==c for c in before['experiments'])
    new=set(cards)-{c['id'] for c in before['experiments']}
    assert new=={'z13_readout_overlap_v1','z13_target_center_v1'}
    for name in ('z13_readout_overlap_20261018','z13_target_center_20261018'):
        p=json.loads((ROOT/'runs'/name/'protocol.json').read_text(encoding='utf-8'))
        for rel,digest in p.get('sha256',p.get('sources_sha256')).items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    preservation=json.loads((ROOT/'runs/z13_readout_overlap_20261018/vault_preservation.json').read_text(encoding='utf-8'))
    for rel,digest in preservation['protected_existing_edits'].items():
        p=ROOT/rel
        if digest is None:assert not p.exists(),rel
        else:assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
    run=ROOT/'runs/z13_target_center_20261018'
    result=json.loads((run/'recovery_benchmark_result.json').read_text())
    assert result['exit_code']==0 and result['worker_ended']
    assert not (run/'report.json').exists() and not (run/'recovery_full.json').exists()
    report=dict(original_cards_unchanged=True,new_cards=sorted(new),frozen_sources_unchanged=True,
                preserved_graph_states=len(preservation['protected_existing_edits']),
                cpu_timing_completed=True,full_experiment_not_started=True)
    (ROOT/'runs/z13_readout_overlap_20261018/verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
