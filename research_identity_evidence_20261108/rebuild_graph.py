"""Sequential catalogue validation, copy-first graph update and protected-file proof."""
import os,sys,runpy,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import read,write,sha,datetime,MSK
from research_singleton_verify_20261108.common import guard
RUN=ROOT/'runs/identity_evidence_20261108/graph_update'


def call(path,args):
    previous=sys.argv
    try:
        sys.argv=[path,*map(str,args)]
        try:runpy.run_path(str(ROOT/path),run_name='__main__')
        except SystemExit as e:assert e.code in [None,0]
    finally:sys.argv=previous


def hashes(folder):return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}


def main(obs):
    from scripts.audit_fundamental_obsidian import audit
    from scripts.obsidian_system_problems import load
    assert read(ROOT/'runs/bbox_normalized_signals_20261108/full/queue_result.json')['passed'];assert read(ROOT/'runs/bbox_readout_calibration_20261108/queue_result.json')['passed'];RUN.mkdir(parents=True,exist_ok=True)
    names=['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json','scripts/obsidian_fundamental_overview.py','scripts/obsidian_system_problems.py','scripts/build_obsidian_vault.py','research_identity_evidence_20261108/rebuild_graph.py','dissertation/ЕДИНЫЙ_ПЛАН.md'];before={n:sha(ROOT/n) for n in names}
    write(RUN/'protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources=before,scope='Only existing task3 and existing system problems С1/С2 updated. No scientific calculation or change of hypotheses/P1–P11.'),new=True)
    call('scripts/validate_project_history.py',[])
    import pytest
    assert pytest.main(['-q','tests/test_project_history.py','tests/test_obsidian_fundamental_overview.py','tests/test_obsidian_transfer_outcome.py','tests/test_obsidian_tracking_hypotheses.py','tests/test_obsidian_checked_hypotheses.py','--basetemp',str(RUN/'test_temp')])==0
    load();vault=ROOT/'ObsidianFlyVis';protected=read(ROOT/'runs/identity_assignment_20261108/graph_protected_before.json');current=hashes(vault);write(RUN/'all_files_before.json',current,new=True)
    assert all(current.get(n)==h for n,h in protected.items()),'Protected files changed before graph update; preserve and investigate.'
    preview=RUN/'preview';assert not preview.exists();shutil.copytree(vault,preview);call('scripts/build_obsidian_vault.py',['--vault',preview]);proof=audit(preview)
    assert proof['hypothesis_node_count']==25 and proof['hypothesis_edge_count']==24 and not proof['external_hypothesis_edges'] and not proof['missing_targets']
    copied=hashes(preview);assert set(current)<=set(copied);assert all(copied.get(n)==h for n,h in protected.items())
    generated=set(read(vault/'.generated.json'));allowed=generated|{'.generated.json','.graph-visible.json','.obsidian/graph.json'}
    assert all(copied[n]==h for n,h in current.items() if n not in allowed),[n for n,h in current.items() if n not in allowed and copied.get(n)!=h]
    for n,h in before.items():assert sha(ROOT/n)==h,n
    write(RUN/'preview_audit.json',proof,new=True)
    # Apply only after preview and source stability have passed; retain all files.
    call('scripts/build_obsidian_vault.py',[]);actual=audit(vault);after=hashes(vault);assert set(current)<=set(after);assert all(after.get(n)==h for n,h in protected.items());assert actual['hashes']==proof['hashes']
    for n,h in before.items():assert sha(ROOT/n)==h,n
    write(RUN/'result.json',dict(passed=True,preview_audit_sha256=sha(RUN/'preview_audit.json'),protected_files=len(protected),no_files_removed=True,protected_exact=True,tests_passed=True,visible_notes=actual['note_count'],changed=[n for n in after if n not in current or after[n]!=current[n]],sources=before),new=True)


if __name__=='__main__':main(guard())
