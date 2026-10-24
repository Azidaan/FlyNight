"""Publish complete matched HOTA, partial stock record and CPU pilot sources."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024';B=ROOT/'runs/z13_matched_trackers_20261024';N=ROOT/'runs/z13_competing_candidates_20261024'


def main():
    assert read(W/'matched_recovery3_verified.json')['checks_green'] and read(B/'finish_recorded.json')['report_sha256']==sha(B/'report.json')
    assert 'OK: каталог согласован' in (W/'heartbeat_catalog_0442.log').read_text(encoding='utf-8') and '62 passed' in (W/'heartbeat_tests_0442.log').read_text(encoding='utf-8')
    names=['AGENTS.md','experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    scripts=['z13_matched_stock_recovery1_20261024.py','z13_record_stock_recovery1_20261024.py','z13_after_stock_recovery1_20261024.py','z13_record_matched_milestone_20261024.py','z13_competing_candidates_20261024.py','z13_record_competing_candidates_20261024.py','z13_after_competing_candidates_20261024.py','z13_competing_candidates_console.ps1','z13_competing_recovery1_20261024.py','z13_competing_recovery2_20261024.py','z13_cpu_visibility_diagnostic_20261024.py','z13_competing_native_diagnostic_20261024.py','z13_publish_matched_and_next_20261024.py']
    names+=['scripts/'+n for n in scripts]+['tracking/z13_competing_candidates.py','tests/test_z13_competing_candidates.py','tests/test_z13_competing_execution.py']
    names+=['scripts/'+n for n in ('z13_val20_inputs_20261024.py','z13_val20_inputs_recovery1_20261024.py','z13_val20_inputs_console.ps1','z13_val20_raw_20261024.py','z13_val20_raw_console.ps1','z13_val20_queue_20261024.py','z13_val20_queue_recovery2_20261024.py','z13_val20_alignment_audit_20261024.py','z13_val20_alignment_native_audit_20261024.py','build_flyvis_architecture.py')]
    names.append('scripts/z13_heartbeat_checks_20261024.py')
    for p in (ROOT/'dissertation/architecture').iterdir():
        if p.is_file():names.append(p.relative_to(ROOT).as_posix())
    V=ROOT/'runs/z13_val20_matched_20261024'
    names.append((V/'selection.json').relative_to(ROOT).as_posix())
    for folder in (V/'inputs',V/'inputs_recovery1'):
        for n in ('protocol.json','controller.json','controller_result.json','price.json','finish_recorded.json','alignment_failure_audit.json','alignment_native_audit.json','queue_redirect_v2.json','queue_execution_v2.json'):
            if (folder/n).is_file():names.append((folder/n).relative_to(ROOT).as_posix())
    for n in ('report.json','seed_metrics.json','cache_manifest.json'):
        names.append((N/'recovery2'/n).relative_to(ROOT).as_posix())
    for n in ('competing_recovery2_verified.json','competing_recovery2.json','competing_recovery2_vault.json','val20_inputs_partial_verified.json','val20_inputs_partial_vault.json','heartbeat_vault_0444.json'):
        if (W/n).is_file():names.append((W/n).relative_to(ROOT).as_posix())
    for n in ('protocol.json','raw_report.json','report.json','results.md','results.csv','verified_summary.json','finish_recorded.json'):
        names.append((B/n).relative_to(ROOT).as_posix())
    for folder in (B/'gpu_recovery3',B/'stock_controls',B/'stock_recovery1',N,N/'recovery1',N/'recovery2'):
        for n in ('protocol.json','controller.json','controller_result.json','price.json','finish_recorded.json','launch.json','completion.json','reproduction.json','native_stage.json','timing.json','cpu_visibility_diagnostic.json'):
            if (folder/n).is_file():names.append((folder/n).relative_to(ROOT).as_posix())
        for marker in (folder/'jobs').glob('*.json'):names.append(marker.relative_to(ROOT).as_posix())
    for n in ('matched_recovery3_verified.json','matched_recovery3.json','matched_recovery3_vault.json','stock_failure.json','stock_recovery1_failure.json','matched_milestone_verification.json','competing_candidates_failure.json','competing_recovery1_failure.json','HANDOFF.md','matched_and_next_stage_vault.json'):
        if (W/n).is_file():names.append((W/n).relative_to(ROOT).as_posix())
    artifact=B/'primary_matched_artifacts.json';r=read(B/'report.json');write(artifact,dict(report_sha256=sha(B/'report.json'),output_sha256=r['output_sha256'],signature_and_masa_jobs={p.name:read(p)['outputs'] for p in (B/'gpu_recovery3/jobs').glob('*.json')}));names.append(artifact.relative_to(ROOT).as_posix())
    protected={n.replace('\\','/') for n in read(W/'vault_preparation.json')['protected_existing_edits']}
    rows=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all','--','ObsidianFlyVis'],cwd=ROOT).decode().split('\0')
    for row in rows:
        if row:
            n=row[3:].replace('\\','/')
            if n not in protected and (ROOT/n).is_file():names.append(n)
    names=sorted(set(names));assert all((ROOT/n).is_file() for n in names)
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    for left in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[left:left+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Record matched HOTA benchmark and bounded next readout pilot','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=commit,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    write(W/'matched_recovery3_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,report_sha256=sha(B/'report.json'),paths=names))


if __name__=='__main__':main()
