"""Publish only explicit owned sources/records; preserve foreign edits and live logs."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/z13_weekend_20261024'


def paths():
    names=['z13_weekend_session_20261024.py','z13_weekend_driver_20261024.py','z13_weekend_progress.ps1','z13_descriptor_parity_cpu_audit_20261024.py','z13_descriptor_peak_audit_20261024.py','z13_descriptor_numeric_proof_20261024.py','z13_filter_descriptor_fixed_20261024.py','z13_filter_descriptor_fixed_console.ps1','z13_filter_descriptor_fixed_train_20261024.py','z13_filter_descriptor_fixed_train_console.ps1','z13_matched_stock_execution2_20261024.py','z13_matched_stock_queue2_20261024.py','z13_publish_weekend_preparation_20261024.py']
    result=['scripts/'+n for n in names]+['tracking/z13_fixed_candidate_replay.py','tests/test_z13_fixed_candidate_replay.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    folders=[RUN,ROOT/'runs/z13_filter_descriptor_fixed_20261024/cache',ROOT/'runs/z13_filter_descriptor_cache_20261023/parity_audit_cpu_20261024',ROOT/'runs/z13_filter_descriptor_cache_20261023/peak_audit_cpu_20261024']
    for folder in folders:
        for path in folder.rglob('*.json'):
            if path.name in ('status.json',) or path.stat().st_size>2_000_000:continue
            result.append(path.relative_to(ROOT).as_posix())
    result.append((RUN/'HANDOFF.md').relative_to(ROOT).as_posix())
    stock=ROOT/'runs/z13_matched_trackers_20261024/stock_controls'
    result += [(stock/n).relative_to(ROOT).as_posix() for n in ('execution_v2.json','queue2.json','queue_recovery_redirect.json')]
    report=json.loads((RUN/'vault_preparation.json').read_text(encoding='utf-8'));protected={Path(n).as_posix() for n in report['protected_existing_edits']}
    rows=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all','--','ObsidianFlyVis'],cwd=ROOT).decode('utf-8').split('\0')
    for row in rows:
        if row:
            n=row[3:].replace('\\','/')
            if n not in protected:result.append(n)
    result=sorted(set(result));assert all((ROOT/n).is_file() for n in result)
    manifest=RUN/'preparation_paths.json';result.append(manifest.relative_to(ROOT).as_posix());manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result


def main():
    selected=paths();staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0'))-{''};assert staged<=set(selected),'Foreign staged paths; refusing'
    assert not (ROOT/'.git/index.lock').exists(),'Existing lock preserved; inspect before retry'
    for left in range(0,len(selected),25):subprocess.run(['git','add','-f','--',*selected[left:left+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Start bounded weekend FlyVis work and fixed-candidate descriptor trial','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=head,paths=len(selected))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)


if __name__=='__main__':main()
