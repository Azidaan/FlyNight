"""Publish verified descriptor outcome and scoped CUDA recovery, exact files only."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024';B=ROOT/'runs/z13_matched_trackers_20261024';F=ROOT/'runs/z13_filter_descriptor_fixed_20261024'


def main():
    assert read(W/'fixed_verified.json')['checks_green'] and read(F/'train/finish_recorded.json')['report_sha256']==sha(F/'train/report.json')
    names=['scripts/z13_matched_pipeline_recovery3_20261024.py','scripts/z13_record_matched_recovery3_20261024.py','scripts/z13_matched_stock_execution4_20261024.py','scripts/z13_matched_stock_queue4_20261024.py','scripts/z13_record_cuda_recovery3_20261024.py','scripts/z13_publish_descriptor_result_20261024.py','scripts/build_flyvis_architecture.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    for base in ('cache','train'):
        folder=F/base
        for n in ('protocol.json','report.json','controller.json','controller_result.json','finish_recorded.json','launch.json','launch2.json','reproduction.json','timing.json','seed_metrics.json','cache_manifest.json'):
            if (folder/n).is_file():names.append((folder/n).relative_to(ROOT).as_posix())
    for path in (F/'train').rglob('training.json'):names.append(path.relative_to(ROOT).as_posix())
    for path in (F/'train').rglob('done.json'):names.append(path.relative_to(ROOT).as_posix())
    # Large row tables and models stay local, with explicit content hashes.
    manifest=F/'train/primary_artifacts.json'
    write(manifest,{p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in (F/'train').rglob('*.json') if p.name in ('rows.json','ensemble_rows.json')})
    names.append(manifest.relative_to(ROOT).as_posix())
    for n in ('HANDOFF.md','cuda_recovery3_verification.json','fixed_verified.json','fixed_result.json','fixed_training_launch.json','fixed_vault.json','cuda_recovery3_vault.json','descriptor_architecture_detail.png'):
        if (W/n).is_file():names.append((W/n).relative_to(ROOT).as_posix())
    for n in ('protocol.json','controller.json','launch.json'):
        names.append((B/'gpu_recovery3'/n).relative_to(ROOT).as_posix())
    names += [(B/'gpu_recovery2'/n).relative_to(ROOT).as_posix() for n in ('controller_result.json','cuda_stall_stack.txt')]
    names += [(B/'stock_controls'/n).relative_to(ROOT).as_posix() for n in ('execution_v4.json','queue3_failure.json','queue4.json')]
    names += [(ROOT/'dissertation/architecture'/n).relative_to(ROOT).as_posix() for n in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_sources.json','validation.json')]
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
    subprocess.run(['git','commit','--quiet','-m','Record matched descriptor failure and recover stalled CUDA inference','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=commit,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    write(W/'descriptor_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
