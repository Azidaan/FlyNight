"""Publish exact verified milestone files; exclude foreign changes and caches."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024';RUN=ROOT/'runs/z13_val20_matched_20261024'


def main():
    checks=read(W/'val20_milestone_checks.json');assert checks['catalog_exit']==checks['tests_exit']==0
    assert '61 passed' in (W/'val20_milestone_tests.log').read_text(encoding='utf-8')
    assert read(RUN/'raw/verified_milestone.json')['report_sha256']==sha(RUN/'raw/report.json')
    names=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','scripts/z13_val20_common_20261024.py','scripts/z13_val20_common_console.ps1','scripts/z13_val20_common_finish_20261024.py','scripts/z13_val20_milestone_20261024.py','scripts/z13_publish_val20_milestone_20261024.py']
    for stage in (RUN/'inputs_recovery1',RUN/'raw',RUN/'common'):
        for n in ('protocol.json','controller.json','controller_result.json','report.json','finish_recorded.json','price.json','results.md','verified_milestone.json','completion.json','launch.json'):
            if (stage/n).is_file():names.append((stage/n).relative_to(ROOT).as_posix())
        for path in (stage/'jobs').glob('*.json'):names.append(path.relative_to(ROOT).as_posix())
        for path in (stage/'alignment_repairs').glob('*.json'):names.append(path.relative_to(ROOT).as_posix())
    for arm in ('xywh','bytetrack','botsort','tracktrack'):names.append((RUN/'raw'/arm/'per_video.json').relative_to(ROOT).as_posix())
    for n in ('HANDOFF.md','val20_chain_result.json','val20_inputs_verified.json','val20_inputs_vault.json','val20_raw_verified.json','val20_raw_vault.json','val20_milestone_checks.json','val20_milestone_vault.json'):names.append((W/n).relative_to(ROOT).as_posix())
    protected={n.replace('\\','/') for n in read(W/'vault_preparation.json')['protected_existing_edits']}
    rows=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all','--','ObsidianFlyVis'],cwd=ROOT).decode().split('\0')
    for row in rows:
        if row:
            n=row[3:].replace('\\','/')
            if n not in protected and (ROOT/n).is_file():names.append(n)
    names=sorted(set(names));assert all((ROOT/n).is_file() for n in names);assert not (set(names)&protected);assert '.gitignore' not in names and not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Verify fixed20 raw comparison and start matched ten-system evaluation','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=commit,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    write(W/'val20_milestone_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
