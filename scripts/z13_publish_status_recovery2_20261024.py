"""Publish the bounded status-reader repair, not caches or live logs."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
W=ROOT/'runs/z13_weekend_20261024';RUN=ROOT/'runs/z13_val20_matched_20261024'


def main():
    e=read(W/'status_recovery2_retry1_checks.json');assert e['catalog_exit']==e['tests_exit']==0
    assert '60 passed' in (W/'status_recovery2_retry1_tests.log').read_text(encoding='utf-8')
    names=['AGENTS.md','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','scripts/z13_val20_native_masa_recovery2_20261024.py','scripts/z13_record_status_recovery2_20261024.py','scripts/z13_publish_status_recovery2_20261024.py','tests/test_z13_status_read_retry.py']
    names.append('scripts/z13_status_recovery2_checks_20261024.py')
    for stage in (RUN/'native_masa_recovery1',RUN/'native_masa_recovery2'):
        for n in ('protocol.json','controller.json','controller_result.json','price.json','previous_end_audit.json','launch.json','completion.json'):
            if (stage/n).is_file():names.append((stage/n).relative_to(ROOT).as_posix())
        for f in (stage/'jobs').glob('*.json'):names.append(f.relative_to(ROOT).as_posix())
    for n in ('HANDOFF.md','status_recovery2_retry1_checks.json','val20_native_masa_recovery1_failure.json'):names.append((W/n).relative_to(ROOT).as_posix())
    names=sorted(set(names));assert all((ROOT/n).is_file() for n in names);assert not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''};assert staged<=set(names)
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Recover transient Windows status reads without repeating completed jobs','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=commit,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True);write(W/'status_recovery2_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
