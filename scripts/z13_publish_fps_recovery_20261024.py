"""Publish exact service repair files and immutable receipts, not live caches/logs."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_weekend_20261024';BENCH=ROOT/'runs/z13_matched_trackers_20261024'


def main():
    scripts=['z13_matched_fps_recovery_20261024.py','z13_matched_pipeline_recovery2_20261024.py','z13_matched_stock_execution3_20261024.py','z13_matched_stock_queue3_20261024.py','z13_record_matched_trackers_20261024.py','z13_verify_fps_recovery_20261024.py','z13_publish_fps_recovery_20261024.py']
    paths=['scripts/'+n for n in scripts]+['dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md','runs/z13_weekend_20261024/fps_recovery_verification.json']
    old=subprocess.check_output(['git','show','ad33ed4:scripts/z13_record_matched_trackers_20261024.py'],cwd=ROOT)
    snapshot=BENCH/'fps_metadata_recovery/previous_finalizer.py';snapshot.write_bytes(old);paths.append(snapshot.relative_to(ROOT).as_posix())
    receipts=[BENCH/'gpu_stage/controller.json',BENCH/'gpu_stage/controller_result.json',BENCH/'gpu_recovery1/protocol.json',BENCH/'gpu_recovery1/controller.json',BENCH/'gpu_recovery1/controller_result.json',BENCH/'gpu_recovery2/protocol.json',BENCH/'gpu_recovery2/controller.json',BENCH/'gpu_recovery2/launch.json',BENCH/'stock_controls/execution_v3.json',BENCH/'stock_controls/queue2_failure.json',BENCH/'stock_controls/queue3.json',RUN/'matched_failure.json']
    paths += [p.relative_to(ROOT).as_posix() for p in receipts]
    paths += [p.relative_to(ROOT).as_posix() for p in (BENCH/'fps_metadata_recovery').rglob('*.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (BENCH/'gpu_recovery1/jobs').glob('*.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (BENCH/'gpu_recovery2/jobs').glob('*.json') if p.name in {q.name for q in (BENCH/'gpu_recovery1/jobs').glob('*.json')}]
    recovered=BENCH/'gpu_recovery2/jobs/signature_legacyD_b1d22449-15fb948f.json'
    assert recovered.exists()
    for n,h in read(recovered)['outputs'].items():assert sha(ROOT/n)==h
    paths.append(recovered.relative_to(ROOT).as_posix())
    paths=sorted(set(paths));assert all((ROOT/n).is_file() for n in paths)
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(paths)
    assert not (ROOT/'.git/index.lock').exists()
    for left in range(0,len(paths),25):subprocess.run(['git','add','-f','--',*paths[left:left+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Recover matched benchmark from missing FPS metadata without changing science','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    print(json.dumps(dict(commit=commit,paths=len(paths))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    write(RUN/'fps_recovery_publication.json',dict(at=now().isoformat(),commit=commit,paths=paths,push_completed=True))


if __name__=='__main__':main()
