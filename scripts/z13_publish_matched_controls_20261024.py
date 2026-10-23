"""Exact owned preparation paths; exclude protected graph edits and live outputs."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/z13_matched_trackers_20261024'


def manifest():
    scripts=['z13_matched_stock_controls_20261024.py','z13_matched_stock_queue_20261024.py','z13_matched_stock_queue_recovery_20261024.py','z13_matched_stock_console.ps1','z13_matched_stock_console_recovery.ps1','z13_matched_pipeline_recovery_20261024.py','z13_record_matched_trackers_20261024.py','z13_record_matched_stock_20261024.py','z13_record_bottleneck_20261024.py','z13_publish_matched_controls_20261024.py']
    paths=['scripts/'+n for n in scripts]+['tests/test_z13_matched_stock_controls.py','tests/test_z13_matched_recovery.py','reports/flyvis_bottleneck_20261024.json','dissertation/ЕДИНЫЙ_ПЛАН.md','experiments/project_history.json']
    fixed=['raw_report.json','controller.json','controller_result.json','vault_stock_prepare.json','stock_controls/protocol.json','stock_controls/launch.json','stock_controls/launch_recovery.json','stock_controls/queue.json','stock_controls/queue_failure.json','stock_controls/queue_recovery.json','gpu_stage/controller.json','gpu_stage/controller_result.json','gpu_stage/price.json','gpu_recovery1/protocol.json','gpu_recovery1/controller.json','gpu_recovery1/launch.json']
    paths += [(RUN/n).relative_to(ROOT).as_posix() for n in fixed]
    for arm in ('xywh','bytetrack','botsort','tracktrack'):
        paths += [p.relative_to(ROOT).as_posix() for p in (RUN/arm/'raw').glob('*.json')]
        paths += [(RUN/arm/n).relative_to(ROOT).as_posix() for n in ('metrics.json','per_video.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (RUN/'gpu_stage/jobs').glob('*.json')]
    protected={Path(n).as_posix() for n in json.loads((RUN/'vault_stock_prepare.json').read_text(encoding='utf-8'))['protected_existing_edits']}
    data=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all','--','ObsidianFlyVis'],cwd=ROOT).decode('utf-8').split('\0')
    for row in data:
        if row:
            n=row[3:].replace('\\','/')
            if n not in protected:paths.append(n)
    paths=sorted(set(paths));assert all((ROOT/n).is_file() for n in paths)
    dest=RUN/'stock_prepare_paths.json';dest.write_text(json.dumps(paths+[dest.relative_to(ROOT).as_posix()],ensure_ascii=False,indent=2),encoding='utf-8')
    return paths+[dest.relative_to(ROOT).as_posix()]


def main():
    paths=manifest();staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0'))-{''};assert staged<=set(paths),'Foreign staged paths'
    assert not (ROOT/'.git/index.lock').exists(),'Lock preserved; inspect before retry'
    for i in range(0,len(paths),25):subprocess.run(['git','add','-f','--',*paths[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    result=subprocess.run(['git','commit','--quiet','-m','Register matched stock FlyVis controls and preserve bounded recovery','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=head,paths=len(paths))))
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)


if __name__=='__main__':main()
