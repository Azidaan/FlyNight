"""Publish exact owned paths; preserve a verified old empty Git lock if needed."""
from pathlib import Path
import csv,io,json,subprocess,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/z13_matched_trackers_20261024'


def git_processes():
    value=subprocess.check_output(['tasklist','/FI','IMAGENAME eq git.exe','/FO','CSV','/NH'],creationflags=subprocess.CREATE_NO_WINDOW).decode(errors='replace')
    return [row for row in csv.reader(io.StringIO(value)) if row and row[0].lower()=='git.exe']


def main():
    lock=(ROOT/'.git/index.lock').resolve();lock.relative_to((ROOT/'.git').resolve())
    if lock.exists():
        for _ in range(100):
            if not git_processes():
                time.sleep(.1)
                if not git_processes():break
            time.sleep(.1)
        else:raise RuntimeError('Active Git processes; lock preserved')
        stat=lock.stat();assert stat.st_size==0 and time.time()-stat.st_mtime>120,'Unexpected/fresh lock; preserved'
        backup=(ROOT/'.git/codex-lock-backups'/('index_20261024_'+datetime.now().strftime('%H%M%S')+'.lock')).resolve();backup.relative_to((ROOT/'.git').resolve());backup.parent.mkdir(exist_ok=True);assert not backup.exists()
        assert not git_processes(),'Git became active; preserving lock';lock.rename(backup)
        (RUN/'git_lock_recovery_20261024.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),old_lock_mtime=stat.st_mtime,bytes=stat.st_size,backup=str(backup),active_git_processes=[],purpose='Author-authorized exact-file commit/push; lock content retained.'),ensure_ascii=False,indent=2),encoding='utf-8')
    paths=json.loads((RUN/'prepare_paths.json').read_text(encoding='utf-8'))
    paths+=['scripts/z13_publish_matched_preparation_20261024.py','tracking/z13_benchmark_adapter.py','tests/test_z13_benchmark_adapter.py','scripts/z13_matched_pipeline_20261024.py','scripts/z13_matched_gpu_console.ps1','runs/z13_matched_trackers_20261024/gpu_stage/protocol.json']
    if (RUN/'git_lock_recovery_20261024.json').exists():paths.append('runs/z13_matched_trackers_20261024/git_lock_recovery_20261024.json')
    paths=sorted(set(paths));staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(paths),'Foreign staged paths; refusing commit'
    for left in range(0,len(paths),30):subprocess.run(['git','add','-f','--',*paths[left:left+30]],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Audit HOTA comparability and register matched tracker benchmark','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)


if __name__=='__main__':main()
