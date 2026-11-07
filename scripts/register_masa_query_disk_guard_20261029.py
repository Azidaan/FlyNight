"""Insert explicit disk gate after catalogue checks and before first GPU."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    c=service.read(RUN/'masa_query_preparation/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended']
    assert not (RUN/'jobs_done/010_masa_query_pilot.json').exists()
    folder=RUN/'masa_query_pilot';path=folder/'protocol.json'
    p=service.read(path)
    shutil.copy2(path,folder/'protocol_before_explicit_disk_guard.json')
    script='scripts/run_masa_query_disk_guard_20261029.py'
    p['sha256'][script]=service.sha(ROOT/script)
    p['technical_revision']='Separate wrapper enforces declared first-video disk gate and5GB live reserve; all scientific settings and old worker preserved'
    service.write(path,p)
    job=service.read(RUN/'ready_jobs/010_masa_query_pilot.json')
    job.update(script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(path))
    service.write(RUN/'ready_jobs/010_masa_query_pilot.json',job)
    service.write(RUN/'masa_disk_guard_registration/report.json',dict(passed=True,scientific_settings_unchanged=True,protocol_sha256=service.sha(path),updated=service.now()))

if __name__=='__main__':main()
