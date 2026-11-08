"""Seal successful operator witnesses and dispatch the registered recovery."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    verify_phase('full_precision_technical_phase.json');full=RUN/'full_precision';assert read(full/'technical_gate.json')['passed']
    sources=dict(read(RUN/'full_precision_technical_phase.json')['sources'])
    sources[(full/'technical_gate.json').relative_to(ROOT).as_posix()]=sha(full/'technical_gate.json')
    sources[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    write(RUN/'full_precision_execution_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        prior_technical_registration_sha256=sha(RUN/'full_precision_technical_phase.json'),scientific_changes=False,
        author_deadline='2026-11-09T07:40:00+03:00',guard='System free reserve unchanged. One independently guarded worker per stage.'),new=True)
    with (full/'manager.log').open('x',encoding='utf-8') as out,(full/'manager.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/full_precision_manager.py'],cwd=ROOT,
            stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    import psutil
    write(full/'manager_started.json',dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),registered=datetime.now(MSK).isoformat()),new=True)
    print(proc.pid)


if __name__=='__main__':main()
