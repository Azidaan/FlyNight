"""Register sources before dispatching only the already-defined final analyses."""
import sys,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    verify_phase('full_precision_execution_phase.json');full=RUN/'full_precision';assert read(full/'additional_technical.json')['passed']
    sources=dict(read(RUN/'full_precision_execution_phase.json')['sources'])
    for name in ['full_post.py','full_precision_statistics.py','full_additional.py','additional_technical.py','start_full_post.py']:
        path=ROOT/'research_identity_assignment_run_20261108'/name;ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [ROOT/'scripts/association_log.py',ROOT/'scripts/output_filter_oracle.py',full/'additional_technical.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_post_registration.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        author_deadline='2026-11-09T07:40:00+03:00',purpose='Sequential statistics and already-registered additional outcome recovery, no new scientific arm.'),new=True)
    with (full/'post.log').open('x',encoding='utf-8') as out,(full/'post.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/full_post.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    import psutil
    write(full/'post_started.json',dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time()),new=True);print(proc.pid)


if __name__=='__main__':main()
