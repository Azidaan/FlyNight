import sys,time,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_signals_20261108.common import *


def main():
    import psutil
    phase();names=[p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]+['research_bbox_cost_pilot_20261108/audit_replay.py','research_bbox_singleton_pilot_20261108/audit.py','research_singleton_verify_full_audit_20261108/audit.py','tracking/own_template_verification.py','tracking/native_confirm_identity.py']
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    write(RUN/'comparison_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},scope='Comparison/auditor implementation frozen before restored maps are scored; scientific criteria remain the registered protocol.'),new=True)
    with (RUN/'compare_controller.log').open('x',encoding='utf-8') as out,(RUN/'compare_controller.err').open('x',encoding='utf-8') as err:child=subprocess.Popen([sys.executable,'-u','research_bbox_normalized_signals_20261108/compare_controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_bbox_normalized_signals_20261108/compare_controller.py');write(RUN/'compare_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    print(root)


if __name__=='__main__':main()
