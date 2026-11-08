import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_readout_calibration_20261108.common import *
RECOVERY=RUN/'recovery1'


def main():
    import psutil
    for path in [RUN/'started.json',RUN/'children/verify.started.json']:
        r=read(path)
        try:assert abs(psutil.Process(r['pid']).create_time()-r['created'])>.01,'Do not duplicate a live original worker.'
        except psutil.NoSuchProcess:pass
    assert 'Another audit verification is running; no second worker was started.' in (RUN/'children/verify.err').read_text(encoding='utf-8') or 'Another audit verification is running; no second worker was started.' in (RUN/'children/verify.log').read_text(encoding='utf-8');assert not (RUN/'comparison_report.json').exists()
    files=[p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]+['runs/bbox_readout_calibration_20261108/protocol.json','runs/bbox_readout_calibration_20261108/children/verify.log','runs/bbox_readout_calibration_20261108/children/verify.err'];write(RECOVERY/'protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in files},reason='A second guarded verification was correctly rejected by the global audit mutex while fit input preparation was active. No second worker or threshold calculation started. Wait for data completion, then run the schema validator and unchanged scientific stages sequentially. Mandatory58tests run once in the final graph verification on the final catalogue; no red check is accepted.',science_settings_and_sources_unchanged=True),new=True)
    with (RECOVERY/'controller.log').open('x',encoding='utf-8') as out,(RECOVERY/'controller.err').open('x',encoding='utf-8') as err:child=subprocess.Popen([sys.executable,'-u','research_bbox_readout_calibration_recovery1_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_bbox_readout_calibration_recovery1_20261108/controller.py');write(RECOVERY/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for a in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if a==9:raise
            time.sleep(.1)
    write(RECOVERY/'deadline_registered.json',dict(passed=True,root=root),new=True);print(root)


if __name__=='__main__':main()
