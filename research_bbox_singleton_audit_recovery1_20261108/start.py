import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_singleton_pilot_20261108.common import *
RECOVERY=ROOT/'runs/bbox_singleton_audit_recovery1_20261108'


def main():
    import psutil
    assert read(RUN/'children/audit.ended.json')['all_owned_ended'];probe=read(RECOVERY/'precision_probe.json');assert probe['passed'] and probe['ordinary_point2_gate_decisions_changed']==0
    # Check the independent arithmetic for every recorded edge before resuming.
    from research_bbox_singleton_audit_recovery1_20261108.native_cost import native_cost
    checked=0
    for video in read(INVENTORY/'protocol.json')['videos']:
        for event in read(INVENTORY/'native'/(video+'.json'))['events']:
            for r in event['rows']:
                for c in r['candidates']:assert native_cost(r['row_data']['mean'],c['box'],c['confidence'])==c['cost'];checked+=1
    names=[p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py')]+['runs/bbox_singleton_pilot_20261108/protocol.json','runs/bbox_singleton_pilot_20261108/comparison_report.json','runs/bbox_singleton_pilot_20261108/children/audit.ended.json','runs/bbox_singleton_audit_recovery1_20261108/precision_probe.json'];write(RECOVERY/'protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},scope='Independent audit only; no scientific outputs, labels, parameters or original failure records changed.',reason='Native float32 conversion differs from raw float64 IoU by up to2.1443e-6. Every1845native cost reconstructed bit-exact; noIoU0.2decision changes. Correct the audit model of native arithmetic, not its tolerance.',checked=checked),new=True)
    with (RECOVERY/'controller.log').open('x',encoding='utf-8') as out,(RECOVERY/'controller.err').open('x',encoding='utf-8') as err:child=subprocess.Popen([sys.executable,'-u','research_bbox_singleton_audit_recovery1_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_bbox_singleton_audit_recovery1_20261108/controller.py');write(RECOVERY/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(RECOVERY/'deadline_registered.json',dict(passed=True,root=root),new=True);print(root)


if __name__=='__main__':main()
