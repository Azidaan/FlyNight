"""Prepare conditional tracker continuation from full original-device audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    source=ROOT/'scripts/b2_tracker_recovery3_20261029.py'
    code=source.read_text(encoding='utf-8')
    code=code.replace('b2_tracker_recovery3','b2_tracker_recovery4')
    code=code.replace('b2_tracker_preparation_recovery3','b2_tracker_preparation_recovery4')
    code=code.replace('b2_sparse_training_audit_recovery3','b2_training_gpu_predictor_audit')
    # Retain explicit CPU loader for the later full tracker audit; its previously
    # registered raw-head tolerance .001 is separate from dev-loss replay 3e-5.
    code=code.replace("('131','b2_tracker_pilot'", "('146','b2_tracker_pilot'")
    anchor="        code=code.replace(\"('130','b2_tracker_pilot'\",\"('146','b2_tracker_pilot'\")"
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+"\n        code=code.replace(\"('135','b2_tracker_audit'\",\"('149','b2_tracker_audit'\").replace(\"('138','b2_tracker_record'\",\"('152','b2_tracker_record'\")")
    anchor="    protocol['price']='Time forecasts"
    # Actual generated protocol records changed device rather than silently
    # claiming that the failed CPU dev-loss audit passed.
    insertion="        code=code.replace(\"    service.write(folder/'protocol.json',protocol)\",\"    protocol['training_audit_device']='cuda, original predictor, separate process; CPU dev-loss equivalence failed and is not claimed'\\n    protocol['failed_cpu_training_audit_sha256']=service.sha(RUN/'b2_sparse_training_audit_recovery3/controller_result.json')\\n    service.write(folder/'protocol.json',protocol)\")\n"
    key="    elif kind=='gpu':"
    assert code.count(key)==1;code=code.replace(key,insertion+key)
    target=ROOT/'scripts/b2_tracker_recovery4_20261029.py';assert not target.exists()
    compile(code,str(target),'exec');target.write_text(code,encoding='utf-8')
    for kind in ['prepare','gpu','audit','record']:
        path=ROOT/f'scripts/b2_tracker_recovery4_{kind}.py'
        assert not path.exists()
        path.write_text("from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))\nfrom scripts.b2_tracker_recovery4_20261029 import main\nif __name__=='__main__':main("+repr(kind)+")\n",encoding='utf-8')
    folder=RUN/'b2_tracker_preparation_recovery4';assert not folder.exists();folder.mkdir()
    p=service.read(RUN/'b2_tracker_preparation_recovery3/protocol.json')
    p.update(scope='Conditional preparation after full1200 independently recomputed GPU losses at unchanged3e-5; failed CPU proof retained, no CPUdevloss equality claim',budget_seconds=86400)
    files=[f'scripts/b2_tracker_recovery4_{k}.py' for k in ['prepare','gpu','audit','record']]+['scripts/b2_tracker_recovery4_20261029.py']
    p['sha256'].update({f:service.sha(ROOT/f) for f in files})
    service.write(folder/'protocol.json',p)
    script='scripts/b2_tracker_recovery4_prepare.py'
    service.write(RUN/'ready_jobs/144_b2_tracker_preparation_recovery4.json',dict(name='144_b2_tracker_preparation_recovery4',folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    print(folder)

if __name__=='__main__':main()
