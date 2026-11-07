"""Новый разрешённый запуск только последнего видео; готовые19не повторяются."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,hashlib,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import service,RUN
read,write,sha,now=service.read,service.write,service.sha,service.now

def worker_source():
    from scripts.run_b1_tracker_continuation1_20261028 import continuation_source
    return continuation_source().replace('if frame % 25 == 0:','if frame % 5 == 0:')

def main():
    from scripts import run_b1_tracker_20261028 as original
    import tracking.b1_tracker_runtime as runtime
    import tracking.foveated_eyes as eyes
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    from tracking.b1_cached_eye import CachedCanonicalEyeBank
    import numpy as np
    folder=RUN/'b1_tracker';p=read(folder/'protocol.json')
    pre=read(RUN/'preflight/report.json');assert pre['passed']
    c=read(RUN/'preflight/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    old=ROOT/'runs/recovery_research_20261027/b1_tracker/continuation_recovery1'
    c=read(old/'controller_result.json');assert not any(service.matching(k,v) for k,v in c['owned'].items())
    runtime.ResidualEngine=DeterministicResidualEngine
    class CheckedBank(CachedCanonicalEyeBank):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            with np.load(folder/'geometry_seed.npz') as z:
                assert np.array_equal(self.points_local,z['points']) and self.eye_size==int(z['eye_size'])
            from scripts.run_b1_tracker_continuation_recovery1_20261028 import renderer_parity
            renderer_parity(self,folder)
    eyes.EyeBank=CheckedBank
    def reuse_regression(engines,target):
        proof_path=ROOT/'runs/recovery_research_20261027/b1_tracker/recovery1/incremental_regression.json'
        proof=read(proof_path);assert proof['passed']
        write(target/'incremental_regression.json',dict(proof,reused_completed_proof=True,source=proof_path.relative_to(ROOT).as_posix(),source_sha256=sha(proof_path)))
    code=worker_source();assert hashlib.sha256(code.encode()).hexdigest()==p['executed_worker_sha256']
    ns={**vars(original),'__name__':'b1_last_video_worker'}
    exec(compile(code,str(ROOT/'scripts/run_b1_tracker_20261028.py'),'exec'),ns)
    def watched_write(path,data,required=True):
        if Path(path).name=='status.json':
            faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(600,repeat=False,file=sys.stderr,exit=True)
        return write(path,data,required=required)
    ns.update(FOLDER=folder,regression=reuse_regression,write=watched_write)
    faulthandler.enable(file=sys.stderr);faulthandler.dump_traceback_later(600,repeat=False,file=sys.stderr,exit=True)
    try:ns['run']()
    finally:faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
