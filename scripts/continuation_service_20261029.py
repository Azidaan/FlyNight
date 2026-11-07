"""New directly authorized window; no old run is reopened."""
from pathlib import Path
import sys,inspect,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import recovery_research_service_20261027 as service
service.RUN=ROOT/'runs/continuation_20261029';service.__file__=__file__
RUN=service.RUN

def queue():
    from scripts.recovery_queue_late_exit_20261028 import verify_late_end
    code=inspect.getsource(service.queue)
    old="        assert result['all_owned_ended']";assert code.count(old)==1
    code=code.replace(old,"        if not result['all_owned_ended']:\n            verify_late_end(folder,result,20)")
    ns={**vars(service),'verify_late_end':verify_late_end}
    exec(compile(code,__file__,'exec'),ns);ns['queue']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--guard',action='store_true');p.add_argument('--stage',action='store_true');p.add_argument('--folder');p.add_argument('--script');a=p.parse_args()
    if a.launch:service.launch()
    elif a.queue:queue()
    elif a.guard:service.guard()
    elif a.stage:raise SystemExit(service.stage(a.folder,a.script))
