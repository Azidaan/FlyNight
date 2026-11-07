"""Inspect final files and actual owned process identities without running science again."""
import json,sys,shutil,re
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,read,write,sha,GUARD

def main():
    assert read(OUT/'finish_controller.json')['passed']
    owned=read(OUT/'controller_verified.json')['owned']
    owned.update(read(OUT/'finish_owned_snapshot.json'))
    checks=[]
    for prefix in ['', 'final_']:
        for stage in ['validator','catalog','graph']:
            path=OUT/(prefix+stage+'.log')
            if not path.exists():continue
            line=next(t for t in path.read_text(encoding='utf8').splitlines() if t.startswith('SYSTEM_MEMORY_GUARD '))
            report=json.loads(line[len('SYSTEM_MEMORY_GUARD '):])
            assert report['reason'] is None and report['returncode']==0
            owned.update({str(k):v for k,v in report['owned'].items()})
            token=report['command'][-1].split('=',1)[1];run=GUARD/'uncapped_checks'/token
            supervisor=read(run/'supervisor.json');owned[str(supervisor['pid'])]=supervisor['birth']
            body=(run/'memory_guard.worker.log').read_text(encoding='utf8',errors='replace')
            shutil.copyfile(run/'memory_guard.worker.log',OUT/(prefix+stage+'_worker.log'))
            counts=re.findall(r'(\d+) passed in ([\d.]+)s',body)
            checks.append(dict(stage=prefix+stage,code=0,pytest=counts,minimum_available_bytes=report['minimum_available_bytes'],guard_sha256=sha(run/'memory_guard.json')))
    for pid,birth in owned.items():
        try:assert psutil.Process(int(pid)).create_time()!=birth,('owned_still_alive',pid)
        except psutil.NoSuchProcess:pass
    names={p.name for p in Path('scripts').glob('*flyvis_two_frame_input*20261107.py')}
    names.discard(Path(__file__).name)
    remaining=[]
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        args=process.info['cmdline'] or []
        if any(Path(arg).name in names for arg in args[1:]):
            remaining.append(process.info)
    assert not remaining,remaining
    protocol=read(OUT/'protocol.json')
    for fn,digest in protocol['source_sha256'].items():assert sha(fn)==digest,fn
    write(OUT/'checks_summary.json',dict(passed=True,all_owned_ended=True,owned=owned,remaining=remaining,checks=checks,
        protocol_sha256=sha(OUT/'protocol.json'),quality_audit_sha256=sha(OUT/'independent_audit.json'),recovery_sha256=sha(OUT/'recovery1_controller.json'),graph_sha256=sha(OUT/'graph_live_verified.json')))
    print('Verified all owned identities ended;',len(owned),'identities;',checks)

if __name__=='__main__':main()
