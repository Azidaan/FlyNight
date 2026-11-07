"""Close the run with immutable source checks and actual PID/birth verification."""
import sys,json,re,shutil
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,GUARD,read,write,sha,verify
if __name__=='__main__':
    finish=read(OUT/'finish_controller.json');assert finish['passed']
    pipeline=read(OUT/'controller.json');assert pipeline['passed']
    owned={**pipeline['owned'],**finish['owned']};checks=[]
    for launch in ['launch.json','finish_launch.json']:
        item=read(OUT/launch);owned[str(item['pid'])]=item['birth']
    for prefix,stages in [('', ['unit','validator','catalog','data','train','audit']),('final_',['validator','catalog','graph'])]:
        for stage in stages:
            path=OUT/(prefix+stage+'.log')
            line=next(t for t in path.read_text(encoding='utf8').splitlines() if t.startswith('SYSTEM_MEMORY_GUARD '))
            report=json.loads(line[len('SYSTEM_MEMORY_GUARD '):]);assert report['reason'] is None and report['returncode']==0
            owned.update({str(k):v for k,v in report['owned'].items()})
            token=report['command'][-1].split('=',1)[1];guard_dir=GUARD/'uncapped_checks'/token
            supervisor=read(guard_dir/'supervisor.json');owned[str(supervisor['pid'])]=supervisor['birth']
            source=guard_dir/'memory_guard.worker.log';body=source.read_text(encoding='utf8',errors='replace')
            shutil.copyfile(source,OUT/(prefix+stage+'_worker.log'))
            checks.append(dict(stage=prefix+stage,code=0,pytest=re.findall(r'(\d+) passed in ([\d.]+)s',body),
                minimum_available_bytes=report['minimum_available_bytes'],guard_sha256=sha(guard_dir/'memory_guard.json')))
    for pid,birth in owned.items():
        try:assert psutil.Process(int(pid)).create_time()!=birth,('owned_still_alive',pid)
        except psutil.NoSuchProcess:pass
    names={'flyvis_readout_comparison_20261107.py','chain_flyvis_readout_comparison_20261107.py','check_flyvis_readout_comparison_20261107.py','finish_flyvis_readout_comparison_20261107.py'}
    remaining=[]
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        if any(Path(arg).name in names for arg in (process.info['cmdline'] or [])[1:]):remaining.append(process.info)
    assert not remaining,remaining
    verify(read(OUT/'protocol.json'))
    write(OUT/'checks_summary.json',dict(passed=True,all_owned_ended=True,owned=owned,remaining=remaining,checks=checks,
        core_and_frozen_sources_unchanged=True,protocol_sha256=sha(OUT/'protocol.json'),audit_sha256=sha(OUT/'independent_audit.json'),
        graph_sha256=sha(OUT/'graph_live_verified.json'),conditional_full_tracker_started=False,reason='Registered choice improvement gate failed'))
    print('All source fingerprints and',len(owned),'owned identities verified;',checks,flush=True)
