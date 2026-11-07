"""Sequential final read-only checks under the system-memory supervisor."""
from pathlib import Path
import json,sys,hashlib,runpy
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from system_memory_guard import guard_entrypoint
job=guard_entrypoint()
mode=sys.argv[1]
if mode=='catalog':
    sys.argv=[str(ROOT/'scripts/validate_project_history.py')]
    runpy.run_path(sys.argv[0],run_name='__main__')
elif mode=='evidence':
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
        return h.hexdigest()
    folder=ROOT/'runs/flyvis_state_subgroups_ram_stream4_20261107'
    m=json.loads((folder/'completion_verified.json').read_bytes())
    assert m['complete'] and m['catalog_passed'] and m['tests_passed']==71
    assert json.loads((folder/'chain_result.json').read_bytes())['all_stages_ended']
    drift={n:dict(before=h,after=sha(folder/n)) for n,h in m['artifacts'].items() if sha(folder/n)!=h}
    assert set(drift)=={'finalize.log','postprocess.stage.json'},drift
    assert (folder/'finalize.log').read_text().strip()=='Completion verified; 64 cases, 71 tests.'
    assert json.loads((folder/'postprocess.stage.json').read_bytes())['stage']=='complete'
    for n,h in m['cases'].items():assert sha(folder/'cases'/(n+'.npz'))==h,n
    assert sha(folder/'frozen_sources.zip')==m['source_archive_sha256']
    recovery=json.loads((folder/'recovery_manifest.json').read_bytes())
    protocol=json.loads((folder/'protocol.json').read_bytes())
    sources=dict(protocol['frozen_sources']);sources.update(recovery['new_sources'])
    for name,h in sources.items():assert sha(ROOT/name)==h,name
    count=0
    for directory in ('tracking','scripts','tests','evaluation'):
        for p in (ROOT/directory).rglob('*.py'):
            compile(p.read_bytes(),str(p),'exec');count+=1
    result=dict(passed=True,case_hashes_checked=len(m['cases']),source_hashes_checked=len(sources),
                stable_artifacts_checked=len(m['artifacts'])-len(drift),mutable_terminal_artifacts=drift,
                explanation='The finalizer captured its own log before printing and the chain status before completing; stable reports and all case arrays match.',
                python_files_compiled=count)
    (job.run_dir/'evidence_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
else:raise ValueError(mode)
