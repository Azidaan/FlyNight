"""Preserve closed execution evidence and verify no owned stage remains running."""
import study as s
from study import *
import psutil,shutil
assert read(OUT/'final_checks.json')['passed']
check=(OUT/'checks_recovery1.log').read_text(encoding='utf8');assert '"returncode": 0' in check
receipts=[]
for path in sorted(OUT.glob('*.log')):
    for line in path.read_text(encoding='utf8',errors='replace').splitlines():
        if not line.startswith('SYSTEM_MEMORY_GUARD '):continue
        r=json.loads(line.split('SYSTEM_MEMORY_GUARD ',1)[1]);token=r['command'][-1].split('=',1)[1];source=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'/token;dest=OUT/'execution_logs'/path.stem;dest.mkdir(parents=True,exist_ok=False)
        assert r['returncode'] is not None
        for pid,birth in r['owned'].items():
            try:assert psutil.Process(int(pid)).create_time()!=birth,(path,pid)
            except psutil.NoSuchProcess:pass
        for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
            if (source/name).exists():shutil.copyfile(source/name,dest/name)
        receipts.append(dict(stage=path.stem,returncode=r['returncode'],reason=r['reason'],source_token=token))
for name in ['protocol.json','map_protocol.json','comparison_protocol.json','candidate_audit_protocol.json','ownership_protocol.json','context_audit_protocol.json','scientific_receipt.json']:verify(read(OUT/name))
write('execution_receipt.json',dict(passed=True,all_previous_owned_stages_ended=True,stages=receipts,failed_attempts_retained=True,remaining_goal='Open: no independent real ID benefit established, earlier-memory hypothesis not promoted.'))
files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['seal.log','seal.err','execution_manifest.json']}
write('execution_manifest.json',dict(passed=True,created=stamp(),sources=files,root='runs/'+OUT.name,own_open_controller_logs_excluded=['seal.log','seal.err']));print('Sealed',len(files),'closed files',flush=True)
