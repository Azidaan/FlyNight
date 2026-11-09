import fix as f
from fix import *
import psutil,shutil
assert read(OUT/'final_checks.json')['passed'];assert '"returncode": 0' in (OUT/'checks.log').read_text(encoding='utf8')
FREE=ROOT/'runs/free_candidate_identity_inventory_20261109';receipts=[]
for folder in [OUT,FREE]:
    for path in folder.glob('*.log'):
        for line in path.read_text(encoding='utf8',errors='replace').splitlines():
            if not line.startswith('SYSTEM_MEMORY_GUARD '):continue
            r=json.loads(line.split('SYSTEM_MEMORY_GUARD ',1)[1]);assert r['returncode']==0 and r['reason'] is None
            token=r['command'][-1].split('=',1)[1];source=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'/token;dest=OUT/'execution_logs'/(folder.name+'_'+path.stem);dest.mkdir(parents=True,exist_ok=False)
            for pid,birth in r['owned'].items():
                try:assert psutil.Process(int(pid)).create_time()!=birth
                except psutil.NoSuchProcess:pass
            for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
                if (source/name).exists():shutil.copyfile(source/name,dest/name)
            receipts.append(dict(folder=folder.name,stage=path.stem,token=token,returncode=r['returncode']))
verify(read(OUT/'fix_protocol.json'));f.verify_before(read(FREE/'protocol.json'))
write('execution_receipt.json',dict(passed=True,all_previous_owned_stages_ended=True,stages=receipts,goal_complete=False))
sources={p.relative_to(ROOT).as_posix():sha(p) for folder in [OUT,FREE] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['seal.log','seal.err','execution_manifest.json']}
sources.update({p:sha(ROOT/p) for p in ['tracking/hex_linear_interpolation.py','tracking/observed_spatial_reader.py','tracking/hex_background_filter.py','tests/test_hex_linear_interpolation.py']})
write('execution_manifest.json',dict(passed=True,sources=sources,own_open_logs_excluded=['seal.log','seal.err']));print('Sealed numerical repair and failed data readiness gate',flush=True)
