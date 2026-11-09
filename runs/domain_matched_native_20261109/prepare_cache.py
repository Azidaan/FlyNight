"""Copy only verified reusable arrays into the new run, never alter old assets."""
import os,sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(900,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
phase=read(OUT/'execution_protocol.json');registry=read(OUT/'cache_registry.json')
for name,digest in phase['sources'].items():assert sha(ROOT/name)==digest,name
copied={};execution=sha(OUT/'execution_protocol.json')
for item in registry['assets'].values():
    source=ROOT/item['source'];dest=OUT/item['target'];assert source.resolve().is_relative_to(ROOT) and dest.resolve().is_relative_to(OUT)
    assert sha(source)==item['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(source,dest)
    assert sha(dest)==item['sha256']
    copied[dest.relative_to(ROOT).as_posix()]=item['sha256']
    if item['kind']=='features':
        original=ROOT/item['metadata'];assert sha(original)==item['metadata_sha256'];m=read(original)
        m.update(execution_sha256=execution,reused_from=item['source'],reused_source_sha256=item['sha256'],original_metadata_sha256=item['metadata_sha256'],original_execution_sha256=m['execution_sha256'])
        meta=dest.with_suffix('.json')
        if meta.exists():assert read(meta)==m
        else:
            with meta.open('x',encoding='utf8') as f:json.dump(m,f,ensure_ascii=False,indent=2);f.write('\n')
for name,digest in phase['sources'].items():assert sha(ROOT/name)==digest,name
with (OUT/'cache_preparation.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,arrays=len(copied),sources=copied,new_execution_sha256=execution,old_arrays_unchanged=True,zero_inference=True),f,indent=2)
print('CACHE_REUSED',len(copied),'arrays, no inference',flush=True)
