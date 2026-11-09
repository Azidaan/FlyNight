"""Run the frozen phase without letting local queue.py shadow the standard library."""
import sys,os,json,hashlib,sysconfig
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
allowed={'compute.py','audit.py','record_result.py','check_records.py'}
if len(sys.argv)>1 and sys.argv[1] in allowed:
    os.environ['DOMAIN_MATCHED_SAFE_TARGET']=sys.argv.pop(1)
name=os.environ.get('DOMAIN_MATCHED_SAFE_TARGET');assert name in allowed
passport=json.loads((HERE/'import_recovery_protocol.json').read_text(encoding='utf8'))
for path,digest in passport['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=HERE]
import queue
assert Path(queue.__file__).resolve()==Path(sysconfig.get_path('stdlib'),'queue.py').resolve(),queue.__file__
target=HERE/name
namespace={'__name__':'__main__','__file__':str(target),'__package__':None,'__builtins__':__builtins__}
exec(compile(target.read_text(encoding='utf8'),str(target),'exec'),namespace)
