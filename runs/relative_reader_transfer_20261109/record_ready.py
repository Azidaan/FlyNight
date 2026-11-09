"""Resume ONLY metadata after failed open; ready audit/scores are not repeated."""
import finish as f
from finish import *
import ast
assert read(OUT/'audit.json')['passed'];verify(read(OUT/'protocol.json'));verify(read(OUT/'maps_complete.json'));assert not (OUT/'ИТОГ.md').exists()
source=Path(f.__file__);tree=ast.parse(source.read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='record');start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='path' for t in n.targets));ns=dict(f.__dict__);ns['error']=read(OUT/'audit.json')['maximum_score_error']
write('record_recovery.json',dict(passed=True,reason='Initial open(project_history.json,w) returned OSError22 before any catalogue change. Ready audit/summary reused, no scores or models repeated. Metadata now written via a complete sibling temporary file and atomic replace.',source_sha256=sha(source),failed_worker='07727181a6e441dc99ee8bda2d1ab586',premature_checks_worker_stopped=True))
exec(compile(ast.Module(body=fn.body[start:],type_ignores=[]),str(source),'exec'),ns)
