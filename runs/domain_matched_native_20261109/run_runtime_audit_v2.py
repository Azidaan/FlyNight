"""Correct only the audit's schema read; execute all frozen checks unchanged."""
import ast,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
recovery=json.loads((OUT/'audit_schema_recovery_protocol.json').read_text(encoding='utf8'))
for name,digest in recovery['sources'].items():assert sha(ROOT/name)==digest,name
source=OUT/'audit_runtime.py';tree=ast.parse(source.read_text(encoding='utf8'));changed=0
class Repair(ast.NodeTransformer):
    def visit_Compare(self,node):
        global changed
        if ast.unparse(node)=="model['independent_gradient_max'] < 1e-07":
            changed+=1;return ast.copy_location(ast.parse("all(v < 1e-7 for v in model['independent_gradient_max'].values())",mode='eval').body,node)
        return self.generic_visit(node)
tree=Repair().visit(tree);assert changed==1;ast.fix_missing_locations(tree)
namespace=dict(__name__='__main__',__file__=str(source));exec(compile(tree,str(source),'exec'),namespace)
if not (OUT/'runtime_audit_recovery.json').exists():
    with (OUT/'runtime_audit_recovery.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,recovery_sha256=sha(OUT/'audit_schema_recovery_protocol.json'),audit_sha256=sha(OUT/'runtime_audit.json'),only_schema_check_changed=True,no_scientific_result_recomputed=True),f,indent=2)
