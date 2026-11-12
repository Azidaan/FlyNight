from common import *
import ast
assert read(OUT/'recovery574/audit_episode.json')['passed']
assert read(OUT/'review/combined_assignment_result_v2.json')['passed']
src=OUT/'recovery574/appearance_recovery_policy_v4.py';dst=ROOT/'tracking/appearance_recovery_policy.py';assert not dst.exists();dst.write_bytes(src.read_bytes());assert sha(dst)==sha(src)
s=(OUT/'review/combined_adapter_v2.py').read_text(encoding='utf8')
new=s.replace('from appearance_recovery_policy_v4 import','from tracking.appearance_recovery_policy import').replace('from selection import','from tracking.correspondence_repair_guard import')
assert s!=new
def body(text):
 tree=ast.parse(text);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))];return ast.dump(tree,include_attributes=False)
assert body(s)==body(new)
adapter=ROOT/'tracking/identity_repair_adapter.py';assert not adapter.exists();adapter.write_bytes((new.rstrip()+'\n').encode())
save('installed_components_v2.json',dict(installed=now(),policy=dict(source=str(src.relative_to(ROOT)),target=str(dst.relative_to(ROOT)),sha256=sha(dst)),adapter=dict(source='research_identity_repairs_20261112/review/combined_adapter_v2.py',target='tracking/identity_repair_adapter.py',sha256=sha(adapter),only_import_names_changed=True,nonimport_AST_exact=True),reference_history=dict(target='tracking/reference_template_history.py',sha256=sha(ROOT/'tracking/reference_template_history.py')),activation='Reusable guarded components tested on saved cases only. No live tracker hook enabled, no full rollout.'))
print('INSTALLED_GUARDED_REPAIR_COMPONENTS')
