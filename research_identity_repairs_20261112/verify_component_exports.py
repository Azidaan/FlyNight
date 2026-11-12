from common import *
import ast,numpy as np
from tracking.reference_template_history import terminal_reference_history
pairs=[('geometry832/registration.py','tracking/census_correspondence.py'),('geometry832/selection.py','tracking/correspondence_repair_guard.py'),('recovery574/appearance_recovery_policy_v4.py','tracking/appearance_recovery_policy.py')]
checks=[]
for source,target in pairs:
 assert sha(OUT/source)==sha(ROOT/target);checks.append(dict(source=source,target=target,sha256=sha(ROOT/target)))
def body(s):
 tree=ast.parse(s);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))];return ast.dump(tree,include_attributes=False)
assert body((OUT/'review/combined_adapter_v2.py').read_text(encoding='utf8'))==body((ROOT/'tracking/identity_repair_adapter.py').read_text(encoding='utf8'))
history=[]
with np.load(ROOT/'research_terminal_response_probe_20261112/expanded_inputs.npz') as z:
 for ci in range(5):
  original=z[f'index_{ci}_0'];expected=z[f'terminal_hold_{ci}_0'];actual=terminal_reference_history(original)
  assert np.array_equal(actual,expected),(ci,actual.shape,expected.shape)
  history.append(dict(case_index=ci,original_shape=list(original.shape),held_shape=list(actual.shape),exact=True))
save('component_export_checks.json',dict(passed=True,exact_source_exports=checks,adapter_nonimport_AST_exact=True,prepared_reference_histories=history,new_model_calls=0,source_inputs_sha256=sha(ROOT/'research_terminal_response_probe_20261112/expanded_inputs.npz')))
print('COMPONENT_EXPORTS_AND_FIVE_HISTORY_INPUTS_EXACT')
