from pathlib import Path
import importlib.util,sys
root=Path.cwd();out=root/'runs/flyvis_repairs_20261106/loading_constraint_followup'
sys.path.insert(0,str(root))
for name in ('flyvis_checkpoint','flyvis_connectome_swap'):
 spec=importlib.util.spec_from_file_location('tracking.'+name,out/'before'/f'{name}.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
import pytest
raise SystemExit(pytest.main(['tests/test_flyvis_checkpoint_strict.py','tests/test_flyvis_connectome_transfer.py','-q','-p','no:cacheprovider','--basetemp='+str(out/'pytest_before')]))
