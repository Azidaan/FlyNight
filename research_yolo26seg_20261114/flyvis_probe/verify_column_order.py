"""Read-only check: every saved output type uses the receptor spatial order."""
import os,sys,ast,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path('E:/Dissertation');OUT=Path(__file__).resolve().parent
for variable in ['TEMP','TMP','TMPDIR']:os.environ[variable]=str(OUT/'tmp')
sys.path.append(str(ROOT/'tools/flyvis-runtime'))
import h5py,numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
directory=ROOT/'runs/flyvis/connectome/ConnectomeFromAvgFilters_0000'
paths=[directory/'nodes'/f'{name}.h5' for name in ['u','v','type']]
values=[]
for path in paths:
    with h5py.File(path,'r') as f:values.append(f['data'][:])
u,v,types=values;types=types.astype(str)
tree=ast.parse((ROOT/'tracking/flyvis_activity.py').read_text(encoding='utf8'))
names=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.AnnAssign) and node.target.id=='OUTPUT_CELL_TYPES')
expected=np.asarray([(i,j) for i in range(-15,16) for j in range(max(-15,-15-i),min(15,-i+15)+1)])
rows=[]
for name in names:
    idx=np.flatnonzero(types==name);actual=np.column_stack([u[idx],v[idx]])
    exact=bool(actual.shape==expected.shape and np.array_equal(actual,expected))
    rows.append(dict(cell_type=name,cells=len(idx),uv_order_exact=exact))
    assert exact,name
result=dict(passed=True,cell_types_checked=len(rows),cells_per_type=721,rows=rows,
    connectome_config=(directory/'_meta.yaml').read_text(encoding='utf8'),
    model_config_path='runs/flyvis/results/flow/0000/000/_meta.yaml',
    inputs_sha256={str(path.relative_to(ROOT)):sha(path) for path in paths+[directory/'_meta.yaml',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',ROOT/'tracking/flyvis_activity.py',Path(__file__)]},
    scope='Source extraction uses type_column_indices then transpose; all34 type groups in matching extent15 connectome have exactly same u,v order as BoxEye receptors. No FlyVis/model/Torch import.')
with (OUT/'column_order_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('COLUMN_ORDER_EXACT',len(rows),721)
