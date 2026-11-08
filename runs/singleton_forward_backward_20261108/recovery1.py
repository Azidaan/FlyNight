"""Shape-only interpolation fix over the immutable registered source."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
protocol = json.loads((HERE/'recovery1_protocol.json').read_text(encoding='utf8'))
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['recovery_sha256']
source = HERE/'study.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['original_sha256']
text = source.read_text(encoding='utf8')
before = "transform[:,:,:2],points-transform[:,:,2]"
after = "transform[...,:2,:],points-transform[...,2,:]"
assert text.count(before)==1
text = text.replace(before,after,1)
dispatch = "{'math':math_checks,'register':register,'run':run}[sys.argv[1]]()"
assert text.count(dispatch)==1
check = '''
geometry_path = OUT/'recovery1_geometry_check.json'
if not geometry_path.exists():
    mesh = Delaunay(np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]))
    values = np.column_stack([mesh.points[:,0]+2*mesh.points[:,1],3*mesh.points[:,0]-mesh.points[:,1]])
    q = np.array([[.2,.3],[.4,.1],[.7,.6]])
    errors = []
    for queries in [q,np.repeat(q[:,None,:],19,axis=1)]:
        result,_ = interpolate(mesh,queries,values)
        expected = np.stack([queries[...,0]+2*queries[...,1],3*queries[...,0]-queries[...,1]],axis=-1)
        errors.append(float(np.max(np.abs(result-expected))))
    assert max(errors)<1e-12
    write('recovery1_geometry_check.json',dict(passed=True,shapes=[[3,2],[3,19,2]],errors=errors))
run()
write('execution_receipt.json',dict(recovery_protocol_sha256=sha(OUT/'recovery1_protocol.json'),
    results_sha256=sha(OUT/'results.json'),audit_sha256=sha(OUT/'audit.json'),
    geometry_check_sha256=sha(geometry_path)))
'''
text = text.replace(dispatch,check,1)
exec(compile(text,str(source),'exec'),dict(__file__=str(source),__name__='__main__'))
