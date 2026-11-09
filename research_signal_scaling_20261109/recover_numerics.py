"""Recover a precision-limited fit without changing objective or audit tolerance."""
import os,sys,json,hashlib,inspect,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
original=ROOT/'tracking/channel_identity_head.py'
assert sha(original)==read(OUT/'protocol.json')['sources']['tracking/channel_identity_head.py']
import tracking.channel_identity_head as head
function=inspect.getsource(head.fit_ridge_logistic)
old="    raise ArithmeticError('Fixed Newton iteration limit reached')"
new="""    loss, gradient, _ = objective(theta, x, y, sw)
    residual = float(np.max(np.abs(gradient)))
    if residual <= 1e-7:
        precision_stops.append(dict(video=current_key[0],arm=current_key[1],kind=current_key[2],gradient_max=residual,iterations=100))
        return dict(theta=theta, loss=loss, gradient_max=residual, iterations=100,
                    class_counts=counts, threshold=0., regularization=1.)
    raise ArithmeticError('Fixed Newton iteration limit reached outside registered audit tolerance')"""
assert function.count(old)==1
ns=dict(head.__dict__,precision_stops=[],current_key=None)
exec(compile(function.replace(old,new),str(OUT/'precision_solver_generated.py'),'exec'),ns)
cache_path=OUT/'checkpoint_fits.json';cache=read(cache_path) if cache_path.exists() else {};events=ns['precision_stops']
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def fit_cached(x,y,video,arm,kind):
    key='|'.join([video,arm,kind]);fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    if key in cache:
        assert cache[key]['input_sha256']==fingerprint
        return cache[key]['fit']
    ns['current_key']=[video,arm,kind];model=ns['fit_ridge_logistic'](x,y)
    cache[key]=dict(input_sha256=fingerprint,fit=cast(model))
    temp=cache_path.with_suffix('.pending');temp.write_text(json.dumps(cache,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    for _ in range(30):
        try:temp.replace(cache_path);break
        except PermissionError:time.sleep(.1)
    else:raise PermissionError(str(cache_path))
    return model
source=OUT/'study.py';text=source.read_text(encoding='utf8')
old='model=fit_ridge_logistic(x[tr],y[tr])';assert text.count(old)==1
text=text.replace(old,'model=fit_cached(x[tr],y[tr],video,arm,kind)')
scope=dict(__name__='__main__',__file__=str(source),fit_cached=fit_cached)
exec(compile(text,str(source),'exec'),scope)
with (OUT/'precision_recovery_result.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,precision_stops=events,checkpoint_models=len(cache),objective_unchanged=True,registered_audit_tolerance=1e-7,production_module_unchanged_sha256=sha(original)),f,indent=2)
