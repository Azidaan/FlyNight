import ast
from scripts.b2_tracker_recovery3_20261029 import code_for


def price_code():
    code,_=code_for('gpu');tree=ast.parse(code)
    return next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='price' for t in n.targets))


def test_cancelled_time_forecast_preserves_disk_rejection():
    statement=price_code()
    for required,available,disk_need,free,expected in [(1000,1,10,100,True),(1,1000,100,10,False)]:
        namespace=dict(required=required,available=available,disk_need=disk_need,free=free,item={'seconds':1})
        exec(compile(ast.Module(body=[statement],type_ignores=[]),'price','exec'),namespace)
        assert namespace['price']['proceed'] is expected
        assert namespace['price']['stage_time_gate_cancelled']


def test_all_numeric_and_state_assertions_survive_wrapper():
    for kind in ['gpu','audit','record','prepare']:
        code,source=code_for(kind);compile(code,str(source),'exec')
        old_asserts=sum(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(source.read_text(encoding='utf-8'))))
        new_asserts=sum(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(code)))
        assert new_asserts==old_asserts
    audit,_=code_for('audit')
    assert 'load_cpu_branches(' in audit and "p['raw_head_cpu_gpu_tolerance']" in audit
    prepare,_=code_for('prepare')
    assert 'full_chain_fits_before_save=None' in prepare
