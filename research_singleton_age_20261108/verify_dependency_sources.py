"""Bind the analytic dependency argument to the current model and integration code."""
import ast,math
from research_singleton_age_20261108.common import *


def main():
    import yaml
    report=read(RUN/'current_input_dependency.json')
    for name,h in report['sources'].items():assert sha(ROOT/name)==h
    meta_path=ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml';meta=yaml.safe_load(meta_path.read_text(encoding='utf-8'))['config']
    assert meta['network']['connectome']['file']=='fib25-fib19_v2.2.json' and meta['network']['dynamics']['type']=='PPNeuronIGRSynapses'
    build=ast.parse((ROOT/'scripts/z13_target_model.py').read_text(encoding='utf-8'));function=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='build_arm')
    call=next(n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_motion_estimator')
    assert [ast.literal_eval(a) for a in call.args]==['flyvis_tiled',30.,.02]
    module=ast.parse((ROOT/'tracking/flyvis_motion.py').read_text(encoding='utf-8'));function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='integration_plan')
    namespace=dict(math=math);exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'frozen_integration_plan','exec'),namespace)
    steps,dt=namespace['integration_plan'](30.,.02);assert steps==report['current_frame_updates']==2 and abs(steps*dt-1/30)<1e-15
    files=[meta_path,ROOT/'tools/flyvis-runtime/flyvis/network/stimulus.py',ROOT/'tracking/flyvis_checkpoint.py',Path(__file__)]
    write(RUN/'dependency_sources_verified.json',dict(passed=True,report_sha256=sha(RUN/'current_input_dependency.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        model='flow/0000/000',connectome=meta['network']['connectome'],dynamics=meta['network']['dynamics'],source_fps=30,requested_max_dt=.02,actual_dt=dt,steps=steps,
        verified_code_contract='Input buffer reaches only connectome input types. add_edges fills spatial offsets only within listed type pairs. Each Euler velocity uses old-state sources. The selected map is the final state after2updates.',
        new_core_calls=0),new=True)
    assert not (RUN/'comparison_phase.json').exists()
    path=ROOT/'experiments/project_history.json';catalog=read(path);card=next(c for c in catalog['experiments'] if c['id']=='singleton_own_template_v1_20261108');old=card['cannot_claim']
    card.setdefault('revisions',[]).append(dict(date='2026-11-08',commit='',field='cannot_claim',old=old,reason='Проверено, может ли текущий вход достигнуть выбранного выхода в этой схеме считывания.'))
    card['cannot_claim']=old+' В конкретном считывании текущий вход за два обновления ещё не достигает T4/T5. Четыре добавленных правильных условных возврата имеют только один поисковый вход; это не подтверждение узнавания текущей внешности через T4/T5. Возможная польза прошлого пространственного образа остаётся открытой (runs/singleton_age_20261108/current_input_dependency.json; dependency_sources_verified.json).'
    card['evidence'].extend([dict(kind='file',ref='runs/singleton_age_20261108/current_input_dependency.json',claim='Граница причинного объяснения коротких возвратов по коду и готовым данным'),
        dict(kind='file',ref='runs/singleton_age_20261108/dependency_sources_verified.json',claim='Проверены реальная конфигурация модели, шаг интегрирования и источник входного сигнала')]);write(path,catalog)


if __name__=='__main__':main()
