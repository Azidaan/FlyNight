"""Package the narrow optimizer guard as an incremental, byte-checked V3 patch."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=Path(__file__).resolve().parent
OUT=ROOT/'runs/flyvis_repairs_20261106/temporal_numpy_addendum_v3'
assert not OUT.exists()
sha=lambda data:hashlib.sha256(data).hexdigest()
baseline=(EVIDENCE/'before/b1_timestamped.py').read_bytes()
source=(ROOT/'tracking/b1_timestamped.py').read_bytes()
test=(ROOT/'tests/test_b1_timestamped_optimizer_numpy.py').read_bytes()
assert sha(baseline)=='b77636e25d8e7f5d9a9c116f9a257814b021eeb9fd96684327234b70f38ab8c7'
assert sha(source)=='af4e4bb47efbb59eb7e1f7cddad5594436b56bab3077d0a8ecff381f58e81fcd'
V2=ROOT/'runs/flyvis_repairs_20261106/temporal_heads_bundle_v2'
v2=json.loads((V2/'manifest.json').read_bytes())
assert next(row for row in v2['changes'] if row['path']=='tracking/b1_timestamped.py')['after_sha256']==sha(baseline)
assert (V2/'after/tracking/b1_timestamped.py').read_bytes()==baseline
def other_ast(raw):
    tree=ast.parse(raw.decode('utf-8'))
    tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='_finite_optimizer_values')]
    return ast.dump(tree,include_attributes=False)
assert other_ast(baseline)==other_ast(source)
assert '28 failed, 20 passed' in (EVIDENCE/'tests_before_final.log').read_text()
assert '150 passed' in (EVIDENCE/'tests_after.log').read_text()
after_report=json.loads((EVIDENCE/'optimizer_after_report.json').read_bytes())
assert all(not row['restore_success'] and not row['head_changed_on_restore']
    and row['next_step_error'] is None and not row['nonfinite_parameters_after_step']
    and row['RNG_unchanged_by_restore'] and row['deterministic_flags_unchanged'] for row in after_report['checks'])
assert not after_report['cuda_initialized']
OUT.mkdir(parents=True)
changes=[];patches=[]
for name,old,new in [('tracking/b1_timestamped.py',baseline,source),('tests/test_b1_timestamped_optimizer_numpy.py',b'',test)]:
    patch=''.join(difflib.unified_diff(old.decode().splitlines(keepends=True),new.decode().splitlines(keepends=True),
        fromfile='a/'+name if old else '/dev/null',tofile='b/'+name)).encode()
    patch_path=OUT/'patches'/(name.replace('/','__')+'.patch');patch_path.parent.mkdir(exist_ok=True);patch_path.write_bytes(patch)
    for label,raw in [('baseline',old),('after',new)]:
        if raw:
            path=OUT/label/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    changes.append(dict(path=name,baseline_sha256=sha(old) if old else None,after_sha256=sha(new),
        patch=patch_path.relative_to(ROOT).as_posix(),patch_sha256=sha(patch)))
    patches.append(patch)
(OUT/'combined.patch').write_bytes(b''.join(patches))
check=OUT/'isolated_apply_check';check.mkdir()
env=os.environ.copy();env['GIT_CEILING_DIRECTORIES']=str(check.parent)
for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'):env.pop(key,None)
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=check,env=env,capture_output=True).returncode!=0
path=check/'tracking/b1_timestamped.py';path.parent.mkdir();path.write_bytes(baseline)
commands=[]
for arguments in [('apply','--check',str(OUT/'combined.patch')),('apply',str(OUT/'combined.patch'))]:
    result=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf',*arguments],cwd=check,env=env,capture_output=True)
    commands.append(dict(args=list(arguments),code=result.returncode,stderr=result.stderr.decode(errors='replace')))
    assert result.returncode==0
for row in changes:assert sha((check/row['path']).read_bytes())==row['after_sha256']
result=subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply','--reverse',str(OUT/'combined.patch')],cwd=check,env=env,capture_output=True)
commands.append(dict(args=['apply','--reverse','combined.patch'],code=result.returncode,stderr=result.stderr.decode(errors='replace')))
assert result.returncode==0 and path.read_bytes()==baseline and not (check/'tests/test_b1_timestamped_optimizer_numpy.py').exists()
assert (ROOT/'tracking/b1_timestamped.py').read_bytes()==source
assert (ROOT/'tests/test_b1_timestamped_optimizer_numpy.py').read_bytes()==test
manifest=dict(base_bundle='runs/flyvis_repairs_20261106/temporal_heads_bundle_v2',base_manifest_sha256=sha((V2/'manifest.json').read_bytes()),
    changes=changes,combined_patch_sha256=sha((OUT/'combined.patch').read_bytes()),
    unchanged_AST_outside_optimizer_finite_guard=True,commands=commands,
    verification=dict(no_Git_repo_discovered=True,forward_exact=True,reverse_exact=True,production_sources_unchanged_by_packaging=True),
    tests=dict(before='28 failed, 20 passed',after='150 passed',new_cases=48,valid_numpy_resume_exact=14,
        unchanged_native_finite_array_errors=2,existing_float64_reject_cases=4),
    builder_sha256=sha(Path(__file__).read_bytes()),
    boundary='Incremental patch on exact published V2 source. V2 history and saved scientific weights are not rewritten. No temporal policy, frame-gap cap, historical engine or optimizer domain/shape rules changed.')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
readme='''1. Дополнение к опубликованной V2.

Восстановление новой временной ветви Б1 проверяло Python float и тензоры оптимизатора, но пропускало нечисловые NumPy-скаляры и числовые массивы. На синтетическом CPU checkpoint np.float32(NaN) в lr, eps, weight_decay или betas допускался как успешное восстановление, после чего настоящий шаг Adam без исключения делал параметры нечисловыми. Это дефект нашей новой добавки; повреждённые исторические модели этим не обнаружены.

Изменены только три строки проверки конечности NumPy numeric значений до изменения живых весов и оптимизатора. Методы времени, исторический исполнитель, допустимые интервалы, уравнения, веса и правила предметной области оптимизатора не менялись. Пакет применяется поверх точного исходника V2 с SHA256 b77636e25d8e7f5d9a9c116f9a257814b021eeb9fd96684327234b70f38ab8c7. Прежний пакет V2 остаётся историческим материалом; это отдельная поправка, а не его перезапись.

2. Проверки.

Окончательный набор до исправления: 28 failed, 20 passed. После исправления совместно с прежними временными и историческими тестами: 150 passed. Новых случаев 48: 32 проверки отказа NumPy nonfinite, 14 точных сравнений корректного возобновления Adam и две проверки сохранения прежних ошибок PyTorch для конечных 0-D массивов eps/weight_decay. В первом черновике теста эти два массива ошибочно считались допустимыми; первоначальный журнал 30 failed, 18 passed и черновик сохранены локально. Исправлена классификация теста, не расширены правила рабочего кода.

После каждого отказа проверены точные веса, состояние оптимизатора, принадлежность Parameter, состояния генераторов Python/NumPy/PyTorch и флаги детерминированного режима. Повторный настоящий CPU-шаг Adam с неизменным прежним оптимизатором остаётся конечным. Прямое и обратное применение патча в отдельной папке без обнаруженного Git-репозитория воспроизводит точные байты.

3. Дополнительная временная граница.

Отдельная синтетическая проверка подтвердила корректное поведение Python/NumPy целочисленных кадров, отказ неправильных типов и повторных/обратных текущих кадров без изменения eye и часов, нулевую длительность одинаковых повторов истории, сохранность RNG при отказе фабрики. Очень большой абсолютный номер при малом интервале работает; NaN/Inf FPS отвергается. Экстремально малый положительный FPS может завершиться исключением переполнения до выделения памяти.

Для огромного положительного интервала отсутствует отдельный бюджет работы: R1 запрашивает шаг на исходный кадр, C/D выделяет массив интегрирования пропорционального размера. Запросы перехвачены до длительного цикла и большого выделения памяти; реальное зависание не запускалось и не заявляется. Произвольный предел кадров не добавлялся: это отдельный договор ресурсов, который нельзя молча подменить новой временной политикой.

4. Ограничения.

Это CPU-проверки на вновь созданных малых моделях и копиях состояния, без CUDA, новых данных и научного обучения. Нового прогона настоящего FlyVis D после этой трёхстрочной поправки не было; прежний real_D_v5 сохраняет свой исходный хеш. HOTA и отсутствие всех возможных ошибок не заявляются. Финитность метаданных не доказывает корректность любых конечных гиперпараметров, их форм и типов. Полные baseline/after и проверочная копия остаются локальными; публикационный список содержит поправки и доказательства.
'''
(OUT/'README.txt').write_text(readme,encoding='utf-8')
paths=[OUT/'README.txt',OUT/'manifest.json',OUT/'combined.patch',*sorted((OUT/'patches').glob('*.patch')),Path(__file__).resolve()]
paths += [EVIDENCE/name for name in ['before_manifest.json','optimizer_probe.py','optimizer_report.json','optimizer_probe.log',
    'optimizer_types_report.json','optimizer_after_probe.py','optimizer_after_report.json','optimizer_after_probe.log',
    'time_probe.py','time_report.json','time_probe.log','tests_before.log','tests_before_final.log','tests_after.log']]
rows=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p.read_bytes()),size=p.stat().st_size) for p in paths]
(OUT/'publication_manifest.json').write_text(json.dumps(dict(artifacts=rows,excluded=['full baseline/after snapshots','isolated apply workspace','historical weights','diagnostic checkpoints']),indent=2),encoding='utf-8')
names=[row['path'] for row in rows]+[(OUT/name).relative_to(ROOT).as_posix() for name in ['publication_manifest.json','artifact_paths.txt']]
(OUT/'artifact_paths.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
print(json.dumps(dict(changes=changes,manifest_sha256=sha((OUT/'manifest.json').read_bytes()),combined_patch_sha256=manifest['combined_patch_sha256'],artifact_count=len(names)),indent=2))
