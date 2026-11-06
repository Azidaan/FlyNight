"""Static review aid: no imported project modules, GPU or native mutexes."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
inventory = json.loads((OUT / 'inventory.json').read_text(encoding='utf-8'))
rows = []
for item in inventory['files']:
    if not item['locks']:
        continue
    source = (ROOT / item['path']).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {name: {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
             for name, node in functions.items()}
    spawn_functions = {x['scope'] for x in item['process_calls'] + item['thread_calls']}
    roots = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(isinstance(x.context_expr, ast.Call) and ast.unparse(x.context_expr.func).endswith('GpuLock') for x in node.items):
            continue
        immediate = {ast.unparse(n.func) for statement in node.body for n in ast.walk(statement) if isinstance(n, ast.Call)}
        reached = set(immediate) & functions.keys()
        pending = list(reached)
        while pending:
            name = pending.pop()
            for extra in (calls[name] & functions.keys()) - reached:
                reached.add(extra)
                pending.append(extra)
        roots.append(dict(line=node.lineno, end_line=node.end_lineno, direct_calls=sorted(immediate),
                          reachable_same_module_functions=sorted(reached),
                          reachable_spawn_functions=sorted(reached & spawn_functions)))
    rows.append(dict(path=item['path'], sha256=item['sha256'], lock_regions=roots,
                     process_functions=sorted(spawn_functions)))
report = dict(scope='AST direct calls plus same-module transitive function calls; no dynamic/imported call proof',
              reviewed=rows,
              lock_reachable_spawn_functions=[dict(path=r['path'], region=x) for r in rows for x in r['lock_regions'] if x['reachable_spawn_functions']],
              direct_process_under_lock=[dict(path=r['path'], call=x) for r in inventory['files'] for x in r['process_calls']
                                         if any(any('GpuLock(' in s for s in w['contexts']) for w in x['enclosing_with'])])
(OUT / 'call_graph.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(files=len(inventory['files']), constructor_calls=sum(inventory['alias_counts'].values()),
                      files_with_lock_and_process=sum(bool(r['locks'] and r['process_calls']) for r in inventory['files']),
                      lock_regions=sum(len(r['lock_regions']) for r in rows),
                      direct_process_under_lock=report['direct_process_under_lock'],
                      lock_reachable_spawn_functions=report['lock_reachable_spawn_functions']), ensure_ascii=False))
