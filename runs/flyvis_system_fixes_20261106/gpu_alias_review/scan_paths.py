"""Read-only AST inventory; never construct a native mutex or import GPU code."""
from pathlib import Path
import ast,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
records=[];aliases=collections.Counter()
for directory in ('scripts','tracking'):
    for path in sorted((ROOT/directory).rglob('*.py')):
        source=path.read_text(encoding='utf-8-sig')
        if 'GpuLock' not in source and 'FOVEA_GPU_LOCK' not in source and 'gpu_lock' not in source:
            continue
        tree=ast.parse(source)
        parent={id(child):node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        def ancestors(node):
            while id(node) in parent:
                node=parent[id(node)];yield node
        def scope(node):
            return '.'.join(reversed([x.name for x in ancestors(node) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))])) or '<module>'
        locks=[];processes=[];threads=[];env=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                fn=ast.unparse(node.func)
                if fn.endswith('GpuLock'):
                    argument=ast.unparse(node.args[0]) if node.args else None
                    aliases[argument]+=1
                    locks.append(dict(line=node.lineno,scope=scope(node),argument=argument,statement=ast.get_source_segment(source,node)))
                if fn in ('subprocess.Popen','subprocess.run','subprocess.call','subprocess.check_call','subprocess.check_output') or fn.endswith(('.Process','ProcessPoolExecutor')):
                    processes.append(dict(line=node.lineno,scope=scope(node),call=fn,source=ast.get_source_segment(source,node),
                                          enclosing_with=[dict(line=x.lineno,contexts=[ast.unparse(i.context_expr) for i in x.items]) for x in ancestors(node) if isinstance(x,ast.With)]))
                if fn.endswith(('Thread','ThreadPoolExecutor')):
                    threads.append(dict(line=node.lineno,scope=scope(node),call=fn,source=ast.get_source_segment(source,node)))
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.Expr)):
                part=ast.get_source_segment(source,node)
                if part and 'FOVEA_GPU_LOCK' in part and len(part)<600:
                    env.append(dict(line=node.lineno,scope=scope(node),source=part))
        records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            locks=locks,process_calls=processes,thread_calls=threads,env_assignments=env))
report=dict(scope='Static direct named calls only; no module execution, GPU or mutex acquisition',alias_counts=dict(aliases),files=records)
(OUT/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(alias_counts=dict(aliases),both=[dict(path=r['path'],locks=[(x['line'],x['scope'],x['argument']) for x in r['locks']],
                      processes=[(x['line'],x['scope'],x['call'],x['enclosing_with']) for x in r['process_calls']],threads=r['thread_calls'])
                     for r in records if r['locks'] and (r['process_calls'] or r['thread_calls'])]),ensure_ascii=False,indent=2))
