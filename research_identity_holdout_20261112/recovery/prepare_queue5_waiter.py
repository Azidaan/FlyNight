from common import ROOT, OUT, read, sha, write, now
import ast
path = OUT / 'queue5_waiter.py'
source = '\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines()).rstrip() + '\n'
ast.parse(source)
path.write_bytes(source.encode('utf8'))
inputs = ['research_identity_holdout_20261112/queue5.json', 'research_identity_holdout_20261112/run_queue.py', 'research_identity_holdout_20261112/common.py']
write('queue5_wait_spec.json', dict(created=now(), inputs={name: sha(ROOT / name) for name in inputs}, waiter_sha256=sha(path),
                                   predecessor_execution='research_identity_holdout_20261112/queue4_execution.json', queue='research_identity_holdout_20261112/queue5.json',
                                   runner='research_identity_holdout_20261112/run_queue.py', maximum_seconds=1800,
                                   rule='Launch exactlyonce onlyafterqueue4completeANDpassed; existingqueue5execution=>no launch; stoponpredecessorfailure. Parent explicitlydelegatedsolequeue5ownership. HiddenIdleCPU, noGPU/fulltracker.'))
print(sha(OUT / 'queue5_wait_spec.json'))
