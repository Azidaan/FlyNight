"""Preserve previous launchers and remove only their per-process caps."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
backup = root / 'memory_policy_previous'
backup.mkdir(exist_ok=False)
changes = {}

def revise(name, replacements):
    path = root / name
    before = path.read_bytes()
    (backup / name).write_bytes(before)
    text = before.decode('utf-8')
    for old, new in replacements:
        assert text.count(old) == 1, (name, old)
        text = text.replace(old, new)
    after = text.encode('utf-8')
    assert path.read_bytes() == before
    path.write_bytes(after)
    changes[name] = dict(before=hashlib.sha256(before).hexdigest(), after=hashlib.sha256(after).hexdigest())

revise('per_file.py', [
    ("if len(sys.argv)>1:", "if len(sys.argv) == 1:\n    raise SystemExit('Run one test file at a time; the old batch controller is retired.')\nif len(sys.argv)>1:"),
    ("    from tracking.windows_memory_job import MemoryJob\n    limit_gib=int(sys.argv[3]) if len(sys.argv)>3 else 2\n    assert limit_gib in (2,4)\n    job=MemoryJob(limit_gib*1024**3)", "    from system_memory_guard import guard_entrypoint\n    job=guard_entrypoint()\n    if len(sys.argv)>3:\n        print('Legacy per-process memory limit argument is ignored.',flush=True)"),
    ("test=sys.argv[1];temp=sys.argv[2]", "test=sys.argv[1];temp=str(job.run_dir/'pytest_temp')"),
    ("'MEMORY_JOB '", "'SYSTEM_MEMORY_POLICY '")])
revise('pipeline_smoke.py', [
    ("from tracking.windows_memory_job import MemoryJob\njob=MemoryJob(4*1024**3)", "from system_memory_guard import guard_entrypoint\njob=guard_entrypoint()\nOUT=job.run_dir")])
revise('manual_cpu.py', [
    ("from tracking.windows_memory_job import MemoryJob\njob=MemoryJob(3*1024**3)", "from system_memory_guard import guard_entrypoint\njob=guard_entrypoint()"),
    ("assert psutil.virtual_memory().available>=13*1024**3", "# System reserve is monitored independently by the supervisor."),
    ("out=Path(__file__).resolve().parent/(name+'_manual.json')", "out=job.run_dir/(name+'_manual.json')")])
revise('run_regression.py', [
    ('Sequential CPU regression groups with a Windows job memory ceiling.', 'Sequential CPU regression groups with a system-memory supervisor.'),
    ("from tracking.windows_memory_job import MemoryJob", "from system_memory_guard import guard_entrypoint\njob=guard_entrypoint()"),
    ("assert psutil.virtual_memory().available>=12*1024**3", "# The external supervisor checks total system memory."),
    ("limit=(4 if group=='native4' else 2)*1024**3\njob=MemoryJob(limit)", "limit=None"),
    ("out=OUT/group", "out=job.run_dir/group")])
# Retire old outer controllers: their fixed reserves and reports belong to old runs.
for name in ('watch_native.py', 'watch_smoke.py'):
    revise(name, [( (root/name).read_text(encoding='utf-8'),
        "raise SystemExit('Historical launcher retired. Run run_regression.py or pipeline_smoke.py directly; each now starts its own system-memory supervisor.')\n")])
(root/'memory_policy_change.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
print(json.dumps(changes,indent=2))
