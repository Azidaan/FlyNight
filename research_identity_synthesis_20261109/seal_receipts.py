"""Retain closed guard receipts, including the repaired graph-build failure."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
base = ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'
stages = [('math','3adb871f13044564a2da789741a4179c',0),
          ('register','f9e1264c84ad40dfa2aae341acfd2d53',0),
          ('data','e84bcf21c8854e458542fabb5096f4a8',0),
          ('last_reference','cacdffa4763e489e8a4f142bb5563798',0),
          ('record','34703f99e57f449dbb82898db5c72d82',0),
          ('catalog_pass_graph_failed','19308ddfd46e4fc0bf32eb3c6098a52f',1),
          ('graph_corrected','a8c78a8f7dde4898aa8595959ef6eaba',0)]
sealed = []
for name, token, expected in stages:
    source = base/token
    receipt = json.loads((source/'memory_guard.json').read_text(encoding='utf8'))
    assert receipt['returncode'] == expected and not receipt['stop_errors'], name
    dest = HERE/'closed_guards'/name
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
        p = source/filename
        if p.exists():
            target = dest/filename
            shutil.copyfile(p,target)
            sealed.append(dict(stage=name,returncode=expected,source=p.relative_to(ROOT).as_posix(),
                copy=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
path=HERE/'closed_receipts.json'
path.write_text(json.dumps(dict(passed=True,closed=True,files=sealed,
    preserved_failure='Initial graph check failed on LF/CRLF document checksum after all 58 catalogue tests had passed. Corrected graph build and audit exited zero; no scientific rerun.'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Sealed',len(stages),'closed stages;',len(sealed),'files')
