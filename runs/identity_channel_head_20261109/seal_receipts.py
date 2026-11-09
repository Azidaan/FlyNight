"""Preserve closed execution receipts, including the graph schema correction."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'
stages=[('registration','4a0b2ed4dfa7432caf8a845f1431e018',0),
    ('mathematics','048c93c7bf6441e6ac8a2c26825927ea',0),
    ('snapshot_restore','7058971c0a7a4bb3ad58f4e84e340886',0),
    ('features','edc03a25b59d4b3dacceeb7137945d58',0),
    ('fit','a10fe3a64ede44debb1082ecd781ffa1',0),
    ('independent_audit','9bff3b31cdad42808506b0461a7882bf',0),
    ('records','72661efc645c4afabc68d0b4e657ac4c',0),
    ('catalog_pass_graph_schema_failed','95c44d5d2f354625abea60f8c1a4bac9',1),
    ('graph_corrected','1b370d44ebb3478ca9d8d811384902e4',0)]
files=[]
for stage,token,expected in stages:
    source=BASE/token;receipt=json.loads((source/'memory_guard.json').read_text(encoding='utf8'))
    assert receipt['returncode']==expected and not receipt['stop_errors'],stage
    dest=OUT/'closed_guards'/stage;dest.mkdir(parents=True,exist_ok=True)
    for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
        path=source/name
        if path.exists():
            target=dest/name;shutil.copyfile(path,target)
            files.append(dict(stage=stage,returncode=expected,source=path.relative_to(ROOT).as_posix(),copy=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(OUT/'closed_receipts.json').write_text(json.dumps(dict(passed=True,stages=len(stages),files=files,
    scientific_admission=False,all_scientific_stages_exited_zero=True,retained_technical_failure='The58 catalogue tests passed before graph generation failed on a missing literature field. Corrected schema, complete graph and audit then exited zero; no scientific rerun.'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Sealed',len(stages),'closed stages')
