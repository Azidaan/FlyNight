from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
ID='identity_weekly_causal_review_20261112'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
def guard():
    sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    guard_entrypoint(timeout=900)
def verify():
    for n,h in read(OUT/'evidence_manifest.json')['files'].items():assert sha(ROOT/n)==h,n
    for receipt in ['receipt.json','event_witnesses.json']:
        for n,h in read(OUT/receipt)['sources'].items():
            if n=='experiments/project_history.json':
                assert sha(OUT/'before/project_history.json')==h
            else:assert sha(ROOT/n)==h,n
    assert sha(OUT/'РАЗБОР.md')==read(OUT/'recording.json')['report_sha256']
