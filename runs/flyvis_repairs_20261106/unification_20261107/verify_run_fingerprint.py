"""Independent SHA-256 verifier; does not import the producer or neural libraries."""
from pathlib import Path
import hashlib,json,sys,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(1024**2)
            if not b:break
            h.update(b)
    return h.hexdigest()
def mismatches(base,expected):
    bad=[]
    for name,value in expected.items():
        path=(base/name).resolve();path.relative_to(base.resolve())
        if not path.is_file() or digest(path)!=value:bad.append(name)
    return bad
def main():
    out=(HERE/'uncapped_checks'/sys.argv[1]).resolve();out.relative_to((HERE/'uncapped_checks').resolve())
    expected=(out/'run_checksums.sha256').read_text().split()[0]
    assert digest(out/'run_checksums.json')==expected
    sums=json.loads((out/'run_checksums.json').read_bytes());protocol=json.loads((out/'run_protocol.json').read_bytes())
    assert sums['passed'] and sums['commit']==protocol['commit'] and sums['run_id']==out.name
    assert not mismatches(out,sums['outputs_sha256'])
    assert not mismatches(ROOT,protocol['source_sha256'])
    assert not mismatches(ROOT,protocol['input_sha256'])
    with zipfile.ZipFile(out/'frozen_sources.zip') as archive:
        assert set(archive.namelist())==set(protocol['source_sha256'])
        for name,h in protocol['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==h
    guard=json.loads((out/'memory_guard.json').read_bytes())
    assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors']
    assert json.loads((out/'pipeline_real_clip.json').read_bytes())['passed']
    # Deliberately wrong expected digest: prove mismatch reporting without changing any file.
    assert mismatches(out,{'run_protocol.json':'0'*64})==['run_protocol.json']
    subprocess.run(['git','cat-file','-e',protocol['commit']+'^{commit}'],cwd=ROOT,check=True)
    report=dict(passed=True,run_id=out.name,run_commit=protocol['commit'],manifest_sha256=expected,
                sources_checked=len(protocol['source_sha256']),inputs_checked=len(protocol['input_sha256']),
                outputs_checked=len(sums['outputs_sha256']),source_archive_checked=True,negative_digest_control_passed=True,
                minimum_available_gib=guard['minimum_available_bytes']/1024**3,
                verifier_sha256=digest(Path(__file__).resolve()),torch_imported='torch' in sys.modules)
    assert not report['torch_imported']
    (out/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
