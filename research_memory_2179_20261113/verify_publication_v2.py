from common import *
import subprocess
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
manifest=read(OUT/'publication.json'); assert read(OUT/'checks_queue_execution.json')['passed']
staged=git('diff','--cached','--name-only','-z').decode().split('\0'); staged={x for x in staged if x}
assert staged <= set(manifest['files']), staged-set(manifest['files'])
# Manifest also contains intentionally unchanged tracked files. The exact byte
# comparison below verifies every requested index entry, including those files.
for name,item in manifest['files'].items():
    source=Path(item['source']); assert sha(source)==item['sha256'],name
    expected=source.read_bytes()
    if source.suffix.lower() not in ['.jpg','.png','.npz','.zip']: expected=expected.replace(b'\r\n',b'\n')
    assert git('show',':'+name)==expected,name
graph=read(OUT/'graph_check.json')
for rel,h in graph['protected_hypotheses'].items(): assert sha(ROOT/'ObsidianFlyVis'/rel)==h,rel
for name,h in read(OUT/'protocol.json')['inputs'].items(): assert sha(ROOT/name)==h,name
save('publication_verification_v2.json',dict(passed=True,staged_file_count=len(staged),source_bytes_match_index=True,protected_hypotheses_unchanged=True,registered_inputs_unchanged=True,checks=read(OUT/'checks.json')))
print('PUBLICATION_VERIFIED',len(staged))
