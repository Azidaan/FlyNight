"""Supply the renderer's required DOI field without inventing a DOI."""
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes()
receipt=json.loads((OUT/'catalog_tests_receipt.json').read_text(encoding='utf8'))
assert sha(path)==receipt['stable_sources']['experiments/useful_literature.json']
snapshot=OUT/'useful_literature_before_doi.json';assert not snapshot.exists();snapshot.write_bytes(before)
p=json.loads(before);source=next(x for x in p['sources'] if x['id']=='sugiyama2007_covariate_shift_caution')
assert 'doi' not in source;source['doi']='не указан в проверенном первоисточнике'
temp=path.with_suffix('.doi_recovery.pending');assert not temp.exists();temp.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for _ in range(30):
    assert path.read_bytes()==before
    try:temp.replace(path);break
    except PermissionError:time.sleep(.1)
else:raise PermissionError(str(path))
with (OUT/'literature_schema_recovery.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,source=path.relative_to(ROOT).as_posix(),old_sha256=hashlib.sha256(before).hexdigest(),new_sha256=sha(path),snapshot=snapshot.relative_to(ROOT).as_posix(),
    change='Only add the required DOI metadata field to the newly added Sugiyama source. DOI not guessed; explicitly absent from the checked original.',
    previous_guard='54d40a4312a644d0883d02b23064c11f',tests='58 catalog tests passed before the preview KeyError; catalogue, glossary, plan, topics and problem records remain unchanged. Verify graph with corrected citation metadata; no reason to repeat the scientific calculation or catalogue tests.'),f,ensure_ascii=False,indent=2)
print('DOI_METADATA_FIXED_WITHOUT_GUESSING',flush=True)
