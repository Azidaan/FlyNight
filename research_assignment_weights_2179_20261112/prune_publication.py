"""Drop test TEMP from publication without deleting any local files."""
from common import *
import zipfile
old=read(OUT/'publication.json');fmt=read(OUT/'publication_formatting.json');prefix=OUT.relative_to(ROOT).as_posix()+'/'
def temporary(name):return name.startswith(prefix) and 'temp' in Path(name[len(prefix):]).parts
removed=[name for name in old['files'] if temporary(name)]
assert removed and not any(name.startswith(('tracking/','experiments/')) for name in removed)
files={name:value for name,value in old['files'].items() if not temporary(name)}
oldarchive=prefix+'frozen_source_bytes.zip';oldfmt=prefix+'publication_formatting.json';files.pop(oldarchive);files.pop(oldfmt)
archive=OUT/'frozen_source_bytes_v2.zip'
with zipfile.ZipFile(OUT/'frozen_source_bytes.zip') as source,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as target:
    for name in source.namelist():
        if not temporary(name):target.writestr(name,source.read(name))
kept=[x for x in fmt['files'] if not temporary(x['file'])]
save('publication_formatting_v2.json',dict(passed=True,archive_sha256=sha(archive),files=kept,frozen_working_files_unchanged=True,python_AST_and_JSON_exact=True,temp_fixtures_excluded=True))
for path in [archive,OUT/'publication_formatting_v2.json',OUT/'prune_publication.py',OUT/'verify_publication_v2.py']:
    files[path.relative_to(ROOT).as_posix()]=dict(source=str(path),sha256=sha(path))
save('publication_v2.json',dict(passed=True,head=old['head'],files=files,excluded_temporary_paths=removed,reason='OriginalpublicationdraftaccidentallyincludedpytestTEMP. ThisversionexcludesTEMPandusesanarchivewithoutfixtures; originaldraft/results/localTEMPremainpreserved. No scientificorgraphchange.'))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and not any(temporary(n) for n in z.namelist())
    for item in kept:assert hashlib.sha256(z.read(item['file'])).hexdigest()==item['frozen_sha256']
print('EXCLUDED_TEMP_PATHS',len(removed),'PUBLICATION_FILES',len(files),'ARCHIVE_BYTES',archive.stat().st_size)
