from common import *
import zipfile
source=OUT/'correspondence/result.json';digest=sha(source)
archive=OUT/'correspondence/full_result_frozen.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    z.write(source,'result.json')
assert sha(source)==digest
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('result.json')).hexdigest()==digest
save('correspondence/archive_receipt.json',dict(passed=True,source=source.relative_to(ROOT).as_posix(),source_sha256=digest,source_bytes=source.stat().st_size,archive=archive.relative_to(ROOT).as_posix(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,source_unchanged=True,reason='Preserve every full-domain Census cost tuple without a large uncompressed Git text blob. Neural arrays remain local and checksummed.'))
print('FULL_OUTPUT_ARCHIVED',source.stat().st_size,archive.stat().st_size)
