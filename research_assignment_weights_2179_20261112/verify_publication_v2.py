from common import *
import subprocess,zipfile,psutil
p=read(OUT/'publication_v2.json');names=list(p['files'])
if len(sys.argv)>1 and sys.argv[1]=='stage':
    assert subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()==p['head']
    assert not subprocess.check_output(['git','diff','--cached','--name-only']).strip()
    rows=[]
    for name,value in p['files'].items():
        path=Path(value['source']);assert sha(path)==value['sha256'],name
        data=path.read_bytes() if path.suffix in ['.zip','.png','.jpg','.npz'] else path.read_bytes().replace(b'\r\n',b'\n')
        oid=subprocess.check_output(['git','hash-object','-w','--stdin'],input=data).decode().strip();rows.append(('100644 '+oid+'\t'+name+'\0').encode())
    subprocess.run(['git','update-index','-z','--index-info'],input=b''.join(rows),check=True);subprocess.run(['git','diff','--cached','--check'],check=True)
    print('STAGED_NO_TEMP',len(rows));raise SystemExit()
blob=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join('HEAD:'+n for n in names)+'\n').encode());offset=0
for name in names:
    end=blob.index(b'\n',offset);header=blob[offset:end].decode();offset=end+1;_,kind,size=header.split();assert kind=='blob';size=int(size);value=blob[offset:offset+size];offset+=size+1
    source=Path(p['files'][name]['source']);assert sha(source)==p['files'][name]['sha256'];expected=source.read_bytes()
    if source.suffix not in ['.zip','.png','.jpg','.npz']:expected=expected.replace(b'\r\n',b'\n')
    assert value==expected,name
fmt=read(OUT/'publication_formatting_v2.json');archive=OUT/'frozen_source_bytes_v2.zip';assert sha(archive)==fmt['archive_sha256']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in fmt['files']:assert hashlib.sha256(z.read(item['file'])).hexdigest()==item['frozen_sha256']
for n,h in read(OUT/'protocol.json')['inputs'].items():assert sha(ROOT/n)==h,n
assert sha(OUT/'active_settings.json')==sha(OUT/'original_settings.json')
q=read(OUT/'queue3_execution.json');assert q['passed'] and q['complete']
for job in q['jobs']:
    for pid,birth in job['result']['owned'].items():
        try:alive=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess:alive=False
        assert not alive,pid
save('publication_verification_v2.json',dict(passed=True,head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),checked_blobs=len(names),temporary_fixtures_excluded=True,archive_verified=True,all_scientific_inputs_unchanged=True,baseline_settings_restored=True,task_total_local_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),calculation_bytes=read(OUT/'storage_summary.json')['calculation_artifact_bytes'],no_cleanup_performed=True))
print('PUBLISHED_SCIENCE_GRAPH_ROLLBACK_NO_TEMP_VERIFIED',len(names))
