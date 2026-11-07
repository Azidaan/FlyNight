"""Изолировать проверенные зависимости; исходный setup.py не исполняется."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,ssl,tarfile,urllib.request,zipfile
import certifi
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tools/sam2-deps-20261027'
def main():
    OUT.mkdir(exist_ok=True);records=[]
    ctx=ssl.create_default_context(cafile=certifi.where())
    for package,version in [('hydra-core','1.3.2'),('omegaconf','2.3.0'),('antlr4-python3-runtime','4.9.3')]:
        data=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',context=ctx,timeout=20))
        wheels=[x for x in data['urls'] if x['filename'].endswith('.whl')]
        artifact=wheels[0] if wheels else next(x for x in data['urls'] if x['packagetype']=='sdist')
        assert artifact['size']<500_000
        blob=urllib.request.urlopen(artifact['url'],context=ctx,timeout=20).read(500001)
        assert len(blob)==artifact['size'] and hashlib.sha256(blob).hexdigest()==artifact['digests']['sha256']
        archived=OUT/'sources'/artifact['filename'];archived.parent.mkdir(exist_ok=True);archived.write_bytes(blob)
        if wheels:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for item in archive.infolist():
                    path=(OUT/item.filename).resolve();path.relative_to(OUT.resolve())
                    if item.is_dir():continue
                    assert item.file_size<5_000_000
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(item))
        else:
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
                for item in archive.getmembers():
                    parts=PurePosixPath(item.name).parts
                    assert '..' not in parts and not item.issym() and not item.islnk()
                    if not item.isfile():continue
                    if 'src' in parts and parts[parts.index('src')+1:][:1]==('antlr4',):
                        relative=Path(*parts[parts.index('src')+1:])
                    elif Path(item.name).name.upper().startswith(('LICENSE','NOTICE')):
                        relative=Path('licenses')/package/Path(item.name).name
                    else:continue
                    path=(OUT/relative).resolve();path.relative_to(OUT.resolve());path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(archive.extractfile(item).read())
        records.append(dict(package=package,version=version,source=artifact['url'],bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    assert (OUT/'antlr4/__init__.py').exists()
    (OUT/'provenance.json').write_text(json.dumps(dict(packages=records,main_environment_changed=False,setup_executed=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(records))
if __name__=='__main__':main()
