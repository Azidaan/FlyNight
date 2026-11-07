"""Изолированная публичная зависимость BUSCA; не меняет основное окружение."""
from pathlib import Path
import hashlib,json,urllib.request,ssl,zipfile
import certifi
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'tools/busca-deps-20261027'
def main():
    assert not TARGET.exists()
    context=ssl.create_default_context(cafile=certifi.where())
    metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/positional-encodings/6.0.3/json',context=context,timeout=25))
    wheel=next(x for x in metadata['urls'] if x['filename'].endswith('.whl'))
    assert wheel['size']<300000 and wheel['url'].startswith('https://files.pythonhosted.org/')
    data=urllib.request.urlopen(wheel['url'],context=context,timeout=25).read(300001)
    assert len(data)==wheel['size'] and hashlib.sha256(data).hexdigest()==wheel['digests']['sha256']
    TARGET.mkdir();archive=TARGET/wheel['filename'];archive.write_bytes(data)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            path=(TARGET/name).resolve();path.relative_to(TARGET.resolve())
        z.extractall(TARGET)
    (TARGET/'provenance.json').write_text(json.dumps(dict(package='positional-encodings',version='6.0.3',
        source=wheel['url'],bytes=len(data),sha256=wheel['digests']['sha256'],
        main_environment_changed=False),indent=2)+'\n',encoding='utf-8')
    print('Verified isolated wheel',len(data),wheel['digests']['sha256'])
if __name__=='__main__':main()
