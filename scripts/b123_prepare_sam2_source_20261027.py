"""Официальные исходники SAM2 и размер нужных весов, без новых данных."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,ssl,urllib.request,sys
import certifi
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from b123_snapshot_recovery_sources_20261027 import fetch
OUT=ROOT/'tools/sam2-runtime-20261027'
WEIGHT_URL='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
def main():
    assert not (OUT/'provenance.json').exists()
    meta=json.loads(fetch('https://api.github.com/repos/facebookresearch/sam2/commits/main'))
    commit=meta['sha'];tree=json.loads(fetch('https://api.github.com/repos/facebookresearch/sam2/git/trees/'+commit+'?recursive=1'))
    assert not tree.get('truncated');selected=[]
    for item in tree['tree']:
        path=PurePosixPath(item['path']);name=path.as_posix()
        if item['type']=='blob' and ((name.startswith('sam2/') and path.suffix in {'.py','.yaml'}) or name in {'README.md','LICENSE','INSTALL.md','setup.py','pyproject.toml'}):
            assert not path.is_absolute() and '..' not in path.parts and item['size']<1_000_000
            selected.append((path,item))
    assert sum(x['size'] for _,x in selected)<8_000_000
    OUT.mkdir(parents=True,exist_ok=True)
    def save(pair):
        rel,item=pair;url='https://raw.githubusercontent.com/facebookresearch/sam2/'+commit+'/'+rel.as_posix()
        data=fetch(url,1_000_000);assert len(data)==item['size']
        target=OUT/Path(*rel.parts);target.resolve().relative_to(OUT.resolve());target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        return dict(path=target.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=url)
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(save,selected))
    request=urllib.request.Request(WEIGHT_URL,method='HEAD')
    with urllib.request.urlopen(request,context=ssl.create_default_context(cafile=certifi.where()),timeout=25) as response:
        size=int(response.headers['Content-Length']);assert 100_000_000<size<250_000_000
    (OUT/'provenance.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),repository='facebookresearch/sam2',
        commit=commit,files=files,weight_url=WEIGHT_URL,weight_bytes=size,
        purpose='Собственная масочная адаптация на SAM2.1tiny, не воспроизведение LTR(SAM+селектор+DeAOT) или SGAR-MOT.',
        models_executed=False,weights_downloaded=False,datasets_downloaded=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(commit=commit,files=len(files),source_bytes=sum(x['bytes'] for x in files),weight_bytes=size)))
if __name__=='__main__':main()
