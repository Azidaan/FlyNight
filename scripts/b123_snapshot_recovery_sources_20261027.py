"""Сохранить только публичные исходники; сторонний код не исполняется."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib, json, ssl, urllib.request
from concurrent.futures import ThreadPoolExecutor
import certifi

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/b123_20261027/recovery_source_snapshot'


def fetch(url,limit=16_000_000):
    request=urllib.request.Request(url,headers={'User-Agent':'Dissertation-source-audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=30,context=ssl.create_default_context(cafile=certifi.where())) as response:
        data=response.read(limit+1)
    assert len(data)<=limit,url
    return data


def main():
    assert not (OUT/'manifest.json').exists()
    OUT.mkdir(exist_ok=True)
    files=[];repositories=[]
    for repo in ('lorenzovaquero/BUSCA','ManuelBGomez/SGAR-MOT'):
        commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main'))
        digest=commit['sha']
        repositories.append(dict(repository=repo,commit=digest,date=commit['commit']['committer']['date']))
        target_root=OUT/repo.split('/')[-1]
        tree=json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+digest+'?recursive=1'))
        assert not tree.get('truncated')
        selected=[]
        for item in tree['tree']:
            if item['type']!='blob':continue
            rel=PurePosixPath(item['path'])
            top=rel.as_posix()
            if repo.endswith('BUSCA'):
                keep=(top.startswith('busca/') and rel.suffix=='.py') or top in {'README.md','LICENSE','requirements.txt','config/ByteTrack/MOT17/config_bytetrack_mot17.yml','adapters/ByteTrack/yolox/tracker/byte_tracker.py'}
            else:
                keep=top in {'README.md','LICENSE','requirements.txt','pyproject.toml','docs/TUTORIAL.md','THIRD_PARTY.md'}
            if keep:
                assert '..' not in rel.parts and not rel.is_absolute() and item['size']<=2_000_000
                selected.append((rel,item))
        assert sum(x['size'] for _,x in selected)<=12_000_000
        def obtain(pair):
            rel,item=pair
            url='https://raw.githubusercontent.com/'+repo+'/'+digest+'/'+rel.as_posix()
            data=fetch(url,2_000_000)
            assert len(data)==item['size']
            path=(target_root/Path(*rel.parts)).resolve();path.relative_to(OUT.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert path.read_bytes()==data
            else:path.write_bytes(data)
            return dict(repository=repo,commit=digest,path=path.relative_to(ROOT).as_posix(),source=url,
                        sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        with ThreadPoolExecutor(max_workers=4) as executor:files.extend(executor.map(obtain,selected))
    manifest=dict(at=datetime.now(timezone.utc).isoformat(),repositories=repositories,files=files,
        source_only=True,models_executed=False,weights_downloaded=False,datasets_downloaded=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(repositories=repositories,files=len(files),bytes=sum(x['bytes'] for x in files))))


if __name__=='__main__':main()
